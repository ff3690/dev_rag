# -*- coding: utf-8 -*-
"""
structured_splitter.py - 结构感知切片器（核心）
================================================

把结构化文档（StructuredDocument）切分为适合 RAG 向量化入库的 Chunk 列表。

与普通文本切片的区别（本方案的三大关键决策）：
    1. **整表完整切片**：表格不按行拆分，整张表作为一个块。
       理由：表格的行/列是"联合表达"，拆行后"随访频率"与"随访内容"
       就分家了，检索无法回答跨列问题。超长表才降级（截断/行分组）。
    2. **流程图箭头关系文本化**：流程图输出"路径文本"（每条分支一行），
       显式保留 →[条件] 箭头关系，保证"什么条件下转诊"这类问题可检索。
    3. **章节锚点注入**：每个块都带 section_path（章节路径）与块类型
       元数据，检索命中后可精确定位"这是哪部分第几节的内容"。

输出 Chunk 的 metadata（会随向量一起入库，供检索筛选与溯源）：
    block_type   - 内容类型（table/flowchart/paragraph/...）
    section_path - 章节路径
    page_no      - 页码
    table_id     - 表格块 id（整表关联）
    flow_id      - 流程图块 id
"""

from typing import List

from docstruct.schemas import Block, Chunk, ParseIssue, StructuredDocument
from docstruct.slice.tokenizer import estimate_tokens, truncate_to_tokens
from docstruct.slice.text_templates import (
    TABLE_TEMPLATE,
    FLOW_TEMPLATE,
    FLOW_WITH_CONTEXT_TEMPLATE,
    CHECKBOX_TEMPLATE,
    HEADING_TEMPLATE,
    IMAGE_TEMPLATE,
    PARAGRAPH_PREFIX,
)
from docstruct.report import W_TABLE_ROW_SPLIT
from docstruct.utils.logger import logger


class StructuredSplitter:
    """
    结构感知切片器。

    参数:
        chunk_size: 常规块的 token 上限（默认 512）
        chunk_overlap: 相邻正文块的重叠 token 数（默认 64）
        max_table_tokens: 表格块的 token 上限（默认 2000，允许超长）
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 max_table_tokens: int = 2000, embed_safe_tokens: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_tokens = max_table_tokens
        # 整表块超过该 token 阈值时，额外生成"行组伴随块"（默认 400 < bge 512 上限）
        self.embed_safe_tokens = embed_safe_tokens
        # 章节上下文缓存：{section_path: [paragraph 文本...]}，供流程图上下文合并
        self._section_paragraphs: dict = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def split_document(self, document_or_sd) -> List[Chunk]:
        """
        把文档的结构化产物切分为 Chunk 列表。

        参数:
            document_or_sd: backend 的 Document（含 .structure）或直接传
                            StructuredDocument（duck typing，二选一均可）

        返回:
            List[Chunk]: 切片列表（按阅读顺序）
        """
        # 兼容两种传参方式：Document.structure 或 StructuredDocument 本体
        sd = getattr(document_or_sd, "structure", document_or_sd)
        if not isinstance(sd, StructuredDocument):
            logger.warning("结构化切片器收到无效输入（缺少 .structure 或非 StructuredDocument）")
            return []

        print(f"\n[DOCSTRUCT][SLICE] ===== 开始结构感知切片: {sd.doc_id} =====")
        report = sd.report

        # 预收集：按章节路径缓存段落文本（供流程图上下文合并）
        self._section_paragraphs = {}
        for b in sd.blocks:
            if b.type == "paragraph" and b.section_path:
                self._section_paragraphs.setdefault(b.section_path, []).append(b.content)

        # 按块类型分派
        raw_chunks: List[Chunk] = []
        for block in sd.blocks:
            chunks = self._dispatch(block, report)
            raw_chunks.extend(chunks)

        # 回填块序号 / 总块数 / 位置信息
        chunks = self._finalize(raw_chunks)

        # 统计
        by_type: dict = {}
        for c in chunks:
            t = c.metadata.get("block_type", "?")
            by_type[t] = by_type.get(t, 0) + 1
        print(f"[DOCSTRUCT][SLICE] 切片完成: 共 {len(chunks)} 块, 类型分布: {by_type}")
        return chunks

    # ------------------------------------------------------------------
    # 分派
    # ------------------------------------------------------------------
    def _dispatch(self, block: Block, report) -> List[Chunk]:
        """按块类型分派到对应切片方法。"""
        if block.type == "table":
            return self._split_table(block, report)
        if block.type == "flowchart":
            return self._split_flowchart(block, report)
        if block.type == "checkbox":
            return self._split_checkbox(block)
        if block.type == "heading":
            return [self._make_chunk(self._render_heading(block), block,
                                     block_type="heading")]
        if block.type == "image":
            return [self._make_chunk(self._render_image(block), block,
                                     block_type="image")]
        if block.type == "paragraph":
            return self._split_paragraph(block)
        # 其他类型（页眉页脚等）不切片
        return []

    # ------------------------------------------------------------------
    # 表格切片（整表优先）
    # ------------------------------------------------------------------
    def _split_table(self, block: Block, report) -> List[Chunk]:
        """
        表格切片：整表完整切片为主，超长才降级。

        降级链：
            1. token <= chunk_size        → 1 块（最优）
            2. token <= max_table_tokens  → 1 块 + 截断行 + 标记（warning）
            3. 否则                       → 表头+上下文 1 块 + 行分组块（warning）

        参数:
            block: 表格块
            report: 质量报告（记录降级）

        返回:
            List[Chunk]
        """
        caption = block.data.get("caption", "")
        section_path = block.section_path or "未知章节"
        page_start = block.page_no
        page_end = block.data.get("page_end", page_start)
        leading = block.data.get("leading_text", "") or ""
        header = block.data.get("header", [])
        rows = block.data.get("rows", [])

        # 完整渲染（含上下文）
        leading_part = f"【说明】{leading}" if leading else ""
        full = TABLE_TEMPLATE.format(
            caption=caption, section_path=section_path,
            page_start=page_start, page_end=page_end,
            leading_part=leading_part,
            markdown_table=self._render_table(caption, header, rows),
        )
        tokens = estimate_tokens(full)

        # ---- 情况 1：正常整表 ----
        if tokens <= self.chunk_size:
            print(f"[DOCSTRUCT][SLICE] 表格整块: {caption}, {tokens}token")
            return [self._make_chunk(full, block, block_type="table", table_id=block.block_id)]

        # ---- 情况 2：允许超长的整表（<= max_table_tokens，完整保留） ----
        if tokens <= self.max_table_tokens:
            # 表格略超常规块大小，但仍在"允许超长"范围内 → **完整保留**。
            # 关键原因：表格的行/列是"联合表达"，拆行会割裂语义
            # （例如"随访频率"与"随访内容"分家后，检索无法回答跨列问题）。
            # 宁可块略大，也要保证整表可检索。max_table_tokens 就是为此放宽的。
            print(f"[DOCSTRUCT][SLICE] 表格超长完整保留: {caption}, "
                  f"{len(rows)}行, {tokens}token")
            chunks = [self._make_chunk(full, block, block_type="table",
                                       table_id=block.block_id)]
            # ---- 缺陷4：行组伴随块（embedding 安全补偿） ----
            # 整表 token 超过 embed_safe_tokens（默认 400）时，embedding 服务
            # （bge 上限 512）会在向量化时静默截断，大表后半部分的行不可检索。
            # 因此额外生成"表头 + 行分组"伴随块（带 table_id + row_group 标记），
            # 与整表块同 id 关联，保证任意行的数据在向量空间都可检索。
            if tokens > self.embed_safe_tokens:
                companion = self._split_table_by_rows(
                    block, caption, section_path, page_start, page_end,
                    leading, header, rows, report, warn=False)
                chunks.extend(companion)
                print(f"[DOCSTRUCT][SLICE] 表格伴随块已生成: {caption}, "
                      f"{len(companion)}块")
            return chunks

        # ---- 情况 3：行分组（表头+上下文块 + 若干行块） ----
        chunks = self._split_table_by_rows(block, caption, section_path,
                                           page_start, page_end, leading, header, rows, report)
        print(f"[DOCSTRUCT][SLICE] 超长表格行分组: {caption}, 拆为{len(chunks)}块")
        return chunks

    def _split_table_by_rows(self, block, caption, section_path, page_start,
                             page_end, leading, header, rows, report,
                             warn: bool = True) -> List[Chunk]:
        """
        超超长表格：表头 + 上下文组成前缀，行按 token 上限分组。

        参数:
            block: 表格块
            caption: 表题
            section_path: 章节路径
            page_start/page_end: 页码范围
            leading: 前导说明文本
            header: 表头
            rows: 数据行
            report: 质量报告
            warn: 是否记录行分组 warning（False 表示作为"整表伴随块"生成，不重复告警）

        返回:
            List[Chunk]
        """
        if warn:
            report.add(ParseIssue(
                level="warning", code=W_TABLE_ROW_SPLIT,
                message=f"超长表格行分组: {caption}",
                page_no=block.page_no, block_id=block.block_id,
            ))

        # 表头+上下文前缀（每个行块共用）
        header_block = self._render_table(caption, header, [])
        leading_part = f"【说明】{leading}" if leading else ""
        prefix = TABLE_TEMPLATE.format(
            caption=caption, section_path=section_path,
            page_start=page_start, page_end=page_end,
            leading_part=leading_part, markdown_table=header_block,
        )
        # 预留前缀 token 空间
        budget = self.chunk_size - estimate_tokens(prefix)
        budget = max(100, budget)

        chunks: List[Chunk] = []
        current_rows: List[list] = []
        current_tokens = 0
        for row in rows:
            row_text = self._render_row(header, row)
            row_tokens = estimate_tokens(row_text)
            if current_rows and current_tokens + row_tokens > budget:
                # 满块：输出
                md = self._render_table(caption, header, current_rows)
                text = TABLE_TEMPLATE.format(
                    caption=caption, section_path=section_path,
                    page_start=page_start, page_end=page_end,
                    leading_part=leading_part, markdown_table=md,
                )
                chunks.append(self._make_chunk(text, block, block_type="table",
                                               table_id=block.block_id,
                                               extra={"row_group": True}))
                current_rows = []
                current_tokens = 0
            current_rows.append(row)
            current_tokens += row_tokens

        # 最后一批
        if current_rows:
            md = self._render_table(caption, header, current_rows)
            text = TABLE_TEMPLATE.format(
                caption=caption, section_path=section_path,
                page_start=page_start, page_end=page_end,
                leading_part=leading_part, markdown_table=md,
            )
            chunks.append(self._make_chunk(text, block, block_type="table",
                                           table_id=block.block_id,
                                           extra={"row_group": True}))
        return chunks

    # ------------------------------------------------------------------
    # 流程图切片（三通道）
    # ------------------------------------------------------------------
    def _split_flowchart(self, block: Block, report) -> List[Chunk]:
        """
        流程图切片：输出"路径文本块"（核心）+ 可选"章节上下文合并块"。

        通道 a：路径文本（含箭头关系）—— 必出
        通道 b：路径文本 + 所属章节正文 —— 上下文增强（token 超限则跳过）

        参数:
            block: 流程图块
            report: 质量报告

        返回:
            List[Chunk]
        """
        title = block.data.get("title", "流程图")
        section_path = block.section_path or "未知章节"
        path_text = block.data.get("path_text", block.content)
        # 流程图区域图片（规则识别阶段已裁剪保存）。
        # 随切片元数据入库后，RAG 检索命中流程图块时可按 image_path + page_no
        # 定位到流程图对应的 PDF 页面。
        image_path = block.image_path or block.data.get("image_path", "")

        # ---- 通道 a：路径文本块（主检索块） ----
        text_a = FLOW_TEMPLATE.format(title=title, section_path=section_path,
                                      path_lines=path_text)
        # 缺陷5：置信度信号下沉到切片 metadata（规则通道低置信度 / LLM 复核来源）
        confidence = getattr(block, "confidence", None)
        low_confidence = confidence is not None and confidence < 0.6
        extra_a = {"image_path": image_path} if image_path else {}
        extra_a["confidence"] = confidence
        extra_a["low_confidence"] = low_confidence
        if block.data.get("rechecked_by_llm"):
            extra_a["rechecked_by_llm"] = True
            extra_a["rule_confidence"] = block.data.get("rule_confidence")
        chunks = [self._make_chunk(text_a, block, block_type="flowchart",
                                   flow_id=block.block_id, extra=extra_a)]

        # ---- 通道 b：章节上下文合并块 ----
        context = self._section_paragraphs.get(section_path, [])
        if context:
            context_text = "\n".join(context[:2])  # 最多取前两段正文，避免膨胀
            text_b = FLOW_WITH_CONTEXT_TEMPLATE.format(
                title=title, section_path=section_path,
                path_lines=path_text, context_text=context_text,
            )
            if estimate_tokens(text_b) <= self.chunk_size * 2:
                extra_b = {"with_context": True}
                if image_path:
                    extra_b["image_path"] = image_path
                chunks.append(self._make_chunk(
                    text_b, block, block_type="flowchart", flow_id=block.block_id,
                    extra=extra_b,
                ))
                print(f"[DOCSTRUCT][SLICE] 流程图上下文块已生成: {title}")

        print(f"[DOCSTRUCT][SLICE] 流程图切片: {title}, {len(chunks)}块")
        return chunks

    # ------------------------------------------------------------------
    # 复选框 / 标题 / 图片 / 正文
    # ------------------------------------------------------------------
    def _split_checkbox(self, block: Block) -> List[Chunk]:
        """复选框切片：条目清单 + 章节路径。"""
        items = block.data.get("items", [])
        lines = []
        for it in items:
            mark = "[x]" if it["checked"] else "[ ]"
            lines.append(f"{mark} {it['text']}")
        text = CHECKBOX_TEMPLATE.format(
            section_path=block.section_path or "未知章节",
            items_lines="\n".join(lines),
        )
        return [self._make_chunk(text, block, block_type="checkbox")]

    def _split_paragraph(self, block: Block) -> List[Chunk]:
        """
        正文切片：段落按 token 累积切块（带重叠）。

        规则：
            1. 一个 Block 是一段（可能含多行），先按段落文本处理；
            2. 超过 chunk_size 的段落按"句号/换行"边界二次切分；
            3. 每块前缀注入【所属章节】。

        参数:
            block: 段落块

        返回:
            List[Chunk]
        """
        text = block.content.strip()
        if not text:
            return []

        section_path = block.section_path or "未知章节"
        prefix = PARAGRAPH_PREFIX.format(section_path=section_path)
        # 预留前缀空间
        budget = self.chunk_size - estimate_tokens(prefix)
        budget = max(100, budget)

        # 若整段不超限 → 单块
        if estimate_tokens(text) <= budget:
            return [self._make_chunk(prefix + text, block, block_type="paragraph")]

        # 超限 → 按句边界二次切分
        segments = self._split_long_text(text, budget)
        return [self._make_chunk(prefix + seg, block, block_type="paragraph")
                for seg in segments]

    def _split_long_text(self, text: str, budget: int) -> List[str]:
        """
        把长文本按"句号/分号/换行"边界切成若干段，每段 token <= budget。

        参数:
            text: 长文本
            budget: 每段 token 预算

        返回:
            List[str]
        """
        import re
        # 切分单位：句号、分号、换行（保留边界避免语义割裂）
        parts = re.split(r"(?<=[。；;])\s*|\n+", text)
        parts = [p.strip() for p in parts if p.strip()]

        segments: List[str] = []
        current = ""
        for part in parts:
            if not current:
                current = part
                continue
            if estimate_tokens(current + part) <= budget:
                current += part
            else:
                segments.append(current)
                current = part
        if current:
            segments.append(current)
        # 极端情况：单句超长（无合适边界）→ 直接截断
        result = []
        for seg in segments:
            if estimate_tokens(seg) > budget:
                result.append(truncate_to_tokens(seg, budget))
            else:
                result.append(seg)
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _render_heading(self, block: Block) -> str:
        """渲染标题锚点文本。"""
        level = int(block.data.get("level", 2))
        hashes = "#" * min(level, 6)
        return HEADING_TEMPLATE.format(
            hashes=hashes, title=block.content,
            section_path=block.section_path or "未知章节",
        )

    def _render_image(self, block: Block) -> str:
        """渲染图片文本（OCR 文本 + LLM 解读 + 位置）。"""
        ocr = block.data.get("ocr_text", "")
        llm = block.data.get("llm_description", "")
        ocr_part = f"【OCR识别】\n{ocr}" if ocr else "（无 OCR 文本）"
        llm_part = f"\n【AI解读】\n{llm}" if llm else ""
        return IMAGE_TEMPLATE.format(
            section_path=block.section_path or "未知章节",
            ocr_text_part=ocr_part,
            llm_part=llm_part,
        )

    @staticmethod
    def _render_table(caption: str, header: list, rows: list) -> str:
        """渲染 Markdown 表格（供模板填充）。"""
        def esc(cell) -> str:
            return str(cell).replace("|", "\\|")

        lines = []
        if header:
            lines.append("| " + " | ".join(esc(c) for c in header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            padded = list(row) + [""] * (len(header) - len(row)) if header else row
            lines.append("| " + " | ".join(esc(c) for c in padded) + " |")
        return "\n".join(lines)

    @staticmethod
    def _render_row(header: list, row: list) -> str:
        """渲染单行（用于 token 预算估算）。"""
        return " | ".join(str(c) for c in row)

    def _make_chunk(self, content: str, block: Block, block_type: str,
                    table_id: str = "", flow_id: str = "", extra: dict = None) -> Chunk:
        """
        构造 Chunk（统一组装 metadata）。

        参数:
            content: 切片文本
            block: 来源内容块
            block_type: 块类型标记
            table_id: 表格块 id（整表关联，默认空）
            flow_id: 流程图块 id
            extra: 附加元数据

        返回:
            Chunk
        """
        metadata = {
            "block_type": block_type,
            "section_path": block.section_path or "",
            "page_no": block.page_no,
        }
        if table_id:
            metadata["table_id"] = table_id
        if flow_id:
            metadata["flow_id"] = flow_id
        if extra:
            metadata.update(extra)
        return Chunk(
            content=content,
            chunk_index=0,       # 最终在 _finalize 统一回填
            total_chunks=0,
            metadata=metadata,
        )

    def _finalize(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        回填 chunk_index / total_chunks / start_pos / end_pos。

        参数:
            chunks: 切片列表

        返回:
            List[Chunk]: 回填后的切片列表
        """
        total = len(chunks)
        cursor = 0
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = total
            chunk.start_pos = cursor
            cursor += len(chunk.content)
            chunk.end_pos = cursor
        return chunks
