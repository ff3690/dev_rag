# -*- coding: utf-8 -*-
"""
md_structured_splitter.py - Markdown 结构感知切片器
====================================================

对 MdStruct 块级结构做"结构感知切片"，语义对齐 docstruct/StructuredSplitter：

    table     -> 整表完整切片；超限时"表头 + 行分组"（共享 table_id）；
                 超过 embedding 安全 token 的整表额外生成"行组伴随块"补偿。
    code      -> 整块不切（mermaid 天然完整；bge 截断由 embedding 服务兜底）。
    image     -> 图意文本（alt + OCR + VL + 位置）独立成块。
    heading   -> 章节锚点块。
    paragraph/list/blockquote -> 章节前缀"累积切块"；超限按句子/子句二次切。

输出直接为 backend TextChunk（本模块从 text_splitter 惰性导入，避免环依赖）。
"""

from typing import List, Optional

from .markdown_struct import MdBlock, MdStruct
from .markdown_text_templates import (
    PARAGRAPH_PREFIX,
    render_block_to_text,
    render_table_markdown,
    render_image_text,
    estimate_tokens,
    split_long_text,
)


class MdStructuredSplitter:
    """
    Markdown 结构化文档切片器。

    参数:
        chunk_size:      正文块 token 预算（默认 512，对齐 STRUCTURED_PARSER_CHUNK_SIZE）
        chunk_overlap:   二次切分时的重叠 token（默认 64）
        max_table_tokens:整表允许的 token 上限（默认 2000，对齐 PDF）
        embed_safe_tokens: embedding 安全 token（默认 400 < bge 512 上限）
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 max_table_tokens: int = 2000, embed_safe_tokens: int = 400):
        self.chunk_size = max(64, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.max_table_tokens = max(self.chunk_size, max_table_tokens)
        self.embed_safe_tokens = max(32, embed_safe_tokens)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def split_structure(self, document) -> List:
        """
        对带 MdStruct 的 Document 执行结构感知切片。

        参数:
            document: Document 对象（structure 为 MdStruct，id/file_path 已填充）

        返回:
            List[TextChunk]；structure 异常时返回空列表（由门面降级）。
        """
        structure = getattr(document, "structure", None)
        if not isinstance(structure, MdStruct):
            return []
        doc_id = getattr(document, "id", "") or structure.doc_id or "md"
        structure.doc_id = doc_id
        blocks = structure.blocks or []
        docs = []  # 内部临时结构，最后统一回填 index/total
        table_seq = 0

        i = 0
        n = len(blocks)
        while i < n:
            block = blocks[i]
            btype = block.type

            if btype == "heading":
                docs.append(self._make(block, render_block_to_text(block),
                                       block_type="heading", start=block, end=block))
            elif btype == "code":
                text = render_block_to_text(block)
                if text.strip():
                    docs.append(self._make(
                        block, text, block_type="code", start=block, end=block,
                        extra={"code_lang": (block.language or "plain").lower()}))
            elif btype == "table":
                table_seq += 1
                docs.extend(self._split_table(block, table_seq, doc_id))
            elif btype == "image":
                docs.extend(self._split_image(block))
            elif btype in ("paragraph", "list", "blockquote"):
                # 累积连续同类正文块，减少碎片化
                j = i + 1
                while j < n and blocks[j].type in ("paragraph", "list", "blockquote"):
                    j += 1
                docs.extend(self._split_text_run(blocks[i:j]))
                i = j
                continue
            # 未知类型：兜底当正文
            else:
                text = (block.content or "").strip()
                if text:
                    docs.append(self._make(block, text, block_type=btype,
                                           start=block, end=block))
            i += 1

        return self._finalize(docs)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _make(block: MdBlock, content: str, block_type: str,
              start: MdBlock, end: MdBlock, extra: Optional[dict] = None) -> dict:
        meta = {"block_type": block_type}
        if start.section_path:
            meta["section_path"] = start.section_path
        if extra:
            meta.update(extra)
        return {
            "content": content,
            "start_pos": start.start_offset,
            "end_pos": end.end_offset,
            "metadata": meta,
        }

    @staticmethod
    def _finalize(docs: List[dict]):
        # 惰性导入 TextChunk，避免与 text_splitter 环依赖
        from backend.rag.text_splitter import TextChunk
        total = len(docs)
        return [TextChunk(
            content=d["content"],
            chunk_index=idx,
            total_chunks=total,
            start_pos=d["start_pos"],
            end_pos=d["end_pos"],
            metadata=d["metadata"],
        ) for idx, d in enumerate(docs)]

    # ------------------------------------------------------------------
    # 图片块
    # ------------------------------------------------------------------
    def _split_image(self, block: MdBlock) -> List[dict]:
        out = []
        for img in block.images:
            text = render_image_text(img, block.section_path)
            if not text.strip():
                continue
            meta = {
                "block_type": "image",
                "has_image": True,
                "image_urls": [img.url] if img.url else [],
                "enhanced_by": img.enhanced_by or "none",
                "alt": img.alt,
            }
            if block.section_path:
                meta["section_path"] = block.section_path
            out.append({
                "content": text,
                "start_pos": block.start_offset,
                "end_pos": block.end_offset,
                "metadata": meta,
            })
        return out

    # ------------------------------------------------------------------
    # 表格块（整表完整 + 超限降级 + 伴随块补偿）
    # ------------------------------------------------------------------
    def _table_md(self, block: MdBlock, rows) -> str:
        return render_table_markdown(block.header, rows, caption="")

    def _table_header_part(self, block: MdBlock, caption: str, section_path: str,
                           leading: str) -> List[str]:
        parts = [f"【表格】{caption or '表格'}"]
        if section_path:
            parts.append(f"【所属章节】{section_path}")
        if leading:
            parts.append(leading)
        return parts

    def _split_table(self, block: MdBlock, table_seq: int,
                     doc_id: str = "md") -> List[dict]:
        caption = block.caption or ""
        section_path = block.section_path or ""
        leading = block.leading_text or ""
        table_id = f"{doc_id}_t{table_seq}"

        header = block.header or []
        rows = block.rows or []

        def _compose(group_rows) -> str:
            parts = self._table_header_part(block, caption, section_path, leading)
            md = self._table_md(block, group_rows)
            if md:
                parts.append(md)
            return "\n".join(p for p in parts if p)

        full = _compose(rows)
        full_tokens = estimate_tokens(full)

        if full_tokens <= self.chunk_size:
            return [self._make(block, full, "table", block, block,
                               extra={"table_id": table_id})]

        # ---- 超长：行分组伴随/降级 ----
        total_rows = len(rows)
        groups: List[List[List[str]]] = []
        cur: List[List[str]] = []
        cur_tokens = self.embed_safe_tokens + 1
        budget = self.chunk_size

        for row in rows:
            row_text = self._table_md(block, [row])
            row_tokens = estimate_tokens(row_text)
            # 每个分组内至少放一行
            if cur and cur_tokens + row_tokens > budget:
                groups.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(row)
            cur_tokens += row_tokens
        if cur:
            groups.append(cur)

        chunks = []
        # 情况 1：整表可容忍（<= max_table_tokens）：保留完整块 + 行组伴随块（缺陷4补偿）
        if full_tokens <= self.max_table_tokens:
            chunks.append(self._make(block, full, "table", block, block,
                                     extra={"table_id": table_id}))
            for gi, group in enumerate(groups):
                gtext = _compose(group)
                chunks.append(self._make(
                    block, gtext, "table", block, block,
                    extra={"table_id": table_id, "row_group": gi + 1,
                           "row_group_total": len(groups)}))
        else:
            # 情况 2：整表超上限：仅"表头 + 行分组"（共享 table_id）
            header_only = self._make(
                block,
                "\n".join(self._table_header_part(block, caption, section_path, leading)
                          + ([self._table_md(block, [])] if header else [])),
                "table", block, block,
                extra={"table_id": table_id, "row_group": 0})
            if header:
                chunks.append(header_only)
            for gi, group in enumerate(groups):
                gtext = _compose(group)
                chunks.append(self._make(
                    block, gtext, "table", block, block,
                    extra={"table_id": table_id, "row_group": gi + 1,
                           "row_group_total": len(groups)}))
        return chunks

    # ------------------------------------------------------------------
    # 正文块（段落/列表/引用：章节前缀累积切块）
    # ------------------------------------------------------------------
    def _split_text_run(self, blocks: List[MdBlock]) -> List[dict]:
        if not blocks:
            return []
        section_path = blocks[0].section_path or ""
        # 汇总所有块的图片（内嵌图意文本拼入正文）
        urls: List[str] = []
        texts: List[str] = []
        for blk in blocks:
            if blk.content.strip():
                texts.append(blk.content.strip())
            for img in blk.images:
                txt = render_image_text(img, section_path)
                if txt.strip():
                    texts.append(txt.strip())
                if img.url:
                    urls.append(img.url)

        body = "\n\n".join(texts)
        if not body.strip():
            return []

        # 汇总元数据：run 内出现任何"清单"则整体标 list（优先于引用/正文）
        if any(b.type == "list" for b in blocks):
            block_type = "list"
        elif all(b.type == "blockquote" for b in blocks):
            block_type = "blockquote"
        else:
            block_type = "paragraph"

        meta = {"block_type": block_type}
        if section_path:
            meta["section_path"] = section_path
        if urls:
            meta["has_image"] = True
            meta["image_urls"] = urls
            if len(blocks) == 1 and blocks[0].images and len(blocks[0].images) == 1:
                meta["enhanced_by"] = blocks[0].images[0].enhanced_by or "none"

        start = blocks[0]
        end = blocks[-1]
        prefix_tokens = estimate_tokens(PARAGRAPH_PREFIX.format(section_path=section_path)) if section_path else 0
        budget = self.chunk_size - prefix_tokens

        if estimate_tokens(body) <= budget:
            content = (PARAGRAPH_PREFIX.format(section_path=section_path) + body) if section_path else body
            return [{"content": content, "start_pos": start.start_offset,
                     "end_pos": end.end_offset, "metadata": dict(meta)}]

        # 超限：按句子/子句二次切分（split_long_text 会重复注入前缀）
        pieces = split_long_text(body, max_tokens=budget,
                                 max_overlap_tokens=self.chunk_overlap,
                                 section_path=section_path)
        return [{"content": p, "start_pos": start.start_offset,
                 "end_pos": end.end_offset, "metadata": dict(meta)} for p in pieces]


def split_md_structure(document) -> List:
    """便捷入口：结构感知切片（MdStruct）或 []。"""
    return MdStructuredSplitter().split_structure(document)
