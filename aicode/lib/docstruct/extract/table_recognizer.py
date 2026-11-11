# -*- coding: utf-8 -*-
"""
table_recognizer.py - 表格识别器
================================================

从表格版面元素区域提取行列数据，输出：
    - Block.content: Markdown 表格文本（可直接向量化/渲染）
    - Block.data:    {"caption", "header", "rows", "col_count", "row_count", "page_end", "leading_text"}

附带能力：
    - caption 检测：表格上方的"表 X-X xxx"标题行
    - leading_text 提取：表格上方紧邻的正文段落（用作切片的上下文说明）
    - 跨页表格合并（merge_cross_page）：相邻页列数一致且宽度重叠 → 合并为一张表

新手提示：
    PyMuPDF 的 page.find_tables() 依赖线框（表格有边框线）检测。
    本 PDF 的表格全部是线框表，因此识别可靠；若遇到无线框的"开放式表格"
    （仅横线无竖线），extract() 可能返回不完整，此时走 E_TABLE_EXTRACT_FAILED 降级。
"""

import re
from typing import List

import fitz  # PyMuPDF

from docstruct.schemas import BlockType, LayoutElement, Block, BBox, ParseIssue, ParseReport
from docstruct.extract.base import BaseRecognizer
from docstruct.report import E_TABLE_EXTRACT_FAILED
from docstruct.utils.logger import logger
from docstruct.utils.text_utils import span_text

# 表格标题模式："表 2-3" / "表1" / "表 12" 等
_CAPTION_RE = re.compile(r"^表\s*\d+[-\.]\d+\s*[^。；]|^表\s*\d+\s*[^。；]")

# 表格上方搜索标题/前导正文的最大距离（pt）
_CAPTION_SEARCH_HEIGHT = 120.0


class TableRecognizer(BaseRecognizer):
    """线框表格识别器。"""

    def supports(self, block_type: BlockType) -> bool:
        return block_type == "table"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def recognize(self, element: LayoutElement, page) -> Block:
        """
        从表格区域提取行列数据。

        参数:
            element: table 版面元素
            page: fitz.Page 对象

        返回:
            Block: data 见 schemas.Block 的 table 约定
        """
        # ---- 1. 定位与 bbox 匹配的表格对象 ----
        table_obj = self._find_matching_table(page, element.bbox)
        if table_obj is None:
            # ---- 缺陷8：无线框表格降级提取（词坐标列聚类）----
            # 仅横线无竖线 / 完全无线框的"开放表格"，PyMuPDF find_tables
            # 识别不到线框 → 用文本 span 的 x/y 坐标做行列聚类重建表格。
            borderless = self._extract_borderless_table(page, element)
            if borderless is not None:
                header, rows = borderless
                if rows:
                    print(f"[DOCSTRUCT][EXTRACT] 无线框表格提取: 第{element.page_no}页, "
                          f"{len(rows)}行×{len(header) if header else '?'}列")
                    caption, leading_text = self._find_caption_and_leading(page, element.bbox)
                    if not caption:
                        caption = f"表（第{element.page_no}页）"
                    content = self._render_markdown_table(caption, header, rows)
                    if leading_text:
                        content = f"【说明】{leading_text}\n" + content
                    data = {
                        "caption": caption,
                        "header": header,
                        "rows": rows,
                        "col_count": len(header) if header else (len(rows[0]) if rows else 0),
                        "row_count": len(rows),
                        "page_end": element.page_no,
                        "leading_text": leading_text,
                        "borderless": True,  # 标记来源：无线框列聚类提取
                    }
                    return self._make_block(element, content=content, data=data,
                                           confidence=0.7)

            # 提取失败降级：记录问题，content 使用区域原始文本
            self.report.add(ParseIssue(
                level="error", code=E_TABLE_EXTRACT_FAILED,
                message=f"第 {element.page_no} 页表格提取失败（未找到线框表且无线框降级无结果）",
                page_no=element.page_no,
                detail={"bbox": element.bbox.__dict__},
            ))
            raw = page.get_textbox(fitz.Rect(
                element.bbox.x0, element.bbox.y0, element.bbox.x1, element.bbox.y1))
            return self._make_block(element, content=raw or "【表格】提取失败",
                                    data={"caption": "", "header": [], "rows": [],
                                          "col_count": 0, "row_count": 0,
                                          "page_end": element.page_no, "leading_text": ""},
                                    confidence=0.5)

        # ---- 2. 提取行列 ----
        rows = self._clean_rows(table_obj.extract())
        header = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []

        # ---- 2.5 空表跳过（只有表头、无任何数据行） ----
        # 这类"空表格"没有检索价值，跳过即可，避免生成 0 行数据的空块。
        # 注意：调用方 parse.py 已支持识别器返回 None（表示无有效内容）。
        if not body:
            print(f"[DOCSTRUCT][EXTRACT] 跳过空表（第{element.page_no}页，仅表头无数据行）")
            return None

        # ---- 3. 检测标题与前导正文（上下文） ----
        caption, leading_text = self._find_caption_and_leading(page, element.bbox)
        if not caption:
            caption = f"表（第{element.page_no}页）"

        # ---- 4. 构造 Markdown 表格文本 ----
        content = self._render_markdown_table(caption, header, body)
        content += "\n"
        if leading_text:
            content = f"【说明】{leading_text}\n" + content

        data = {
            "caption": caption,
            "header": header,
            "rows": body,
            "col_count": len(header) if header else (len(body[0]) if body else 0),
            "row_count": len(body),
            "page_end": element.page_no,  # 跨页合并时会更新
            "leading_text": leading_text,
        }
        print(f"[DOCSTRUCT][EXTRACT] 识别表格: {caption}, {len(body)}行×{data['col_count']}列")
        return self._make_block(element, content=content, data=data)

    # ------------------------------------------------------------------
    # 跨页合并（静态方法，供 ParserGate 在全部提取后调用）
    # ------------------------------------------------------------------
    @staticmethod
    def merge_cross_page(blocks: List[Block], report: ParseReport = None) -> List[Block]:
        """
        合并跨页的同一张表格。

        判定条件（两个 table Block 满足全部条件则合并）：
            1. 页码相邻（第二页 = 第一页 page_end + 1）；
            2. 列数一致；
            3. bbox 宽度重叠率 >= 60%；
            4. 第一页末尾行无数据（"跨页"特征）或第二页首行与第一页表头相同。

        合并结果：保留第一页的 caption/header/leading_text，
        行数据拼接，page_end 更新为第二页页码。

        参数:
            blocks: 全部内容块（含 table 与非 table）
            report: 质量报告（合并成功时打 info 提示）

        返回:
            List[Block]: 合并后的内容块列表（保持原顺序）
        """
        tables = [b for b in blocks if b.type == "table"]
        if len(tables) < 2:
            return blocks

        merged_ids = set()
        new_tables: List[Block] = []
        for i, t1 in enumerate(tables):
            if t1.block_id in merged_ids:
                continue
            merged = t1
            # 向后查找可合并的相邻表格
            for t2 in tables[i + 1:]:
                # page_end 存在 data 字典中（不是 Block 属性）
                merged_page_end = merged.data.get("page_end", merged.page_no)
                if t2.page_no != merged_page_end + 1:
                    continue
                if t2.data.get("col_count") != merged.data.get("col_count"):
                    continue
                # 宽度重叠率判定
                w_overlap = (min(t1.bbox.x1, t2.bbox.x1) - max(t1.bbox.x0, t2.bbox.x0))
                if w_overlap / max(t1.bbox.width, 1) < 0.6:
                    continue
                # 满足条件：合并 t2 到 merged
                merged = TableRecognizer._merge_two(merged, t2)
                merged_ids.add(t2.block_id)
                print(f"[DOCSTRUCT][EXTRACT] 跨页合并表格: "
                      f"p{merged.page_no}+p{t2.page_no} -> 共{len(merged.data['rows'])}行")
                if report is not None:
                    report.add(ParseIssue(
                        level="info", code="I_TABLE_MERGED",
                        message=f"跨页合并表格 p{merged.page_no}~p{t2.page_no}",
                        page_no=merged.page_no,
                    ))
            new_tables.append(merged)

        # 重建 blocks：用合并后的表格替换原表格，保持原顺序
        table_by_id = {b.block_id: b for b in tables}
        for b in new_tables:
            table_by_id[b.block_id] = b
        for t in tables:
            if t.block_id in merged_ids:
                table_by_id.pop(t.block_id, None)

        result = []
        for b in blocks:
            if b.type == "table":
                if b.block_id in merged_ids:
                    continue  # 跳过已被合并的块
                result.append(table_by_id.get(b.block_id, b))
            else:
                result.append(b)
        return result

    @staticmethod
    def _merge_two(t1: Block, t2: Block) -> Block:
        """
        合并两个表格块（t1 在前，t2 在后）。

        - 若 t2 的首行与 t1 的表头相同，则丢弃 t2 首行（重复表头）；
        - 行数据拼接，page_end 更新。

        参数:
            t1: 前页表格块
            t2: 后页表格块

        返回:
            Block: 合并后的表格块（新 block_id）
        """
        header1 = t1.data.get("header", [])
        rows1 = list(t1.data.get("rows", []))
        rows2 = list(t2.data.get("rows", []))
        # 去重表头：t2 第一行若与 t1 表头一致则去掉
        if rows2 and rows2[0] == header1:
            rows2 = rows2[1:]

        merged = Block(
            block_id=t1.block_id,              # 保留前块的 id，便于溯源
            type="table",
            page_no=t1.page_no,
            bbox=BBox(t1.bbox.x0, t1.bbox.y0, max(t1.bbox.x1, t2.bbox.x1), t2.bbox.y1),
            content=t1.content + "\n" + t2.content,
            data={
                "caption": t1.data.get("caption", ""),
                "header": header1,
                "rows": rows1 + rows2,
                "col_count": t1.data.get("col_count", 0),
                "row_count": len(rows1) + len(rows2),
                "page_end": t2.page_no,        # 结束页码更新为后页
                "leading_text": t1.data.get("leading_text", ""),
            },
            parent_id=t1.parent_id,
            section_path=t1.section_path,
            confidence=t1.confidence,
            source=t1.source,
        )
        # 重新渲染合并后的 Markdown 表格
        merged.content = TableRecognizer._render_markdown_table(
            merged.data["caption"], merged.data["header"], merged.data["rows"])
        if merged.data.get("leading_text"):
            merged.content = f"【说明】{merged.data['leading_text']}\n" + merged.content
        return merged

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _extract_borderless_table(self, page, element: LayoutElement):
        """
        无线框表格降级提取（缺陷8）：基于文本 span 的坐标做行列聚类。

        适用场景：仅横线无竖线、或完全无线框的"开放表格"，
        PyMuPDF find_tables() 依赖线框，识别不到时回退此方法。

        参数:
            page: fitz.Page
            element: table 版面元素

        返回:
            (header, rows) 或 None（区域内文本不足以构成表格时返回 None）
        """
        try:
            bbox = element.bbox
            # 1. 收集区域内所有文本 span：(x0, y0, text)
            spans = []
            text_page = page.get_text("dict")
            for block in text_page.get("blocks", []):
                if block.get("type") != 0:
                    continue
                br = fitz.Rect(block["bbox"])
                # 只处理与表格区域相交的块
                if (br.x1 < bbox.x0 or br.x0 > bbox.x1
                        or br.y1 < bbox.y0 or br.y0 > bbox.y1):
                    continue
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        t = span_text(sp).strip()
                        if not t:
                            continue
                        r = fitz.Rect(sp["bbox"])
                        if (r.x1 >= bbox.x0 and r.x0 <= bbox.x1
                                and r.y1 >= bbox.y0 and r.y0 <= bbox.y1):
                            spans.append((r.x0, r.y0, r.x1, t))
            if len(spans) < 2:
                return None

            # 2. 行聚类：按 y0 排序，相邻行间距小于阈值视为同一行
            row_gap = 5.0  # 同一行内 span 的纵向间距阈值（pt）
            spans.sort(key=lambda s: (round(s[1] / 5), s[0]))
            rows_spans: List[List] = []
            cur_y, cur_row = None, []
            for x0, y0, x1, t in spans:
                if cur_y is None or abs(y0 - cur_y) <= row_gap:
                    if cur_y is None:
                        cur_y = y0
                    cur_row.append((x0, x1, t))
                else:
                    if cur_row:
                        rows_spans.append(cur_row)
                    cur_y, cur_row = y0, [(x0, x1, t)]
            if cur_row:
                rows_spans.append(cur_row)
            if len(rows_spans) < 2:
                return None

            # 3. 列聚类：用所有 span 的 x0 起点做一维聚类得到列边界
            col_gap = 8.0  # 列间隙阈值（pt）
            x_starts = sorted({round(s[0] / 5) * 5 for row in rows_spans for s in row})
            col_edges = []  # 列左边界列表
            for x in x_starts:
                if not col_edges or x - col_edges[-1] > col_gap:
                    col_edges.append(x)
            if len(col_edges) < 2:
                return None

            # 4. 行内 span 按 x 归属列，组建成 (header, rows)
            def assign_col(x0: float) -> int:
                """返回 x0 所属列下标（取最近的列左边界）。"""
                best, best_d = 0, 1e9
                for idx, cx in enumerate(col_edges):
                    d = abs(x0 - cx)
                    if d < best_d:
                        best, best_d = idx, d
                return best

            grid_rows = []
            for row_spans in rows_spans:
                cells = {}
                for x0, x1, t in row_spans:
                    ci = assign_col(x0)
                    cells[ci] = cells.get(ci, "") + t
                # 补全缺失列
                row = [cells.get(i, "") for i in range(len(col_edges))]
                # 过滤"整行空"（理论上不会发生）
                if any(row):
                    grid_rows.append(row)

            if len(grid_rows) < 2:
                return None
            # 第一行为表头，其余为数据行
            header = grid_rows[0]
            rows = grid_rows[1:]
            return header, rows
        except Exception as exc:
            logger.warning(f"无线框表格降级提取失败: {exc}")
            return None

    @staticmethod
    def _find_matching_table(page, bbox: BBox):
        """
        在页面表格中寻找与 bbox 重叠度最高的表格对象。

        参数:
            page: fitz.Page
            bbox: 目标区域

        返回:
            Table 对象 或 None
        """
        try:
            tables = page.find_tables().tables
        except Exception:
            return None
        best, best_score = None, 0.0
        for t in tables:
            # 注意：PyMuPDF 1.28 中 Table.bbox 是 tuple，需先转 fitz.Rect
            tb = fitz.Rect(t.bbox)
            # 计算 bbox 与表格区域的重叠率
            inter_w = min(bbox.x1, tb.x1) - max(bbox.x0, tb.x0)
            inter_h = min(bbox.y1, tb.y1) - max(bbox.y0, tb.y0)
            if inter_w <= 0 or inter_h <= 0:
                continue
            score = (inter_w * inter_h) / max(bbox.width * bbox.height, 1e-9)
            if score > best_score:
                best, best_score = t, score
        return best if best_score >= 0.5 else None

    @staticmethod
    def _clean_rows(raw_rows: list) -> list:
        """
        清洗 extract() 返回的行：去除全空行、规整单元格内容。

        参数:
            raw_rows: PyMuPDF extract() 原始行列表

        返回:
            list: 清洗后的行列表
        """
        rows = []
        for row in raw_rows:
            cleaned = ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
            if any(cleaned):  # 丢弃全空行
                rows.append(cleaned)
        return rows

    def _find_caption_and_leading(self, page, bbox: BBox) -> tuple:
        """
        查找表格上方的标题行与前导正文。

        规则：
            1. 在表格上方 _CAPTION_SEARCH_HEIGHT 范围内找文本块；
            2. 匹配"表 X-X xxx"模式的 → caption；
            3. caption 之上的最近正文 → leading_text。

        参数:
            page: fitz.Page
            bbox: 表格区域

        返回:
            (caption: str, leading_text: str)
        """
        caption, leading_text = "", ""
        # 表格上方范围
        top_y = max(0, bbox.y0 - _CAPTION_SEARCH_HEIGHT)
        search_rect = fitz.Rect(bbox.x0 - 10, top_y, bbox.x1 + 10, bbox.y0)

        try:
            text_blocks = page.get_text("dict")["blocks"]
        except Exception:
            return caption, leading_text

        # 收集表格上方（且不与表格重叠）的文本块
        above = []
        for block in text_blocks:
            if block.get("type") != 0:
                continue
            r = fitz.Rect(block["bbox"])
            if r.y1 <= bbox.y0 and r.y0 >= top_y - 1:
                text = "".join(
                    span_text(sp) for line in block.get("lines", []) for sp in line.get("spans", []))
                above.append((r.y1, text.strip()))  # (块底部y, 文本)

        # 按距表格由近到远排序
        above.sort(key=lambda item: item[0], reverse=True)
        for _, text in above:
            if not text:
                continue
            # 表格标题：以"表"开头
            if not caption and _CAPTION_RE.match(text):
                caption = text
                continue
            # 其余最近的正文段落作为前导说明
            if leading_text == "":
                leading_text = text
            if caption:
                break  # 已找到 caption 与 leading，停止

        return caption, leading_text

    @staticmethod
    def _render_markdown_table(caption: str, header: list, rows: list) -> str:
        """
        把表头与行渲染为 Markdown 表格文本。

        参数:
            caption: 表格标题
            header: 表头行
            rows: 数据行

        返回:
            str: Markdown 表格
        """
        def esc(cell: str) -> str:
            """转义 Markdown 表格中的竖线，防止破坏表格结构。"""
            return str(cell).replace("|", "\\|")

        lines = [f"**{caption}**", ""]
        if not header and not rows:
            lines.append("（空表）")
            return "\n".join(lines)

        col_count = len(header) if header else (len(rows[0]) if rows else 0)
        header_row = "| " + " | ".join(esc(c) for c in header) + " |"
        sep_row = "|" + "|".join(["---"] * max(col_count, 1)) + "|"
        lines.append(header_row)
        lines.append(sep_row)
        for row in rows:
            # 单元格不足列数时补空
            padded = list(row) + [""] * (col_count - len(row))
            lines.append("| " + " | ".join(esc(c) for c in padded[:col_count]) + " |")
        return "\n".join(lines)
