# -*- coding: utf-8 -*-
"""
detector.py - LayoutDetector 版面元素识别器（第一层）
================================================

职责：回答"文档里哪里有哪类内容"，产出 List[LayoutElement]。

识别顺序（优先级从高到低）：
    1. 表格    page.find_tables() 线框表检测
    2. 复选框  文本含 ☑□☐✓✔ 符号 / 4~12pt 细线小方块
    3. 流程图  get_drawings() 中的填充矩形 + 连线聚类
    4. 图片    get_images() + get_image_rects()
    5. 页眉页脚 页面上下边缘重复文本
    6. 标题/正文 剩余文本按"编号模式 + 字号 + 粗体"聚类

容错设计：
    - 单页失败只记录 ParseIssue，不中断整篇解析；
    - 识别结束后统一做重叠仲裁（_resolve_overlaps），防止
      同一个内容被多种规则重复识别（如流程图里的文本框）。

新手提示：
    - "版面元素"只记录类型 + 位置 + 置信度，不提取内容；
    - 提取内容由下一层 extract/ 下的 Recognizer 负责。
"""

import math
from typing import Dict, List, Optional, Tuple

import re

import fitz  # PyMuPDF

from docstruct.schemas import BBox, BlockType, LayoutElement, ParseIssue, ParseReport
from docstruct.utils.logger import logger
from docstruct.utils.text_utils import line_text, span_text
from docstruct.report import I_EMPTY_PAGE, I_HEADER_FOOTER_FILTERED
from docstruct.layout.heuristics import (
    CHECKBOX_CHARS,
    contains_checkbox,
    estimate_title_level,
    is_bold_span,
    is_flow_line_segment,
    is_flow_node_rect,
    is_header_footer_band,
    heading_level,
)

# 元素类型优先级：数值越大优先级越高。
# 高优元素与低优元素区域重叠时，低优元素被丢弃（防止重复识别）。
_PRIORITY: Dict[str, int] = {
    "table": 6,
    "flowchart": 5,
    "image": 4,
    "checkbox": 3,
    "heading": 2,
    "paragraph": 1,
    "page_header": 0,
    "page_footer": 0,
}

# 表格检测的最小行数（少于该行数的"表格"多半是版式误检）
_MIN_TABLE_ROWS = 2

# 目录条目行尾页码模式（目录页特征："标题....页码"，行尾为 1~3 位数字）
_TOC_LINE_RE = re.compile(r"\d{1,3}\s*$")


def _to_bbox(rect) -> BBox:
    """把 fitz.Rect / 四元组转换为本库的 BBox（坐标顺序一致：x0,y0,x1,y1）。"""
    return BBox(rect.x0, rect.y0, rect.x1, rect.y1)


class LayoutDetector:
    """
    版面元素识别器。

    参数:
        pdf_doc: fitz.Document（已打开的 PDF）
        report: ParseReport 实例（用于记录问题；可为 None，内部自动创建）
    """

    def __init__(self, pdf_doc, report: Optional[ParseReport] = None):
        self.pdf_doc = pdf_doc
        self.report = report if report is not None else ParseReport()
        # 缓存上一页的页眉/页脚候选，用于"跨页重复文本"判定
        self._prev_header_candidates: Optional[List[Tuple[BBox, str]]] = None
        self._prev_footer_candidates: Optional[List[Tuple[BBox, str]]] = None

    # ------------------------------------------------------------------
    # 主流程：整篇文档逐页识别 + 重叠仲裁
    # ------------------------------------------------------------------
    def detect(self) -> List[LayoutElement]:
        """
        识别整篇 PDF 的全部版面元素。

        返回:
            List[LayoutElement]: 按阅读顺序排列的元素列表（已做重叠仲裁）
        """
        print(f"\n[DOCSTRUCT][LAYOUT] ===== 开始版面识别，共 {len(self.pdf_doc)} 页 =====")
        all_elements: List[LayoutElement] = []

        # 逐页识别，单页异常不影响其他页
        for page_no in range(1, len(self.pdf_doc) + 1):
            page = self.pdf_doc[page_no - 1]
            try:
                page_elements = self._detect_page(page, page_no)
            except Exception as exc:  # 防御：单页失败仅记录
                logger.error(f"第 {page_no} 页版面识别失败: {exc}")
                self.report.add(ParseIssue(
                    level="error", code="E_LAYOUT_PAGE_FAILED",
                    message=f"第 {page_no} 页版面识别失败: {exc}",
                    page_no=page_no,
                    detail={"exception": str(exc)},
                ))
                continue

            all_elements.extend(page_elements)
            # 汇总每页各类元素数量，便于调试
            counts: Dict[str, int] = {}
            for el in page_elements:
                counts[el.type] = counts.get(el.type, 0) + 1
            print(f"[DOCSTRUCT][LAYOUT] 第 {page_no:>3} 页识别到: "
                  + (", ".join(f"{k}×{v}" for k, v in counts.items()) or "无内容"))

        # 识别完成后统一做重叠仲裁
        final = self._resolve_overlaps(all_elements)

        # 汇总统计并写入报告
        stats: Dict[str, int] = {}
        for el in final:
            stats[el.type] = stats.get(el.type, 0) + 1
        self.report.stats.update({"layout_" + k: v for k, v in stats.items()})

        print(f"[DOCSTRUCT][LAYOUT] ===== 版面识别完成：共 {len(final)} 个元素 =====")
        print(f"[DOCSTRUCT][LAYOUT] 元素统计: {stats}")
        return final

    # ------------------------------------------------------------------
    # 单页识别
    # ------------------------------------------------------------------
    def _detect_page(self, page: "fitz.Page", page_no: int) -> List[LayoutElement]:
        """
        识别单页的全部版面元素。

        参数:
            page: fitz.Page 对象
            page_no: 页码（从 1 开始）

        返回:
            List[LayoutElement]: 该页元素列表
        """
        elements: List[LayoutElement] = []
        # 记录已被"高优元素"占用的区域，文本识别时跳过这些区域（防重复）
        occupied_bboxes: List[BBox] = []

        # 1) 表格（优先级最高）
        table_els = self._detect_tables(page, page_no)
        elements.extend(table_els)
        occupied_bboxes.extend(el.bbox for el in table_els)

        # 2) 复选框
        checkbox_els = self._detect_checkboxes(page, page_no)
        elements.extend(checkbox_els)
        occupied_bboxes.extend(el.bbox for el in checkbox_els)

        # 3) 流程图（矢量分析）
        flow_els = self._detect_flowcharts(page, page_no)
        elements.extend(flow_els)
        occupied_bboxes.extend(el.bbox for el in flow_els)

        # 4) 图片
        image_els = self._detect_images(page, page_no)
        elements.extend(image_els)
        occupied_bboxes.extend(el.bbox for el in image_els)

        # 5) 页眉页脚（跨页重复文本判定）
        header_els, footer_els = self._detect_header_footer(page, page_no)
        elements.extend(header_els)
        elements.extend(footer_els)
        # 页眉页脚区域不参与正文识别，也加入占用区
        occupied_bboxes.extend(el.bbox for el in header_els)
        occupied_bboxes.extend(el.bbox for el in footer_els)

        # 6) 文本（标题 + 正文），跳过占用区域
        text_els = self._detect_text(page, page_no, occupied_bboxes)
        elements.extend(text_els)

        return elements

    # ------------------------------------------------------------------
    # 各类元素的识别实现
    # ------------------------------------------------------------------
    def _detect_tables(self, page: "fitz.Page", page_no: int) -> List[LayoutElement]:
        """
        表格识别：使用 PyMuPDF 内置的线框表检测（find_tables）。

        参数:
            page: 当前页
            page_no: 页码

        返回:
            List[LayoutElement]: 表格元素列表
        """
        els: List[LayoutElement] = []
        try:
            tables = page.find_tables()  # TableFinder 对象
            table_list = tables.tables if hasattr(tables, "tables") else []
        except Exception as exc:
            # 某些畸形 PDF 可能触发 find_tables 异常，防御处理
            logger.warning(f"第 {page_no} 页 find_tables 异常: {exc}")
            return els

        for t in table_list:
            # 过滤太小的"表格"（很可能是误检）
            if t.row_count < _MIN_TABLE_ROWS or t.col_count < 1:
                continue
            els.append(LayoutElement(
                type="table",
                # 注意：PyMuPDF 1.28 中 Table.bbox 是 tuple，需先转 fitz.Rect
                bbox=_to_bbox(fitz.Rect(t.bbox)),
                page_no=page_no,
                confidence=1.0,
                source="rule",
                detail={"rows": t.row_count, "cols": t.col_count},
            ))
        return els

    def _detect_checkboxes(self, page: "fitz.Page", page_no: int) -> List[LayoutElement]:
        """
        复选框识别：两种信号。

        信号一：文本 span 中包含 ☑□☐✓✔ 等复选框符号；
        信号二：页面上 4~12pt 的细线小方块（drawings）。

        参数:
            page: 当前页
            page_no: 页码

        返回:
            List[LayoutElement]: 复选框元素列表
        """
        els: List[LayoutElement] = []

        # ---- 信号一：文本符号 ----
        symbol_rects: List[BBox] = []
        rawdict = page.get_text("rawdict")
        for block in rawdict.get("blocks", []):
            if block.get("type") != 0:  # 只处理文本块
                continue
            for line in block.get("lines", []):
                line_bbox = _to_bbox(fitz.Rect(line["bbox"]))
                line_content = line_text(line)  # 兼容不同 PyMuPDF 版本
                if contains_checkbox(line_content):
                    symbol_rects.append(line_bbox)

        # ---- 信号二：细线小方块 ----
        small_rects: List[BBox] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for item in drawings:
            if item.get("type") not in ("f", "fs", "fr", "frs"):  # 填充/描边矩形
                continue
            r = item.get("rect")
            if r is None:
                continue
            w, h = r.width, r.height
            # 小方块：边长在 4~12pt 之间（流程图节点 >= 18pt，二者分开）
            if 4 <= w <= 12 and 4 <= h <= 12:
                small_rects.append(_to_bbox(r))

        # 合并两类信号（同一行的多个复选框算一个区域）
        all_rects = symbol_rects + small_rects
        if not all_rects:
            return els

        merged = self._merge_nearby_rects(all_rects, gap=30.0)
        for bbox in merged:
            els.append(LayoutElement(
                type="checkbox",
                bbox=bbox,
                page_no=page_no,
                confidence=0.9,
                source="rule",
                detail={"symbol_rects": len(symbol_rects), "small_rects": len(small_rects)},
            ))
        return els

    def _detect_flowcharts(self, page: "fitz.Page", page_no: int) -> List[LayoutElement]:
        """
        流程图识别（矢量分析）。

        流程图特征：多个填充矩形（节点）+ 若干线段（连线），分布在页面中部。
        本 PDF 的 12 张服务流程图全部是这种矢量形式。

        参数:
            page: 当前页
            page_no: 页码

        返回:
            List[LayoutElement]: 流程图元素列表
        """
        try:
            drawings = page.get_drawings()
        except Exception as exc:
            logger.warning(f"第 {page_no} 页 get_drawings 异常: {exc}")
            return []

        # ---- 收集节点矩形 ----
        node_rects: List[BBox] = []
        for item in drawings:
            if item.get("type") not in ("f", "fs", "fr", "frs"):
                continue
            r = item.get("rect")
            if r is None:
                continue
            # 只有"填充矩形"（fill 非空）才算节点候选；
            # 纯描边矩形（无填充）通常是装饰性边框，不算节点
            if item.get("fill") is None:
                continue
            if is_flow_node_rect(r.width, r.height):
                node_rects.append(_to_bbox(r))

        # ---- 收集连线 ----
        line_count = 0
        line_bboxes: List[BBox] = []
        for item in drawings:
            for sub in item.get("items", []):
                if sub[0] == "l" and len(sub) >= 3:  # ("l", p1, p2) 线段
                    p1, p2 = sub[1], sub[2]
                    if is_flow_line_segment(p1.x, p1.y, p2.x, p2.y):
                        line_count += 1
                        line_bboxes.append(BBox(
                            min(p1.x, p2.x), min(p1.y, p2.y),
                            max(p1.x, p2.x), max(p1.y, p2.y),
                        ))

        # ---- 判定：节点足够多 + 有连线，才认为是流程图 ----
        if len(node_rects) < 3 or line_count < 2:
            return []

        # ---- 聚类：节点可能分散为多个区域（一页多图场景） ----
        clusters = self._cluster_rects(node_rects, gap_x=80.0, gap_y=40.0)
        els: List[LayoutElement] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            bbox = self._union_bbox(cluster)
            els.append(LayoutElement(
                type="flowchart",
                bbox=bbox,
                page_no=page_no,
                confidence=0.95 if len(cluster) >= 3 else 0.8,
                source="rule",
                detail={"nodes": len(cluster), "lines": line_count},
            ))
        return els

    def _detect_images(self, page: "fitz.Page", page_no: int) -> List[LayoutElement]:
        """
        图片识别：查找页面上的位图及其放置位置。

        参数:
            page: 当前页
            page_no: 页码

        返回:
            List[LayoutElement]: 图片元素列表
        """
        els: List[LayoutElement] = []
        try:
            images = page.get_images(full=True)  # [(xref, smask, w, h, ...)]
        except Exception as exc:
            logger.warning(f"第 {page_no} 页 get_images 异常: {exc}")
            return []

        for img in images:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects:
                # 统一转 fitz.Rect（不同版本可能返回 tuple）
                bbox = _to_bbox(fitz.Rect(rect))
                # 过滤过小的图片（< 15pt，多为装饰或页眉 logo）
                if bbox.width < 15 or bbox.height < 15:
                    continue
                els.append(LayoutElement(
                    type="image",
                    bbox=bbox,
                    page_no=page_no,
                    confidence=1.0,
                    source="rule",
                    detail={"xref": xref},
                ))
        return els

    def _detect_header_footer(self, page: "fitz.Page", page_no: int) -> Tuple[List[LayoutElement], List[LayoutElement]]:
        """
        页眉页脚识别。

        规则：
            1. 位于页面顶部/底部边缘区域的文本行；
            2. 与上一页"相同位置、相同文本"的候选 → 判定为页眉/页脚；
            3. 纯数字页码直接判定为页脚。

        参数:
            page: 当前页
            page_no: 页码

        返回:
            (页眉元素列表, 页脚元素列表)
        """
        page_height = page.rect.height
        rawdict = page.get_text("rawdict")

        top_candidates: List[Tuple[BBox, str]] = []
        bottom_candidates: List[Tuple[BBox, str]] = []

        for block in rawdict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox = _to_bbox(fitz.Rect(line["bbox"]))
                text = line_text(line).strip()  # 兼容不同 PyMuPDF 版本
                if not text:
                    continue
                if is_header_footer_band(bbox.y0, page_height):
                    if bbox.y0 < page_height / 2:
                        top_candidates.append((bbox, text))
                    else:
                        bottom_candidates.append((bbox, text))

        # 页脚：纯数字（页码）直接判定
        footer_els: List[LayoutElement] = []
        for bbox, text in bottom_candidates:
            if text.isdigit():
                footer_els.append(LayoutElement(
                    type="page_footer", bbox=bbox, page_no=page_no,
                    confidence=1.0, source="rule", detail={"text": text},
                ))
                self.report.add(ParseIssue(
                    level="info", code=I_HEADER_FOOTER_FILTERED,
                    message=f"第 {page_no} 页过滤页脚: {text}", page_no=page_no,
                ))

        # 页眉/非数字页脚：与上一页候选比对（相同位置 + 相同文本）
        header_els: List[LayoutElement] = []
        prev_top = self._prev_header_candidates
        prev_bottom = self._prev_footer_candidates

        for bbox, text in top_candidates:
            if prev_top and self._same_position_text(bbox, text, prev_top):
                header_els.append(LayoutElement(
                    type="page_header", bbox=bbox, page_no=page_no,
                    confidence=0.9, source="rule", detail={"text": text},
                ))
                self.report.add(ParseIssue(
                    level="info", code=I_HEADER_FOOTER_FILTERED,
                    message=f"第 {page_no} 页过滤页眉: {text}", page_no=page_no,
                ))

        for bbox, text in bottom_candidates:
            if text.isdigit():
                continue  # 已处理
            if prev_bottom and self._same_position_text(bbox, text, prev_bottom):
                header_els.append(LayoutElement(
                    type="page_footer", bbox=bbox, page_no=page_no,
                    confidence=0.9, source="rule", detail={"text": text},
                ))

        # 更新缓存供下一页比对（第一页缓存为空，从第二页开始比对）
        self._prev_header_candidates = top_candidates
        self._prev_footer_candidates = bottom_candidates
        return header_els, footer_els

    @staticmethod
    def _same_position_text(bbox: BBox, text: str, prev_candidates: List[Tuple[BBox, str]]) -> bool:
        """
        判断当前文本是否与上一页候选"位置相近 + 文本相同"。

        参数:
            bbox: 当前文本 bbox
            text: 当前文本
            prev_candidates: 上一页的候选 [(bbox, text), ...]

        返回:
            True 表示与上一页某候选相同（跨页重复 → 页眉页脚特征）
        """
        for prev_bbox, prev_text in prev_candidates:
            if text == prev_text and bbox.overlaps(prev_bbox, ratio=0.8):
                return True
        return False

    # ------------------------------------------------------------------
    # 文本识别（标题 + 正文）
    # ------------------------------------------------------------------
    def _detect_text(self, page: "fitz.Page", page_no: int,
                     occupied_bboxes: List[BBox]) -> List[LayoutElement]:
        """
        识别标题与正文。

        流程：
            1. 从 rawdict 取出所有文本 span；
            2. 过滤掉落在占用区域（表格/流程图/图片/复选框/页眉页脚）内的 span；
            3. 按行合并（同一 y 区间），按段聚类（垂直间距小 → 同一段）；
            4. 每段判断：编号模式 + 字号 + 粗体 → 标题 or 正文。

        参数:
            page: 当前页
            page_no: 页码
            occupied_bboxes: 已被高优元素占用的区域列表

        返回:
            List[LayoutElement]: 标题 + 正文元素
        """
        rawdict = page.get_text("rawdict")

        # ---- 步骤1&2：收集未占用的文本行 ----
        # 每一行记录为 (bbox, 行文本, 平均字号, 是否粗体)
        lines: List[Tuple[BBox, str, float, bool]] = []
        for block in rawdict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_bbox = _to_bbox(fitz.Rect(line["bbox"]))
                # 过滤：行完全落在占用区域内（表格内容等）
                if any(occ.contains(line_bbox) for occ in occupied_bboxes):
                    continue
                spans = line.get("spans", [])
                text = line_text(line).strip()  # 兼容不同 PyMuPDF 版本
                if not text:
                    continue
                # 平均字号（按文本长度加权）
                span_texts = [span_text(sp) for sp in spans]
                total_len = sum(len(t) for t in span_texts) or 1
                avg_size = sum(sp["size"] * len(t) for sp, t in zip(spans, span_texts)) / total_len
                is_bold = any(is_bold_span(sp.get("flags", 0)) for sp in spans)
                lines.append((line_bbox, text, avg_size, is_bold))

        # ---- 步骤3：按垂直间距聚类为段落 ----
        paragraphs: List[Tuple[BBox, str, float, bool]] = []
        for line_bbox, text, size, bold in lines:
            # 标题行自成一段，不与正文粘连。
            # 判定：编号正则（"一、"、"1."等）命中，或字号/粗体特征明显
            # （如"1 居民健康档案管理服务规范"是数字+空格标题，正则匹配不到，
            #   但字号大且加粗，estimate_title_level 可以识别）。
            # 否则"标题+正文首段"会被聚类成一个块，标题判定与章节路径都会失真。
            stripped_line = text.strip()
            is_heading_line = (heading_level(stripped_line) > 0
                               or estimate_title_level(size, bold, stripped_line) > 0)
            if paragraphs and not is_heading_line:
                # 与上一段末尾的距离
                last_bbox = paragraphs[-1][0]
                last_text = paragraphs[-1][1]
                last_size = paragraphs[-1][2]
                last_bold = paragraphs[-1][3]
                gap = line_bbox.y0 - last_bbox.y1
                # 上一段若是标题段，正文行也不合并（标题与正文各自独立成段）
                last_is_heading = (heading_level(last_text.strip()) > 0
                                   or estimate_title_level(last_size, last_bold,
                                                           last_text.strip()) > 0)
                # 间距小于 0.6 倍行高 → 属于同一段（正文行才合并）
                if not last_is_heading and gap < 0.6 * size:
                    # 合并 bbox 并追加文本（用换行符分隔，保留行结构）
                    pb = paragraphs[-1]
                    new_bbox = BBox(
                        min(pb[0].x0, line_bbox.x0), min(pb[0].y0, line_bbox.y0),
                        max(pb[0].x1, line_bbox.x1), max(pb[0].y1, line_bbox.y1),
                    )
                    paragraphs[-1] = (
                        new_bbox, pb[1] + "\n" + text, max(pb[2], size), pb[3] or bold,
                    )
                    continue
            paragraphs.append((line_bbox, text, size, bold))

        # ---- 步骤4：判定标题 / 正文 ----
        els: List[LayoutElement] = []
        for bbox, text, size, bold in paragraphs:
            stripped = text.strip()
            line_parts = stripped.split("\n")
            single_line = len(line_parts) <= 1
            lvl = heading_level(stripped) if single_line else 0

            # 目录条目检测：单行标题 + 行尾带页码（目录页特征）
            # 典型目录行："1.居民健康档案封面........8" / "5 附件1 表单目录  8"
            # 目录页不应生成 heading（否则整页目录文本会污染 section_path），
            # 这里把目录行降级为普通段落。
            if single_line and lvl > 0 and _TOC_LINE_RE.search(stripped):
                print(f"[DOCSTRUCT][LAYOUT] 目录条目（p{page_no}）: "
                      f"{stripped[:25]}... 降级为段落")
                els.append(LayoutElement(
                    type="paragraph", bbox=bbox, page_no=page_no,
                    confidence=0.8, source="rule",
                    detail={"font_size": round(size, 1), "bold": bold,
                            "note": "目录条目，降级为段落"},
                ))
                continue

            # 多行"标题清单"检测：段落含多行、且第 2 行起仍有大量行匹配
            # 标题编号模式 → 这是目录页整页清单（行间距小被聚类合并），
            # 而非单个标题。若判为 heading，整段目录文本会写入 section_path。
            if not single_line:
                rest_heading_lines = sum(
                    1 for _l in line_parts[1:]
                    if _l.strip() and heading_level(_l.strip()) > 0
                )
                if rest_heading_lines >= max(1, len(line_parts) // 3):
                    print(f"[DOCSTRUCT][LAYOUT] 标题清单（目录页, p{page_no}）: "
                          f"{line_parts[0][:25]}... 降级为段落")
                    els.append(LayoutElement(
                        type="paragraph", bbox=bbox, page_no=page_no,
                        confidence=0.8, source="rule",
                        detail={"font_size": round(size, 1), "bold": bold,
                                "note": "标题清单(目录页)，降级为段落"},
                    ))
                    continue

            level = estimate_title_level(size, bold, stripped)
            if level > 0:
                els.append(LayoutElement(
                    type="heading", bbox=bbox, page_no=page_no,
                    confidence=1.0, source="rule",
                    detail={"level": level, "font_size": round(size, 1), "bold": bold},
                ))
            else:
                els.append(LayoutElement(
                    type="paragraph", bbox=bbox, page_no=page_no,
                    confidence=1.0, source="rule",
                    detail={"font_size": round(size, 1), "bold": bold},
                ))
        return els

    # ------------------------------------------------------------------
    # 重叠仲裁
    # ------------------------------------------------------------------
    def _resolve_overlaps(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        重叠仲裁：同区域被多种规则识别时，保留优先级高的，丢弃低优的。

        例如：流程图节点是"填充矩形"，如果页面上恰好有一个表格边框
        被误判为矩形聚类，两个元素重叠时保留表格、丢弃误判的流程图。

        两个关键设计（新手注意）：
            1. **只比较同页元素**：不同页的坐标空间彼此独立（都从页面
               左上角起算），若不区分页码，第 10 页的表格会与第 20 页的
               正文"假重叠"，导致大量正文被误删。
            2. **流程图外框预处理**：服务流程图常整页画在一个大线框里，
               PyMuPDF 会把线框误判为"表格"。这里先移除与流程图高度
               重叠（>=70%）的表格，避免流程图被表格覆盖丢弃。

        参数:
            elements: 仲裁前的全部元素

        返回:
            List[LayoutElement]: 仲裁后的元素列表
        """
        # ---- 预处理：流程图外框误判为表格 ----
        flowchart_els = [el for el in elements if el.type == "flowchart"]
        if flowchart_els:
            filtered: List[LayoutElement] = []
            for el in elements:
                if el.type == "table":
                    is_flow_frame = False
                    for f in flowchart_els:
                        if f.page_no != el.page_no:
                            continue  # 只比较同页
                        # 表格与流程图任一方高度重叠（>=70%）→ 表格是外框
                        if (el.bbox.overlaps(f.bbox, ratio=0.7)
                                or f.bbox.overlaps(el.bbox, ratio=0.7)):
                            is_flow_frame = True
                            break
                    if is_flow_frame:
                        print(f"[DOCSTRUCT][LAYOUT] 识别到流程图外框（误判表格），"
                              f"移除该表格 (p{el.page_no})")
                        continue  # 丢弃该表格元素
                filtered.append(el)
            elements = filtered

        # 按优先级从高到低排序（同优先级保持原顺序，稳定排序）
        sorted_els = sorted(elements, key=lambda el: _PRIORITY.get(el.type, 0), reverse=True)

        result: List[LayoutElement] = []
        for el in sorted_els:
            # 与已保留元素比较：若重叠率高则丢弃当前元素
            keep = True
            for kept in result:
                if kept.page_no != el.page_no:
                    continue  # 跨页元素不比较（坐标空间独立）
                if kept.type == el.type:
                    continue  # 同类型元素允许共存
                if el.bbox.overlaps(kept.bbox, ratio=0.6):
                    # 调试输出，便于定位误判
                    print(f"[DOCSTRUCT][LAYOUT] 重叠仲裁: 丢弃 {el.type} "
                          f"(p{el.page_no})，与已保留的 {kept.type} 重叠")
                    keep = False
                    break
            if keep:
                result.append(el)

        # 恢复阅读顺序（按页码 + 纵向坐标排序）
        result.sort(key=lambda el: (el.page_no, el.bbox.y0, el.bbox.x0))
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _union_bbox(rects: List[BBox]) -> BBox:
        """计算多个矩形的最小外接矩形（包围盒）。"""
        return BBox(
            min(r.x0 for r in rects),
            min(r.y0 for r in rects),
            max(r.x1 for r in rects),
            max(r.y1 for r in rects),
        )

    @staticmethod
    def _merge_nearby_rects(rects: List[BBox], gap: float) -> List[BBox]:
        """
        合并位置相近的矩形（用于复选框多符号聚合、流程图节点聚类）。

        思路：每次取出一个矩形，找到所有与它"横向/纵向距离 < gap"的矩形
        合并为一个大区域，重复直到没有可合并的。

        参数:
            rects: 矩形列表
            gap: 合并阈值（pt）

        返回:
            List[BBox]: 合并后的区域列表
        """
        pool = [r for r in rects]
        merged: List[BBox] = []
        while pool:
            base = pool.pop(0)
            changed = True
            while changed:
                changed = False
                for other in pool[:]:
                    # 两个矩形外扩 gap 后是否相交
                    base_gap = BBox(base.x0 - gap, base.y0 - gap, base.x1 + gap, base.y1 + gap)
                    other_gap = BBox(other.x0 - gap, other.y0 - gap, other.x1 + gap, other.y1 + gap)
                    if (base_gap.x0 <= other_gap.x1 and base_gap.x1 >= other_gap.x0
                            and base_gap.y0 <= other_gap.y1 and base_gap.y1 >= other_gap.y0):
                        # 合并 base 与 other
                        base = BBox(
                            min(base.x0, other.x0), min(base.y0, other.y0),
                            max(base.x1, other.x1), max(base.y1, other.y1),
                        )
                        pool.remove(other)
                        changed = True
            merged.append(base)
        return merged

    @staticmethod
    def _cluster_rects(rects: List[BBox], gap_x: float, gap_y: float) -> List[List[BBox]]:
        """
        把矩形按空间位置聚类成多个簇（一页可能有多张流程图）。

        简单实现：从第一个矩形开始，把所有与簇内任一矩形
        在 x 方向间距 < gap_x 且 y 方向间距 < gap_y 的矩形并入该簇。

        参数:
            rects: 矩形列表
            gap_x: x 方向合并阈值（pt）
            gap_y: y 方向合并阈值（pt）

        返回:
            List[List[BBox]]: 聚类后的矩形簇
        """
        clusters: List[List[BBox]] = []
        pool = [r for r in rects]

        while pool:
            cluster = [pool.pop(0)]
            changed = True
            while changed:
                changed = False
                for other in pool[:]:
                    for member in cluster:
                        dx = max(0.0, max(member.x1, other.x1) - min(member.x0, other.x0)
                                 - (member.width + other.width))
                        dy = max(0.0, max(member.y1, other.y1) - min(member.y0, other.y0)
                                 - (member.height + other.height))
                        if dx <= gap_x and dy <= gap_y:
                            cluster.append(other)
                            pool.remove(other)
                            changed = True
                            break
                    if changed:
                        break
            clusters.append(cluster)
        return clusters
