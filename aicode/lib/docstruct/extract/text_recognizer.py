# -*- coding: utf-8 -*-
"""
text_recognizer.py - 文本识别器（标题 + 正文）
================================================

从版面元素（heading / paragraph）区域中提取文本内容。

版面识别阶段（LayoutDetector）只判定了"这里是什么 + 在哪"，
本识别器负责把该区域里的实际文本取出来，并规整换行与空白。

为什么用 page.get_textbox(rect) 重新取文本？
    因为 LayoutElement 只存了 bbox，没有存文本（存了会占用内存）。
    get_textbox 是按坐标区域取文本的 PyMuPDF 标准接口。
"""

import re

import fitz  # PyMuPDF

from docstruct.schemas import BlockType, LayoutElement, Block
from docstruct.extract.base import BaseRecognizer

# 连续空白（含换行）压缩为正则，规整段落文本
_WS_RE = re.compile(r"[ \t\u3000]+")
_NEWLINE_RE = re.compile(r"\n+")

# 标题区域内的编号条目片段（防御性目录清单检测用）
# 例如："1.居民健康档案封面"、"2.个人基本信息表"、"4.1.5 男童…"
# 前缀排除数字本身（避免把"12.30 月龄"里的"2.30"误算成条目）
_HEADING_SEG_RE = re.compile(r"(?:\A|[^\d])\d{1,2}[\.、]\s*[^\s，。；]{2,}")


class TextRecognizer(BaseRecognizer):
    """
    标题 / 正文文本识别器。

    参数:
        doc_id: 文档标识
        report: 质量报告
        seq_counter: 共享块序号计数器（保证全篇 block_id 唯一）
    """

    def supports(self, block_type: BlockType) -> bool:
        """只处理标题与正文两种类型。"""
        return block_type in ("heading", "paragraph")

    def recognize(self, element: LayoutElement, page) -> Block:
        """
        从元素区域提取文本，产出 Block。

        参数:
            element: heading 或 paragraph 版面元素
            page: fitz.Page 对象

        返回:
            Block: 文本内容块（heading 带 data.level 层级）
        """
        # 按 bbox 提取该区域的原始文本
        rect = fitz.Rect(element.bbox.x0, element.bbox.y0,
                         element.bbox.x1, element.bbox.y1)
        raw_text = page.get_textbox(rect)

        if element.type == "heading":
            # 标题：压缩空白 + 去除换行（标题通常一行）
            text = _NEWLINE_RE.sub(" ", raw_text)
            text = _WS_RE.sub(" ", text).strip()
            level = element.detail.get("level", 2)

            # 防御性目录清单检测：
            # 版面识别偶尔会把"目录页整页"误判为单个标题（目录行距极小，
            # PyMuPDF 把多行合并，get_textbox 提取出整段目录文本）。
            # 特征：文本内含 ≥3 个编号条目（"1.xxx 2.xxx 3.xxx"）。
            # 此时降级为正文段落，避免整段目录文本写入 section_path，
            # 污染后续所有块的章节路径。
            segments = _HEADING_SEG_RE.findall(text)
            if len(segments) >= 3:
                # 注意：必须显式传 block_type="paragraph" 覆盖块类型，
                # 否则 _make_block 默认沿用 element.type（heading），
                # 降级块仍会进入标题树，污染后续块的 section_path。
                print(f"[DOCSTRUCT][EXTRACT] 标题区域含多个编号条目（目录清单），"
                      f"降级为段落: {text[:30]}...")
                data = {}
                return self._make_block(element, content=text, data=data,
                                        block_type="paragraph")

            data = {"level": level}
            print(f"[DOCSTRUCT][EXTRACT] 识别标题 (level={level}): {text[:40]}")
        else:
            # 正文：压缩空白但保留换行（段落内换行）
            text = _WS_RE.sub("", raw_text)
            text = _NEWLINE_RE.sub("\n", text).strip()
            data = {}

        # 用基类的 _make_block 统一构造 Block（block_id 自动编号）
        return self._make_block(element, content=text, data=data)
