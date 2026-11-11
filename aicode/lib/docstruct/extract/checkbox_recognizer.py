# -*- coding: utf-8 -*-
"""
checkbox_recognizer.py - 复选框识别器
================================================

从复选框区域提取"勾选项清单"。

典型场景（如《国家基本公共卫生服务规范》的预防接种记录表）：
    ☑ 已建立预防接种证
    ☑ 已按程序完成接种
    □ 需补种

输出到 Block.data["items"]：
    [{"text": "已建立预防接种证", "checked": True}, ...]
"""

import re

import fitz  # PyMuPDF

from docstruct.schemas import BlockType, LayoutElement, Block
from docstruct.extract.base import BaseRecognizer
from docstruct.layout.heuristics import is_checked_char, CHECKBOX_CHARS


class CheckboxRecognizer(BaseRecognizer):
    """复选框区域识别器。"""

    def supports(self, block_type: BlockType) -> bool:
        return block_type == "checkbox"

    def recognize(self, element: LayoutElement, page) -> Block:
        """
        从复选框区域提取条目清单。

        参数:
            element: checkbox 版面元素
            page: fitz.Page 对象

        返回:
            Block: data 结构见 schemas.Block 中的 checkbox 约定
        """
        rect = fitz.Rect(element.bbox.x0, element.bbox.y0,
                         element.bbox.x1, element.bbox.y1)
        raw_text = page.get_textbox(rect)

        # 逐行解析：每行是一个候选条目
        items = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 判断该行是否有"已勾选"符号
            checked = any(is_checked_char(ch) for ch in line)
            # 去掉复选框符号，保留纯文本
            clean = "".join(ch for ch in line if ch not in CHECKBOX_CHARS).strip()
            if clean:
                items.append({"text": clean, "checked": checked})

        # 构造条目清单文本（供向量化直接使用）
        content_lines = []
        for it in items:
            mark = "[x]" if it["checked"] else "[ ]"
            content_lines.append(f"{mark} {it['text']}")
        content = "【复选框】\n" + "\n".join(content_lines)

        print(f"[DOCSTRUCT][EXTRACT] 识别复选框: {len(items)} 项")
        return self._make_block(element, content=content, data={"items": items})
