# -*- coding: utf-8 -*-
"""
extract/__init__.py - 内容提取子包（第二层）
================================================

注册所有 Recognizer，并提供按 BlockType 路由的工厂函数。

用法：
    from docstruct.extract import get_recognizer
    recognizer = get_recognizer("table")   # 返回 TableRecognizer 实例
    block = recognizer.recognize(element, page)
"""

from typing import Dict

from docstruct.schemas import BlockType
from docstruct.extract.base import BaseRecognizer
from docstruct.extract.text_recognizer import TextRecognizer
from docstruct.extract.table_recognizer import TableRecognizer
from docstruct.extract.flowchart_recognizer import FlowchartRecognizer
from docstruct.extract.image_recognizer import ImageRecognizer
from docstruct.extract.checkbox_recognizer import CheckboxRecognizer

# 识别器注册表：每种元素类型对应一个 Recognizer 实例。
# 新增类型时只需在此注册，get_recognizer 无需改动。
_RECOGNIZERS: Dict[BlockType, BaseRecognizer] = {
    "heading": TextRecognizer,
    "paragraph": TextRecognizer,
    "table": TableRecognizer,
    "flowchart": FlowchartRecognizer,
    "image": ImageRecognizer,
    "checkbox": CheckboxRecognizer,
    "page_header": None,  # 页眉页脚不需要提取内容
    "page_footer": None,
}


def get_recognizer(block_type: BlockType) -> BaseRecognizer:
    """
    根据元素类型返回对应的 Recognizer 类（工厂函数）。

    参数:
        block_type: 元素类型（见 BlockType）

    返回:
        BaseRecognizer: 对应识别器类；页眉页脚等无需提取的类型返回 None
    """
    return _RECOGNIZERS.get(block_type)
