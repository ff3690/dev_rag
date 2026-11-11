# -*- coding: utf-8 -*-
"""
text_utils.py - 文本提取辅助（兼容不同 PyMuPDF 版本）
================================================

PyMuPDF 不同版本中 page.get_text("rawdict") 返回的 span 结构有差异：
    旧版本（<=1.26）: span 直接带 "text" 字段
    新版本（>=1.27）: span 不再有 "text"，文本在 span["chars"] 里，
                      每个 char 的字符在 char["c"] 字段

为了不依赖具体版本，统一通过 span_text() 提取，二者都兼容。

新手提示：
    rawdict 是 PyMuPDF 的"最底层"文本结构，结构层次为：
        page -> blocks[] -> lines[] -> spans[] -> chars[]
    其中 chars 是"字符级"细节（含每个字符的精确位置），
    通常用不到那么细，只要把 chars 拼成字符串即可。
"""


def span_text(span: dict) -> str:
    """
    从 rawdict 的 span 字典中提取完整文本（跨版本兼容）。

    参数:
        span: rawdict 中的 span 字典

    返回:
        str: 该 span 的文本内容（无内容返回空串）
    """
    # 1) 旧版本：span 直接带 "text" 字段
    text = span.get("text")
    if text:
        return text

    # 2) 新版本：文本拆在 span["chars"] 的每个 char["c"] 里
    chars = span.get("chars")
    if isinstance(chars, list) and chars:
        return "".join(ch.get("c", "") for ch in chars if isinstance(ch, dict))

    # 3) 都没有 → 空串
    return ""


def line_text(line: dict) -> str:
    """
    从 rawdict 的 line 字典中提取完整文本（内部拼接所有 span）。

    参数:
        line: rawdict 中的 line 字典

    返回:
        str: 该行文本
    """
    spans = line.get("spans", []) or []
    return "".join(span_text(sp) for sp in spans)
