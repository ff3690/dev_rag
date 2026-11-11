# -*- coding: utf-8 -*-
"""
tokenizer.py - 中文 token 数估算
================================================

背景：
    bge-small-zh-v1.5 的输入上限是 512 token。切片时需要知道
    "这段文本占多少 token"，以便控制块大小。

为什么不直接用模型 tokenizer？
    加载 tokenizer 需要初始化 transformers / 模型组件，开销大；
    且在 docstruct 解耦原则下不宜依赖 embedding 服务。
    因此采用**字符估算**方案（误差可控，够用即可）：

估算规则：
    1. 中文 / 全角字符：每个约 0.6~0.75 token，取 0.7（保守向上取整）
       —— 即"中文字符数 × 0.7"太松？ 实测 bge-small-zh 一个汉字
       通常略少于 1 个 token（约 0.6~0.7）。为了**不超限**，
       我们按 0.9 计算（留足余量，宁可块略小）。
    2. ASCII 字符：约 4 字符 = 1 token
    3. 其他（数字、标点）：按 2 字符 = 1 token

    结论公式：
        tokens ≈ 中文字符数 × 0.9 + 英文字符数 / 4 + 其他 / 2
"""

import re

# 中文字符（含全角标点）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
# ASCII 字母数字
_ASCII_RE = re.compile(r"[A-Za-z0-9]")


def estimate_tokens(text: str) -> int:
    """
    估算一段文本的 token 数（中文场景）。

    参数:
        text: 待估算文本

    返回:
        int: 估算 token 数（至少 1）
    """
    if not text:
        return 1
    cjk = len(_CJK_RE.findall(text))
    ascii_chars = len(_ASCII_RE.findall(text))
    other = len(text) - cjk - ascii_chars

    tokens = int(cjk * 0.9 + ascii_chars / 4.0 + other / 2.0)
    return max(1, tokens)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    按估算 token 数截断文本（保留开头，适用于"保留表头+前 N 行"场景）。

    参数:
        text: 原文
        max_tokens: 目标 token 上限

    返回:
        str: 截断后的文本（不超过 max_tokens）
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    # 二分查找截断位置（字符粒度）
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low]
