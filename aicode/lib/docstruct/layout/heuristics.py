# -*- coding: utf-8 -*-
"""
heuristics.py - 版面判定辅助规则
================================================

把"如何判断一个东西是标题 / 复选框 / 流程图节点"等**判定规则**集中在这里，
让 detector.py 只负责流程编排，规则集中便于调试与调整。

本文件全部是纯函数，输入文本/字号返回判定结果，不依赖 PDF 对象，
因此可以独立写单元测试。
"""

import re
from typing import Optional, Tuple


# =============================================================================
# 1. 标题层级判定
# =============================================================================

# 标题编号的层级模式（按优先级从高到低排列）。
# 以《国家基本公共卫生服务规范》为例：
#   level 1: "第一部分 居民健康档案管理"
#   level 2: "一、服务对象"
#   level 3: "（一）X"
#   level 4: "1. X"
# 缺陷6扩充：
#   - 阿拉伯数字层级 "1.1"（置于 "1." 之前，避免被短模式截胡）
#   - 英文章节模式 Chapter/Section/§（涉外规范与英文排版文档）
#   - 半角括号 （1） / (1)（兼容中英混排）
HEADING_LEVEL_PATTERNS: Tuple[Tuple[int, re.Pattern], ...] = (
    (1, re.compile(r"^第[一二三四五六七八九十百]+部分")),      # 第一部分 / 第十二部分
    (1, re.compile(r"^第[一二三四五六七八九十百]+章")),        # 第一章（缺陷6：常见英文译本结构）
    (1, re.compile(r"^Chapter\s+\d+", re.IGNORECASE)),        # Chapter 1（缺陷6）
    (2, re.compile(r"^[一二三四五六七八九十]+、")),           # 一、 二、 …… 十、
    (2, re.compile(r"^Section\s+\d+", re.IGNORECASE)),        # Section 2（缺陷6）
    (2, re.compile(r"^第[一二三四五六七八九十百]+节")),        # 第一节（缺陷6）
    (3, re.compile(r"^（[一二三四五六七八九十]+）")),         # （一） （二） ……
    (3, re.compile(r"^\([一二三四五六七八九十]+\)")),         # (一) (二) 半角（缺陷6）
    (4, re.compile(r"^\d{1,2}\.\d{1,2}[\.、\s]")),            # 1.1  2.3（缺陷6，必须排在 1. 之前）
    (4, re.compile(r"^\d{1,2}[\.、]")),                       # 1.  2.  12、
    (4, re.compile(r"^\([0-9]+\)")),                          # (1) (2)（缺陷6）
)

# 标题否决规则（缺陷6：模式误判纠偏）。
# 命中任意一条 → 即使匹配编号模式也判定为"非标题"（正文/目录行）。
HEADING_NEGATIVE_RULES = (
    "ends_with_sentence_punct",  # 以句号/逗号结尾 → 更像正文句子
    "too_long",                  # 超长行（> 60 字符）→ 更可能是正文或目录项
    "contains_strong_verb",      # 含动词短语标记词 → 正文叙述句
)

# 标题行长度上限（否决规则 too_long 用）
HEADING_MAX_LENGTH = 60
# 标题不可出现在正文中常见的"动词引导句"标记词
HEADING_SENTENCE_MARKERS = ("是", "包括", "分为", "如下", "如下表", "如下所示")


def _hit_negative_rule(text: str) -> bool:
    """
    判断文本是否命中标题否决规则（缺陷6）。

    参数:
        text: 行文本（已 strip）

    返回:
        True 表示该行不应判为标题
    """
    # 1) 句末标点否决：正文叙述句通常以句号/逗号/分号结尾
    if text.endswith(("。", "，", "；", ".", ",")):
        return True
    # 2) 超长否决：标题一般较短
    if len(text) > HEADING_MAX_LENGTH:
        return True
    # 3) 动词引导句否决："本规范包括…"这类以标记词开头的叙述句
    for marker in HEADING_SENTENCE_MARKERS:
        if text.startswith(marker):
            return True
    return False


def heading_level(text: str) -> int:
    """
    根据文本内容判断标题层级。

    参数:
        text: 一行文本（已去除首尾空白）

    返回:
        int: 标题层级 1~4；不是标题返回 0
    """
    if not text:
        return 0
    text = text.strip()
    # 缺陷6：否决规则优先于编号模式（句号结尾/超长/动词引导 → 正文而非标题）
    if _hit_negative_rule(text):
        return 0
    for level, pattern in HEADING_LEVEL_PATTERNS:
        if pattern.match(text):
            return level
    return 0


def is_heading_line(text: str) -> bool:
    """
    快速判断一行文本是否是标题（供聚类时使用）。

    参数:
        text: 一行文本

    返回:
        True 表示该行是标题
    """
    return heading_level(text) > 0


# =============================================================================
# 2. 复选框符号判定
# =============================================================================

# 常见复选框符号（Unicode）。
# ☑ = 已勾选（带对勾的方框）、□ / ☐ = 未勾选、✓ / ✔ = 对勾、☒ = 打叉
CHECKBOX_CHARS = set("☑□☐✓✔☒")


def contains_checkbox(text: str) -> bool:
    """
    判断文本中是否包含复选框符号。

    参数:
        text: 待检查文本

    返回:
        True 表示包含复选框符号
    """
    return any(ch in CHECKBOX_CHARS for ch in text)


def is_checked_char(ch: str) -> bool:
    """
    判断单个字符是否表示"已勾选"状态。

    ☑（带对勾方框）和 ✓/✔（对勾）视为已勾选；□/☐/☒ 视为未勾选。

    参数:
        ch: 单个字符

    返回:
        True 表示已勾选
    """
    return ch in "☑✓✔"


# =============================================================================
# 3. 字号/粗体判定
# =============================================================================

# PyMuPDF span 的 flags 字段中，第 4 位（值 16）表示粗体。
# 参考 fitz 常量：TEXT_FLAGS_BOLD = 2**4
_BOLD_FLAG = 2 ** 4


def is_bold_span(span_flags: int) -> bool:
    """
    根据 span 的 flags 判断是否粗体。

    参数:
        span_flags: PyMuPDF rawdict 中 span 的 flags 整数值

    返回:
        True 表示粗体
    """
    return bool(span_flags & _BOLD_FLAG)


def estimate_title_level(font_size: float, is_bold: bool, text: str) -> int:
    """
    综合"文本编号模式 + 字号 + 粗体"推断标题层级。

    优先级：
        1. 文本编号模式（最可靠，heading_level）；
        2. 无编号但字号明显大于正文（> 1.2 倍正文）且加粗 → 视为 level 2（保守）；
        3. 其余返回 0。

    参数:
        font_size: 该行字号（pt）
        is_bold: 是否粗体
        text: 行文本

    返回:
        int: 标题层级 0~4
    """
    # 1) 先看编号模式
    level = heading_level(text)
    if level > 0:
        return level
    # 2) 无编号的"大字号+粗体"也当作标题（如文档封面标题、章节引导句）
    if font_size >= 14 and is_bold:
        return 2
    return 0


# =============================================================================
# 4. 流程图判定辅助
# =============================================================================

# 流程图节点的最小边长（pt）。复选框小方块（4~12pt）不在此列，避免混淆。
FLOW_NODE_MIN_SIZE = 18.0
# 流程图连线的判定条件：线段长度（pt）
FLOW_LINE_MIN_LENGTH = 20.0
# 页面上下边距（pt），此区域内的文本更可能是页眉页脚而非流程图节点
PAGE_EDGE_MARGIN = 50.0


def is_flow_node_rect(rect_w: float, rect_h: float) -> bool:
    """
    判断一个填充矩形是否可能是流程图节点。

    规则：矩形边长都 >= FLOW_NODE_MIN_SIZE。
    复选框小方块（4~12pt）会被此规则排除。

    参数:
        rect_w: 矩形宽度（pt）
        rect_h: 矩形高度（pt）

    返回:
        True 表示可能是流程图节点
    """
    return rect_w >= FLOW_NODE_MIN_SIZE and rect_h >= FLOW_NODE_MIN_SIZE


def is_flow_line_segment(x0: float, y0: float, x1: float, y1: float) -> bool:
    """
    判断一条线段是否可能是流程图连线（长度足够长）。

    参数:
        x0, y0: 线段起点
        x1, y1: 线段终点

    返回:
        True 表示可能是连线
    """
    dx = x1 - x0
    dy = y1 - y0
    return (dx * dx + dy * dy) ** 0.5 >= FLOW_LINE_MIN_LENGTH


# =============================================================================
# 5. 页眉页脚判定辅助
# =============================================================================

def is_header_footer_band(y0: float, page_height: float, margin: float = PAGE_EDGE_MARGIN) -> bool:
    """
    判断一个文本带是否位于页面顶部/底部的页眉页脚区域。

    参数:
        y0: 文本带顶部 y 坐标
        page_height: 页面总高度
        margin: 边缘判定阈值（pt）

    返回:
        True 表示位于页眉页脚区域
    """
    return y0 < margin or y0 > page_height - margin
