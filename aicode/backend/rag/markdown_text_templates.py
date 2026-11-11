# -*- coding: utf-8 -*-
"""
markdown_text_templates.py - Markdown 结构块的"向量化文本模板"
=============================================================

对齐 docstruct/slice/text_templates.py 的设计原则：
    1. 上下文自包含：每个块都能独立回答"这是哪里的什么内容"；
    2. 结构化可读：用【标记】前缀区分字段；
    3. 模板与算法分离。

同时提供两个纯文本渲染入口：
    - render_block_to_text(block)  : 单块 -> 向量文本（切片器逐块调用）
    - render_struct_to_text(struct): 整篇 -> 纯文本（doc.content 用，
      使图谱/旧链路同样受益于图意化文本）
"""

import re
from typing import List, Optional

from .markdown_struct import MdBlock, MdImageRef, MdStruct

# ---------------------------------------------------------------------------
# 表格渲染
# ---------------------------------------------------------------------------


def render_table_markdown(header: Optional[List[str]],
                          rows: Optional[List[List[str]]],
                          caption: str = "") -> str:
    """
    把 GFM 表格原始行列渲染为规范 Markdown 表格文本。

    设计：原始 md 里单元格可能有换行/竖线转义等，切片前统一规范化为
    "单行可读"的表格，避免 embedding 输入混杂原始语法噪音。
    """
    def _clean(cell: str) -> str:
        if cell is None:
            return ""
        # 去掉单元格内的块级换行（折行展示），转义竖线还原
        text = re.sub(r"\s*\|\s*", " ", str(cell))
        text = text.replace("\\|", "|")
        text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
        return " ".join(text.split())

    lines: List[str] = []
    if caption:
        lines.append(f"表标题：{caption}")
    if not header and not rows:
        return "\n".join(lines)
    if header:
        lines.append("| " + " | ".join(_clean(h) for h in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows or []:
        lines.append("| " + " | ".join(_clean(c) for c in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 单块模板
# ---------------------------------------------------------------------------

IMAGE_TEMPLATE = """【图片】{section_path}
{alt_part}{ocr_part}{vl_part}{loc_part}"""

TABLE_TEMPLATE = """【表格】{caption_part}
【所属章节】{section_path}
{leading_part}{markdown_table}"""

CODE_TEMPLATE = """【代码块】{section_path}（{language}）
{code}"""

HEADING_TEMPLATE = "{hashes} {title}\n【章节路径】{section_path}"

PARAGRAPH_PREFIX = "【所属章节】{section_path}\n"

CHECKBOX_TEMPLATE = """【清单】{section_path}
{items}"""


def _cap(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _img_body(img: MdImageRef, section_path: str) -> str:
    """拼接图片块正文（不含开头的【图片】标记）。"""
    lines = []
    if section_path:
        lines.append(f"【所属章节】{section_path}")
    if img.alt.strip():
        lines.append(f"【图片说明】{img.alt.strip()}")
    if img.ocr_text:
        lines.append(f"【OCR识别】{img.ocr_text.strip()}")
    if img.vl_description:
        lines.append(f"【AI解读】{img.vl_description.strip()}")
    if img.url:
        lines.append(f"【图片位置】{img.url}")
    if not img.alt.strip() and not img.ocr_text and not img.vl_description:
        lines.append("（该图片未识别出文字内容）")
    return "\n".join(lines)


def render_image_text(img: MdImageRef, section_path: str) -> str:
    """单张图片的完整文本（章节 + alt + OCR + VL + 位置），无语义时保留占位。"""
    return "【图片】" + _img_body(img, section_path)


def render_block_to_text(block: MdBlock) -> str:
    """
    把单个 MdBlock 渲染为可向量化的文本（切片器逐块调用）。

    说明：表格块在此阶段不做整表/行组拆分（那是切片器的职责），
    这里只产出"含上下文前缀"的整表 markdown；切片器超限时用
    split_table 相关辅助自行截取。
    """
    t = block.type
    if t == "heading":
        hashes = "#" * (block.level or 1)
        return HEADING_TEMPLATE.format(
            hashes=hashes, title=block.content,
            section_path=block.section_path)
    if t == "code":
        return CODE_TEMPLATE.format(
            section_path=block.section_path or "（未标注章节）",
            language=block.language or "code",
            code=block.content)
    if t == "table":
        caption = block.caption or ""
        md = render_table_markdown(block.header, block.rows, caption="")
        leading = ""
        if block.leading_text:
            leading = f"{block.leading_text}\n"
        return TABLE_TEMPLATE.format(
            caption_part=f"表标题：{caption}" if caption else "表格",
            section_path=block.section_path or "（未标注章节）",
            leading_part=leading,
            markdown_table=md)
    if t == "image":
        img = block.images[0] if block.images else None
        if img is not None:
            return render_image_text(img, block.section_path)
        return "【图片】" + (f"（图片说明：{block.content}）" if block.content else "")
    if t in ("list", "blockquote", "paragraph"):
        text = block.content
        if not text.strip():
            return ""
        # 统一加章节前缀（与 docstruct PARAGRAPH_PREFIX 语义一致）
        prefix = PARAGRAPH_PREFIX.format(
            section_path=block.section_path or "（未标注章节）")
        return prefix + text
    # 未知类型兜底
    return block.content or ""


def render_struct_to_text(struct: MdStruct) -> str:
    """
    整篇 md -> 纯文本（doc.content 用）。

    - 图片块：渲染 OCR/VL/alt/位置（若已图意化）；
    - 表格块：渲染为规范 markdown 表格；
    - 正文块：带章节前缀，纯文本无 md 语法噪音。
    """
    if not struct or not struct.blocks:
        return ""
    sections = []
    for blk in struct.blocks:
        text = render_block_to_text(blk)
        if text and text.strip():
            sections.append(text.strip())
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Token 估算（与 bge 模型的近似字符预算，复用 docstruct 量级约定）
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    估算文本 token 数（中文约 1 字/1 token，ASCII 约 1 token/2 字符）。

    与 docstruct.slice.tokenizer 量级一致，无需第三方 tokenizer。
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    cjk_chars = len(text) - ascii_chars
    return int(cjk_chars + ascii_chars * 0.5 + 0.999)


def split_long_text(text: str, max_tokens: int,
                    max_overlap_tokens: int = 64,
                    section_path: str = "") -> List[str]:
    """
    将正文/代码长文本按句子边界二次切分（段落前缀会重复注入）。

    分隔优先级：段落 -> 句子（。！？；） -> 子句（，、；） -> 硬切。
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    out: List[str] = []
    parts = re.split(r"(\n\s*\n)", text)
    buf: List[str] = []
    buf_tokens = 0
    prefix_tokens = estimate_tokens(PARAGRAPH_PREFIX.format(section_path=section_path)) if section_path else 0
    budget = max_tokens - prefix_tokens

    def _flush():
        nonlocal buf, buf_tokens
        if buf:
            piece = "".join(buf)
            if section_path:
                piece = PARAGRAPH_PREFIX.format(section_path=section_path) + piece
            out.append(piece)
        buf = []
        buf_tokens = 0

    for part in parts:
        if not part:
            continue
        part_tokens = estimate_tokens(part)
        if buf_tokens + part_tokens <= budget:
            buf.append(part)
            buf_tokens += part_tokens
            continue
        _flush()
        if part_tokens > budget:
            # 超长段落：按句子/子句/硬切逐级降级
            for sub in _split_hard(part, budget):
                out.append((PARAGRAPH_PREFIX.format(section_path=section_path) + sub) if section_path else sub)
        else:
            buf = [part]
            buf_tokens = part_tokens

    _flush()
    return out or [text]


_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")
_CLAUSE_SPLIT = re.compile(r"(?<=[，、；,;：:])")


def _split_hard(text: str, max_tokens: int) -> List[str]:
    """对超长文本逐级降级切分，保证每段不超过 max_tokens（以字符预算近似）。"""
    max_chars = max(32, int(max_tokens * 1.8))  # 中英混合保守预算

    def _char_join(chunks: List[str]) -> List[str]:
        result: List[str] = []
        buf = ""
        for c in chunks:
            if estimate_tokens(c) > max_tokens:
                if buf:
                    result.append(buf)
                    buf = ""
                # 硬切
                cur = ""
                for ch in c:
                    cur += ch
                    if estimate_tokens(cur) >= max_tokens:
                        result.append(cur)
                        cur = ""
                if cur:
                    result.append(cur)
            elif estimate_tokens(buf + c) <= max_tokens:
                buf += c
            else:
                result.append(buf)
                buf = c
        if buf:
            result.append(buf)
        return result

    # 先按句子，再按子句，最后硬切
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) > 1:
        return _char_join(sentences)
    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c.strip()]
    if len(clauses) > 1:
        return _char_join(clauses)
    return _char_join([text[i:i + max_chars] for i in range(0, len(text), max_chars)])
