# -*- coding: utf-8 -*-
"""
markdown_struct.py - Markdown 块级结构的数据结构定义
====================================================

为 Markdown 复杂文档（含表格/代码块/图片）提供与 docstruct 对等的
"结构化中间产物"，供 MdStructuredSplitter 做结构感知切片使用。

与 docstruct 的边界：
    - docstruct 解析 PDF；本模块族解析 Markdown（纯 Python，零第三方依赖）
    - MdStruct 挂在 Document.structure 上，splitter_key="md" 用于
      TextSplitter._split_structured_document 分流（区别于 PDF 的 docstruct）。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MdImageRef:
    """
    Markdown 单张图片引用（解析期占位，图意化/落盘阶段回填）。

    属性说明:
        index         - 文档内图片序号（从 0 起，稳定 ID）
        src           - 原始引用文本（data URI / 相对路径 / 外链 URL）
        alt           - 图片 alt 文本
        src_type      - 来源类型: path / base64 / url / missing
        ocr_text      - 内存 OCR 识别文本（图意化阶段回填）
        vl_description- 视觉大模型语义描述（auto/all 档回填）
        file_path     - 落盘物理路径（内容寻址 _cache 或文档目录）
        url           - 静态访问 URL（/md_images/...），供检索来源携带
        enhanced_by   - 图意化方式: none / explicit_text / ocr / vl
    """
    index: int = 0
    src: str = ""
    alt: str = ""
    src_type: str = "missing"
    ocr_text: str = ""
    vl_description: str = ""
    file_path: str = ""
    url: str = ""
    enhanced_by: str = "none"

    def combined_text(self) -> str:
        """图意文本（OCR + 视觉描述），供图片块切片/正文拼接使用。"""
        parts = []
        if self.ocr_text:
            parts.append(self.ocr_text.strip())
        if self.vl_description:
            parts.append(self.vl_description.strip())
        return "\n".join(parts)

    def has_semantic_text(self) -> bool:
        """是否已获得可用图意文本（OCR 或 VL）。"""
        return bool(self.combined_text())


@dataclass
class MdBlock:
    """
    Markdown 块级解析产物。

    属性说明:
        type        - heading/paragraph/table/code/image/list/blockquote
        level       - heading 层级（1-6）或列表嵌套层级
        content     - 纯文本化内容：
                      * 表格为空（数据在 header/rows，渲染在切片器完成）
                      * 段落/引用/普通列表 = 去 markdown 语法后的文本
        language    - code fence 语言（含 mermaid）；段落等为空
        section_path- 所属章节路径（如 "第一章 概述/1.1 背景"）
        caption     - 表格前紧邻段落截取（表格标题候选）
        header/rows - GFM 表格原始行列
        leading_text- 表格前紧邻说明段落（正文块）
        images      - 本块内图片引用（独立图片块 / 段落内嵌图片）
        has_explicit_enhance - 紧随该块已有 doc_to_md 增强（注释/表意引用/mermaid），
                               此时图意化跳过 OCR/VL（避免重复付费）
        pos         - 原文起始行号（0-based）
        start_offset- 原文起始字符偏移（溯源用，解析期计算）
        end_offset  - 原文结束字符偏移
    """
    type: str = "paragraph"
    level: int = 0
    content: str = ""
    language: str = ""
    section_path: str = ""
    caption: str = ""
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    leading_text: str = ""
    images: List[MdImageRef] = field(default_factory=list)
    has_explicit_enhance: bool = False
    pos: int = 0
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class MdStruct:
    """
    Markdown 结构化解析产物（挂在 Document.structure 上）。

    属性说明:
        doc_id          - 文档 ID（图意化/落盘时由 Document.id 回填）
        file_path       - 源 md 文件路径（相对图片解析基准目录）
        blocks          - 块级列表（按原文顺序）
        splitter_key    - 固定 "md"，TextSplitter 据此走 MdStructuredSplitter
        stats           - 解析统计（标题/表格/代码/图片计数等）
        warnings        - 解析期告警（不阻断入库）
        images_processed- 图片图意化/落盘是否已执行（防重复处理，幂等）
        image_stats     - 图片处理统计（OCR/VL/缓存命中数等）
    """
    doc_id: str = ""
    file_path: str = ""
    blocks: List[MdBlock] = field(default_factory=list)
    splitter_key: str = "md"
    stats: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    images_processed: bool = False
    image_stats: dict = field(default_factory=dict)


def iter_image_refs(struct: "MdStruct"):
    """遍历所有含图片的块及其图片引用（解析/图意化共用）。"""
    if not struct or not struct.blocks:
        return
    for block in struct.blocks:
        if block.images:
            yield block, block.images


def count_block_types(struct: "MdStruct") -> dict:
    """按块类型统计（写入 MdStruct.stats 用）。"""
    counter = {"heading": 0, "paragraph": 0, "table": 0, "code": 0,
               "image": 0, "list": 0, "blockquote": 0}
    if not struct or not struct.blocks:
        return counter
    for blk in struct.blocks:
        counter[blk.type] = counter.get(blk.type, 0) + 1
    return counter
