# -*- coding: utf-8 -*-
"""
docstruct - 文档结构化解析与结构感知切片库
================================================

本库用于将《国家基本公共卫生服务规范》这类 PDF（含正文、标题层级、
线框表格、矢量流程图、图片、复选框符号）解析为结构化中间产物
（树状 JSON + Markdown），并生成适合 RAG 向量化入库的切片。

设计原则：
1. 与现有 backend/rag/ 完全解耦（本库不 import backend 的任何模块）；
2. 全部为纯函数式处理（输入 PDF 路径 -> 输出结构化对象），可重跑、可单测；
3. 逐页容错：单页解析失败不影响其他页，通过 ParseReport 记录问题；
4. 自动降级：任何解析失败都可由调用方回退到旧的纯文本链路。

对外统一入口：
    from docstruct.parse import ParserGate

核心数据结构（schemas.py）：
    LayoutElement   - 第一层识别产物："哪里有哪类内容"（版面元素）
    Block           - 第二层提取产物："该内容的具体结构化数据"
    StructureNode   - 标题锚点树节点
    StructuredDocument - 最终结构化中间产物（挂在 Document.structure 上）
    Chunk           - 结构感知切片产物（后续转换为现有 TextChunk 入库）
"""

import os
import sys

# ---------------------------------------------------------------------------
# 把 aicode/lib 加入 sys.path（与 backend/rag/document_loader.py 的机制一致）
# ---------------------------------------------------------------------------
# 原因：pymupdf(fitz) 等本地第三方库放在 aicode/lib/ 目录下。
#       backend 在 import document_loader 时会注入该目录，但如果本库被
#       独立使用（例如单独跑测试、脱离 backend 运行），就需要自己注入，
#       否则 `import fitz` 会失败。
_LIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "lib",
)
if os.path.exists(_LIB_DIR) and _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# 对外暴露的核心 API，方便使用者"from docstruct import xxx"
from docstruct.schemas import (
    BBox,
    BlockType,
    LayoutElement,
    Block,
    FlowEdge,
    StructureNode,
    StructuredDocument,
    Chunk,
    ParseIssue,
    ParseReport,
)

__all__ = [
    "BBox",
    "BlockType",
    "LayoutElement",
    "Block",
    "FlowEdge",
    "StructureNode",
    "StructuredDocument",
    "Chunk",
    "ParseIssue",
    "ParseReport",
]

__version__ = "1.0.0"
