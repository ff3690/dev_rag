# -*- coding: utf-8 -*-
"""
render.py - Markdown / JSON 渲染
================================================

把 StructureAssembler 建好的树渲染为两种可交付格式：
    - Markdown：人工阅读、调试、外部工具查看
    - JSON：落盘、传输、程序化处理

渲染规则：
    标题  -> "#" ~ "####"（按 level）
    正文  -> 原样文本
    表格  -> Block.content（识别阶段已渲染为 Markdown 表格）
    流程图-> Mermaid 代码块 + 路径文本
    复选框-> 条目清单
    图片  -> Markdown 图片语法
"""

from dataclasses import asdict, is_dataclass
from typing import Dict, List

from docstruct.schemas import Block, StructureNode


def _jsonable(value):
    """
    把任意值递归转换为可 JSON 序列化的对象。

    为什么需要：flowchart 块的 data["edges"] 里是 FlowEdge dataclass 对象，
    json.dump 无法直接序列化 dataclass。这里统一把 dataclass 转成 dict，
    其它类型（str/int/float/bool/None/嵌套 dict/list）原样保留。

    参数:
        value: 任意值

    返回:
        object: 可 JSON 序列化的等价对象
    """
    if is_dataclass(value) and not isinstance(value, type):
        # dataclass 实例 → 先转 dict，再递归清理嵌套值
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# =============================================================================
# Markdown 渲染
# =============================================================================

def render_markdown(tree: StructureNode) -> str:
    """
    把标题锚点树渲染为 Markdown 文档。

    参数:
        tree: 树根节点

    返回:
        str: Markdown 全文
    """
    lines: List[str] = []

    def walk(node: StructureNode) -> None:
        """深度优先遍历：标题 → 直属内容块 → 子节点。"""
        if node.title:
            lines.append("#" * node.level + " " + node.title)
        for block in node.blocks:
            rendered = render_block(block)
            if rendered:
                lines.append(rendered)
        for child in node.children:
            walk(child)

    walk(tree)
    # 用两个空行分隔块，保证 Markdown 段落/表格渲染正确
    return "\n\n".join(lines)


def render_block(block: Block) -> str:
    """
    渲染单个内容块。

    参数:
        block: 内容块

    返回:
        str: Markdown 片段（空块返回空串）
    """
    if block.type == "paragraph":
        return block.content
    if block.type == "table":
        # 表格在识别阶段已渲染为 Markdown 表格文本
        return block.content
    if block.type == "flowchart":
        parts = []
        mermaid = block.data.get("mermaid", "")
        if mermaid:
            parts.append(mermaid)
        path_text = block.data.get("path_text", "")
        if path_text:
            parts.append(path_text)
        return "\n\n".join(parts)
    if block.type == "checkbox":
        return block.content
    if block.type == "image":
        if block.image_path:
            return f"![图片]({block.image_path})"
        return "（图片，未保存文件）"
    return block.content


# =============================================================================
# JSON 渲染
# =============================================================================

def render_json(doc_id: str, title: str, parser_version: str,
                toc: List[Dict], tree: StructureNode,
                blocks: List[Block], stats: Dict) -> Dict:
    """
    渲染为 JSON 字典（可落盘/传输）。

    参数:
        doc_id: 文档标识
        title: 文档标题
        parser_version: 解析器版本
        toc: 目录列表
        tree: 树根节点
        blocks: 全部内容块
        stats: 解析统计

    返回:
        dict: 结构完整的 JSON 字典
    """
    return {
        "doc_id": doc_id,
        "title": title,
        "parser_version": parser_version,
        "toc": toc,
        "tree": node_to_dict(tree),
        "blocks": [block_to_dict(b) for b in blocks],
        "stats": stats,
    }


def node_to_dict(node: StructureNode) -> Dict:
    """
    树节点转字典（递归）。

    参数:
        node: 树节点

    返回:
        dict: 节点字典（含 children / blocks 递归）
    """
    return {
        "node_id": node.node_id,
        "title": node.title,
        "level": node.level,
        "heading_block_id": node.heading.block_id if node.heading else None,
        "children": [node_to_dict(c) for c in node.children],
        "blocks": [block_to_dict(b) for b in node.blocks],
    }


def block_to_dict(block: Block) -> Dict:
    """
    内容块转字典。

    参数:
        block: 内容块

    返回:
        dict: 块字典（bbox 平铺为 x0/y0/x1/y1，便于外部消费）
    """
    return {
        "block_id": block.block_id,
        "type": block.type,
        "page_no": block.page_no,
        "bbox": {
            "x0": block.bbox.x0, "y0": block.bbox.y0,
            "x1": block.bbox.x1, "y1": block.bbox.y1,
        },
        "content": block.content,
        "data": _jsonable(block.data),
        "parent_id": block.parent_id,
        "section_path": block.section_path,
        "confidence": block.confidence,
        "source": block.source,
        "image_path": block.image_path,
    }
