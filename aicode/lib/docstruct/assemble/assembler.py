# -*- coding: utf-8 -*-
"""
assembler.py - StructureAssembler 结构组装器
================================================

职责：把识别出的扁平内容块（Block[]）组装为**标题锚点树**，
并计算每个块的章节路径（section_path）、生成目录（toc）、
渲染 Markdown / JSON 中间产物。

树的层级（以《国家基本公共卫生服务规范》为例）：
    level 1  第一部分 居民健康档案管理
    ├─ level 2  一、服务对象
    │   └─ level 3  （一）X
    │       └─ level 4  1. X
    └─ level 2  二、服务内容
        ├─ （内容块：表格 / 正文 / 流程图……）

树构建算法（栈式）：
    遍历 blocks，遇到 heading 时：
        - 若新标题层级 > 栈顶层级 → 挂为栈顶的子节点，入栈；
        - 否则 → 不断出栈直到栈顶层级 < 新标题层级，再挂为子节点。
    非 heading 块 → 挂到当前栈顶节点（即最近的标题之下）。
"""

from typing import Dict, List, Optional

from docstruct.schemas import Block, ParseIssue, ParseReport, StructureNode, StructuredDocument
from docstruct.assemble import render
from docstruct.report import E_ASSEMBLE_NO_HEADING
from docstruct.utils.logger import logger


class StructureAssembler:
    """
    结构组装器。

    参数:
        parser_version: 解析器版本号（写入产物）
        report: 质量报告
    """

    def __init__(self, parser_version: str = "1.0.0", report: Optional[ParseReport] = None):
        self.parser_version = parser_version
        self.report = report if report is not None else ParseReport()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def assemble(self, blocks: List[Block], doc_id: str,
                 file_path: str, title: str = "") -> StructuredDocument:
        """
        组装结构化文档。

        参数:
            blocks: 全部内容块（按阅读顺序）
            doc_id: 文档标识
            file_path: 源文件路径
            title: 文档标题（可空，自动推断）

        返回:
            StructuredDocument: 结构化中间产物
        """
        if not title:
            title = self._infer_title(blocks)

        # ---- 1. 建树（含无标题兜底） ----
        tree = self._build_tree(blocks)
        if not tree.title and not tree.children and not tree.blocks:
            # 连标题都没有 → 单层兜底：全部块挂根
            tree.blocks = [b for b in blocks if b.type != "heading"]
            self.report.add(ParseIssue(
                level="error", code=E_ASSEMBLE_NO_HEADING,
                message="未检测到任何标题，全部内容按单层结构组织",
            ))
            print("[DOCSTRUCT][ASSEMBLE] 未检测到标题，使用单层兜底结构")

        # ---- 2. 计算章节路径 ----
        self._assign_section_paths(tree, [])

        # ---- 3. 生成目录 ----
        toc = self._build_toc(tree)

        # ---- 4. 渲染 Markdown 与 JSON ----
        markdown = render.render_markdown(tree)
        json_doc = render.render_json(
            doc_id=doc_id, title=title, parser_version=self.parser_version,
            toc=toc, tree=tree, blocks=blocks, stats=self.report.stats,
        )

        print(f"[DOCSTRUCT][ASSEMBLE] 组装完成: 目录{len(toc)}项, "
              f"块{len(blocks)}个, Markdown {len(markdown)}字符")

        return StructuredDocument(
            doc_id=doc_id,
            file_path=file_path,
            title=title,
            parser_version=self.parser_version,
            toc=toc,
            blocks=blocks,
            tree=tree,
            markdown=markdown,
            json=json_doc,
            report=self.report,
        )

    # ------------------------------------------------------------------
    # 树构建
    # ------------------------------------------------------------------
    def _build_tree(self, blocks: List[Block]) -> StructureNode:
        """
        用栈式算法把扁平 blocks 组织为标题锚点树。

        参数:
            blocks: 内容块列表

        返回:
            StructureNode: 树根节点（title 为空，代表文档根）
        """
        # 根节点：level=0，代表"文档本身"
        root = StructureNode(node_id="root", title="", level=0, heading=None)
        stack: List[StructureNode] = [root]
        node_seq = [0]  # 节点编号计数器（用 list 便于闭包修改）

        for block in blocks:
            if block.type == "heading":
                level = int(block.data.get("level", 2))
                # 出栈：找到 level 更低的父节点（根节点 level=0 兜底）
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1]

                # 创建新节点并挂到父节点
                node_seq[0] += 1
                node = StructureNode(
                    node_id=f"n{node_seq[0]:04d}",
                    title=block.content,
                    level=level,
                    heading=block,
                )
                parent.children.append(node)
                # 标题块的父指向父节点的标题块（根节点无标题块）
                block.parent_id = parent.heading.block_id if parent.heading else None
                stack.append(node)
            else:
                # 非标题块：挂到当前最近标题节点下
                target = stack[-1]
                target.blocks.append(block)
                block.parent_id = target.heading.block_id if target.heading else None

        return root

    # ------------------------------------------------------------------
    # 章节路径 / 目录 / 标题推断
    # ------------------------------------------------------------------
    def _assign_section_paths(self, node: StructureNode, parent_path: List[str]) -> None:
        """
        递归计算每个节点的章节路径，并回填到其直属内容块。

        参数:
            node: 当前节点
            parent_path: 父节点的路径段列表（不含当前节点）
        """
        node_path = parent_path + ([node.title] if node.title else [])
        node.section_path = "/".join(node_path)

        # 内容块的章节路径 = 所在节点路径
        for block in node.blocks:
            block.section_path = node.section_path

        # 递归子节点
        for child in node.children:
            self._assign_section_paths(child, node_path)

    def _build_toc(self, root: StructureNode) -> List[Dict]:
        """
        深度优先收集目录。

        参数:
            root: 树根

        返回:
            [{"level", "title", "node_id", "page_no"}, ...]
        """
        toc: List[Dict] = []

        def walk(node: StructureNode) -> None:
            if node.title:
                toc.append({
                    "level": node.level,
                    "title": node.title,
                    "node_id": node.node_id,
                    "page_no": node.heading.page_no if node.heading else None,
                })
            for child in node.children:
                walk(child)

        walk(root)
        return toc

    def _infer_title(self, blocks: List[Block]) -> str:
        """
        推断文档标题：取第一个标题块（level 1 优先），否则取文件名。

        参数:
            blocks: 内容块

        返回:
            str: 推断的标题
        """
        level1 = [b for b in blocks if b.type == "heading" and b.data.get("level") == 1]
        if level1:
            return level1[0].content
        headings = [b for b in blocks if b.type == "heading"]
        if headings:
            return headings[0].content
        return "未命名文档"
