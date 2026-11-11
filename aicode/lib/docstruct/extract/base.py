# -*- coding: utf-8 -*-
"""
base.py - Recognizer 抽象基类
================================================

所有内容提取器（表格/流程图/文本/图片/复选框）都继承 BaseRecognizer，
统一实现以下约定：
    1. supports(block_type)：是否负责该类型；
    2. recognize(element, page)：从版面元素区域提取内容，产出 Block；
    3. block_id 自动编号（格式 {doc_id}__p{page}__{seq}），保证全篇唯一。
"""

from abc import ABC, abstractmethod

from docstruct.schemas import BlockType, Block, LayoutElement, ParseReport


class BaseRecognizer(ABC):
    """
    内容提取器抽象基类。

    参数:
        doc_id: 文档标识（用于生成唯一 block_id）
        report: 解析质量报告（用于记录提取阶段的问题）
    """

    def __init__(self, doc_id: str = "", report: ParseReport = None, seq_counter: list = None):
        self.doc_id = doc_id
        self.report = report if report is not None else ParseReport()
        # 块序号共享计数器（list 是可变对象，可跨实例共享）。
        # 注意：全篇有多个 Recognizer 实例（表格/文本/流程图……），
        #       若每个实例各自从 0 计数，生成的 block_id 会重复！
        #       因此 ParserGate 会创建一个共享计数器传给所有实例，
        #       保证 block_id 全篇唯一。
        self._seq_counter = seq_counter if seq_counter is not None else [0]

    # ------------------------------------------------------------------
    # 子类必须实现的方法
    # ------------------------------------------------------------------
    @abstractmethod
    def supports(self, block_type: BlockType) -> bool:
        """返回该识别器是否负责指定元素类型。"""

    @abstractmethod
    def recognize(self, element: LayoutElement, page) -> Block:
        """
        从版面元素区域提取内容，产出 Block。

        参数:
            element: 版面元素（含类型/位置/置信度）
            page: fitz.Page 对象（用于读取该区域的原始内容）

        返回:
            Block: 提取后的内容块
        """

    # ------------------------------------------------------------------
    # 公共工具方法
    # ------------------------------------------------------------------
    def _next_block_id(self, page_no: int) -> str:
        """
        生成唯一的块 ID。

        格式：{doc_id}__p{page_no}__{序号}，例如 "doc123__p12__3"

        参数:
            page_no: 页码

        返回:
            str: 唯一块 ID
        """
        seq = self._seq_counter[0]
        self._seq_counter[0] += 1
        return f"{self.doc_id}__p{page_no}__{seq}"

    def _make_block(self, element: LayoutElement, content: str, data: dict,
                    confidence: float = None, block_type: str = None) -> Block:
        """
        按标准字段构造 Block（减少各子类的重复代码）。

        参数:
            element: 版面元素（提供类型/页码/位置/来源）
            content: 块的主文本内容
            data: 类型专属结构化数据
            confidence: 置信度（默认沿用 element.confidence）
            block_type: 覆盖块类型（默认沿用 element.type）。
                用途：识别器发现元素实际内容与版面判定不符时（如把
                "目录清单"误判成标题），降级为其他类型（如 paragraph）。

        返回:
            Block: 构造好的内容块
        """
        return Block(
            block_id=self._next_block_id(element.page_no),
            type=block_type or element.type,
            page_no=element.page_no,
            bbox=element.bbox,
            content=content,
            data=data,
            confidence=confidence if confidence is not None else element.confidence,
            source=element.source,
        )
