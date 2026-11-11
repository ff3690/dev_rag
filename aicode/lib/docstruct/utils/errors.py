# -*- coding: utf-8 -*-
"""
errors.py - docstruct 异常体系
================================================

设计目标：
1. 让调用方（如 RAG 入库流程）可以"一次捕获、统一降级"；
2. 异常类型尽量细，便于定位是哪一个阶段出的问题；
3. 所有异常都继承自 ParseError，因此调用方只需：
       try:
           gate.parse(...)
       except ParseError as e:
           # 走降级路径（structure = None）

异常分级（从轻到重）：
    ParseError           - 基类，任何解析错误
    PDFOpenError         - PDF 无法打开/解密/损坏
    LayoutDetectError    - 版面识别阶段失败
    RecognizeError       - 内容提取阶段失败
    AssembleError        - 结构组装阶段失败
"""


class ParseError(Exception):
    """
    解析错误基类。

    捕获这一类即可覆盖所有解析异常，用于触发"降级回旧链路"。
    """
    def __init__(self, message: str, page_no: int = None, detail: dict = None):
        # super().__init__ 会把 message 存进异常实例，str(e) 能直接显示
        super().__init__(message)
        self.message = message          # 人类可读的错误描述
        self.page_no = page_no          # 出错页码（可为 None）
        self.detail = detail or {}      # 附加信息（如原始异常）


class PDFOpenError(ParseError):
    """PDF 文件无法打开、被加密或已损坏。"""
    pass


class LayoutDetectError(ParseError):
    """版面识别（LayoutDetector）阶段失败。"""
    pass


class RecognizeError(ParseError):
    """内容提取（Recognizer）阶段失败。"""
    pass


class AssembleError(ParseError):
    """结构组装（StructureAssembler）阶段失败。"""
    pass
