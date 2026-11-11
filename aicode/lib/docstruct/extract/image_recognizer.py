# -*- coding: utf-8 -*-
"""
image_recognizer.py - 图片识别器
================================================

从图片区域提取图片本身（保存到磁盘）+ OCR 文本。

OCR 说明（可选依赖设计）：
    docstruct 设计原则是"不 import backend"，但 OCR 能力在 backend 的
    OCRProcessor 里。为兼顾"独立运行"与"集成复用"，采用可选导入：
        - 集成环境：能 import 到 backend.rag.ocr_processor → 走 OCR
        - 独立环境：OCR 不可用 → 只保留图片位置信息，记录 warning

图片保存：
    按 element.detail["xref"] 从 PDF 中提取原图，保存到 output_dir，
    image_path 字段记录保存路径，供人工抽检与 LLM 通道使用。
"""

import os

import fitz  # PyMuPDF

from docstruct.schemas import BlockType, LayoutElement, Block, ParseIssue
from docstruct.extract.base import BaseRecognizer
from docstruct.report import W_OCR_FAILED
from docstruct.utils.logger import logger


class ImageRecognizer(BaseRecognizer):
    """
    图片识别器。

    参数:
        doc_id: 文档标识
        report: 质量报告
        seq_counter: 共享块序号计数器
        output_dir: 图片保存目录（默认不保存）
        llm_client: 通用视觉 LLM 客户端（缺陷7，默认 None）。
            提供 describe_image(image_path) -> str 接口；
            OCR 无文本时调用生成图片语义描述，避免图片块不可检索。
    """

    def __init__(self, doc_id: str = "", report=None, seq_counter: list = None,
                 output_dir: str = "", llm_client=None):
        super().__init__(doc_id, report, seq_counter)
        self.output_dir = output_dir
        self.llm_client = llm_client

    def supports(self, block_type: BlockType) -> bool:
        return block_type == "image"

    def recognize(self, element: LayoutElement, page) -> Block:
        """
        提取图片并尝试 OCR（OCR 无文本且配置 LLM 客户端时降级描述）。

        参数:
            element: image 版面元素
            page: fitz.Page 对象

        返回:
            Block: data 含 ocr_text / llm_description / img_ext / image_path
        """
        xref = element.detail.get("xref")
        image_path = None
        ocr_text = ""
        llm_description = ""

        # ---- 1. 提取图片并保存（若配置了输出目录） ----
        if xref is not None and self.output_dir:
            try:
                image_path = self._save_image(page, xref, element)
            except Exception as exc:
                logger.warning(f"图片提取失败 xref={xref}: {exc}")

        # ---- 2. OCR（可选依赖） ----
        if image_path:
            ocr_text = self._ocr_image(image_path, element)

        # ---- 3. 缺陷7：OCR 无文本时，尝试 LLM 解读（可选） ----
        if image_path and not ocr_text and self.llm_client is not None:
            try:
                llm_description = self._describe_image(image_path, element)
            except Exception as exc:
                logger.warning(f"图片 LLM 解读失败: {exc}")
                self.report.add(ParseIssue(
                    level="warning", code="W_IMAGE_LLM_FAILED",
                    message=f"第 {element.page_no} 页图片 LLM 解读失败: {exc}",
                    page_no=element.page_no,
                    detail={"image_path": image_path},
                ))

        # ---- 4. 构造 Block ----
        content = "【图片】"
        if ocr_text:
            content += f"\n{ocr_text}"
        elif llm_description:
            # LLM 描述作为兜底语义（标记来源，便于溯源与后续人工校验）
            content += f"\n（图片语义描述）{llm_description}"
        if image_path:
            content += f"\n图片文件: {image_path}"

        data = {
            "ocr_text": ocr_text,
            "llm_description": llm_description,
            "img_ext": os.path.splitext(image_path)[1] if image_path else "",
            "image_path": image_path,
        }
        block = self._make_block(element, content=content, data=data)
        block.image_path = image_path
        print(f"[DOCSTRUCT][EXTRACT] 识别图片: xref={xref}, "
              f"OCR文本={len(ocr_text)}字, LLM描述={len(llm_description)}字")
        return block

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _save_image(self, page, xref: int, element: LayoutElement) -> str:
        """
        从 PDF 中按 xref 提取原图并保存到 output_dir。

        参数:
            page: 所在页（用于取图片扩展名）
            xref: 图片对象引用
            element: 版面元素（用于生成唯一文件名）

        返回:
            str: 保存后的文件绝对路径
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # 从页面对象中提取图片二进制数据
        info = page.parent.extract_image(xref)  # {"image": bytes, "ext": "png", ...}
        ext = info.get("ext", "png")
        filename = f"img_p{element.page_no}_x{xref}.{ext}"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(info["image"])
        return filepath

    def _describe_image(self, image_path: str, element: LayoutElement) -> str:
        """
        调用视觉 LLM 客户端生成图片语义描述（缺陷7）。

        参数:
            image_path: 图片文件路径
            element: 版面元素（用于报告定位）

        返回:
            str: 图片描述文本（失败抛异常，由 recognize 捕获降级）
        """
        if self.llm_client is None:
            return ""
        if hasattr(self.llm_client, "describe_image"):
            return (self.llm_client.describe_image(image_path) or "").strip()
        return ""

    def _ocr_image(self, image_path: str, element: LayoutElement) -> str:
        """
        调用 backend 的 OCRProcessor 识别图片文字（可选依赖）。

        若 OCR 不可用或失败，记录 warning 并返回空字符串。

        参数:
            image_path: 图片文件路径
            element: 版面元素（用于报告定位）

        返回:
            str: OCR 识别出的文本（失败为空字符串）
        """
        try:
            # 可选导入：集成环境中 backend 已加载 OCRProcessor
            from backend.rag.ocr_processor import OCRProcessor
            ocr = OCRProcessor()
            text = ocr.extract_text(image_path) if hasattr(ocr, "extract_text") else ""
            return text or ""
        except Exception as exc:
            logger.warning(f"图片 OCR 失败: {exc}")
            self.report.add(ParseIssue(
                level="warning", code=W_OCR_FAILED,
                message=f"第 {element.page_no} 页图片 OCR 失败: {exc}",
                page_no=element.page_no,
                detail={"image_path": image_path},
            ))
            return ""
