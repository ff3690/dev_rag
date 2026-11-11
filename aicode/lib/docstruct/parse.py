# -*- coding: utf-8 -*-
"""
parse.py - ParserGate 解析门面（对外统一入口）
================================================

编排完整解析流水线：
    PDF 打开 → LayoutDetector（版面识别）
             → Recognizers（内容提取）
             → 跨页表格合并
             → StructureAssembler（建树/渲染/报告）
             → 中间产物落盘（md/json，可选）

对外提供两个接口：
    parse(pdf_path, doc_id)                 # 解析，失败抛 ParseError
    parse_with_fallback(pdf_path, doc_id)   # 解析，失败返回 (None, 原因)
        —— 后者供 RAG 集成使用：解析失败自动降级回旧纯文本链路。

典型使用：
    from docstruct.parse import ParserGate
    gate = ParserGate()
    sd = gate.parse("规范.pdf", doc_id="doc001")
    print(sd.title, len(sd.blocks))
"""

import os
import time
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from docstruct.config import ParserConfig
from docstruct.layout.detector import LayoutDetector
from docstruct.extract import get_recognizer
from docstruct.extract.base import BaseRecognizer
from docstruct.extract.table_recognizer import TableRecognizer
from docstruct.extract.flowchart_recognizer import FlowchartRecognizer
from docstruct.extract.llm_flow_recognizer import LLMFlowRecognizer
from docstruct.assemble.assembler import StructureAssembler
from docstruct.assemble.render import _jsonable
from docstruct.schemas import Block, ParseIssue, ParseReport, StructuredDocument
from docstruct.utils.errors import ParseError, PDFOpenError
from docstruct.utils.logger import logger


class ParserGate:
    """
    解析门面：对外统一入口。

    参数:
        config: ParserConfig 配置（默认使用库内默认值）。
                RAG 集成时由调用方传入 settings 对应的配置。
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config if config is not None else ParserConfig()
        # 每次 parse 都会新建 report（避免多次解析的状态污染）
        self.report: Optional[ParseReport] = None
        # 共享块序号计数器：所有 Recognizer 共用，保证 block_id 全篇唯一
        self._seq_counter: Optional[list] = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def parse(self, pdf_path: str, doc_id: str = "") -> StructuredDocument:
        """
        执行完整解析。

        参数:
            pdf_path: PDF 文件路径
            doc_id: 文档标识（建议传 backend 的 Document.id；空则用文件名）

        返回:
            StructuredDocument: 结构化中间产物

        异常:
            ParseError: 解析失败（PDF 无法打开 / 版面识别失败等）

        完整调用链
          └─ ParserGate.parse(pdf_path, doc_id)          ← parse.py:64
            ├─ ① _open_pdf()                           打开 PDF（PyMuPDF）
            ├─ ② LayoutDetector.detect()               版面识别 → 产出 LayoutElement 列表
            ├─ ③ _extract_blocks()                     ★ 识别器在这里被调用，"用 LLM 识别流程图"也在这里调用
            ├─ ④ TableRecognizer.merge_cross_page()    跨页表格合并（静态方法，另一次调用）
            ├─ ⑤ StructureAssembler.assemble()         建结构树/渲染/报告
            └─ ⑥ _persist()                            中间产物落盘
        """
        t0 = time.time()
        if not doc_id:
            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.report = ParseReport()
        self._seq_counter = [0]

        print(f"\n[DOCSTRUCT][PARSE] ===== 开始解析: {pdf_path} (doc_id={doc_id}) =====")

        # ---- 1. 打开 PDF ----
        pdf_doc = self._open_pdf(pdf_path)

        try:
            # ---- 2. 版面识别 ----
            detector = LayoutDetector(pdf_doc, self.report)
            elements = detector.detect()

            # ---- 3. 内容提取 ----
            blocks = self._extract_blocks(pdf_doc, elements, doc_id)

            # ---- 4. 跨页表格合并 ----
            blocks = TableRecognizer.merge_cross_page(blocks, self.report)

            # ---- 5. 结构组装 ----
            title = (pdf_doc.metadata or {}).get("title") or \
                os.path.splitext(os.path.basename(pdf_path))[0]
            assembler = StructureAssembler(self.config.parser_version, self.report)
            structured = assembler.assemble(blocks, doc_id, pdf_path, title=title)

            # ---- 6. 中间产物落盘（md/json） ----
            self._persist(structured)

            elapsed = time.time() - t0
            n_err = len(self.report.errors)
            n_warn = len(self.report.warnings)
            print(f"[DOCSTRUCT][PARSE] ===== 解析完成: {len(blocks)}块, "
                  f"{n_warn}个警告, {n_err}个错误, 耗时{elapsed:.1f}s =====")
            return structured

        finally:
            pdf_doc.close()

    def parse_with_fallback(self, pdf_path: str, doc_id: str = "") -> Tuple[Optional[StructuredDocument], Optional[str]]:
        """
        带降级的解析：任何异常都不会抛出，而是返回原因。

        参数:
            pdf_path: PDF 文件路径
            doc_id: 文档标识

        返回:
            (StructuredDocument 或 None, 降级原因或 None)
            成功 → (sd, None)；失败 → (None, "原因描述")
        """
        try:
            return self.parse(pdf_path, doc_id), None
        except ParseError as exc:
            logger.error(f"[docstruct] 结构化解析失败: {exc.message}")
            return None, exc.message
        except Exception as exc:  # 兜底：任何未知异常都走降级
            logger.error(f"[docstruct] 结构化解析异常: {exc}")
            return None, f"未预期异常: {exc}"

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _open_pdf(self, pdf_path: str):
        """
        打开 PDF 并做基础检查（加密/页数）。

        参数:
            pdf_path: PDF 路径

        返回:
            fitz.Document

        异常:
            PDFOpenError: 文件不存在 / 无法打开 / 需要密码
        """
        if not os.path.exists(pdf_path):
            raise PDFOpenError(f"文件不存在: {pdf_path}")
        try:
            pdf_doc = fitz.open(pdf_path)
        except Exception as exc:
            raise PDFOpenError(f"无法打开 PDF: {exc}", detail={"path": pdf_path})
        if pdf_doc.needs_pass:
            pdf_doc.close()
            raise PDFOpenError(f"PDF 已加密需要密码: {pdf_path}")
        if pdf_doc.page_count == 0:
            pdf_doc.close()
            raise PDFOpenError(f"PDF 无有效页: {pdf_path}")
        print(f"[DOCSTRUCT][PARSE] PDF 已打开: {pdf_doc.page_count} 页")
        return pdf_doc

    def _extract_blocks(self, pdf_doc, elements: List, doc_id: str) -> List[Block]:
        """
        遍历版面元素，用对应 Recognizer 提取内容块。

        流程图优先尝试 LLM 通道（若启用），失败回退规则通道。

        参数:
            pdf_doc: fitz.Document
            elements: 版面元素列表
            doc_id: 文档标识

        返回:
            List[Block]: 内容块（按阅读顺序）
        """
        # 一次性实例化所有识别器（共享 doc_id / report / seq_counter）
        recognizers: dict = self._build_recognizers(doc_id)

        # 可选：LLM 流程图通道
        llm_flow = None
        if self.config.flow_llm_enabled:    # 开关默认 False
            llm_flow = LLMFlowRecognizer(
                llm_client=self.config.flow_llm_client,
                output_dir=self.config.flow_image_dir,
                report=self.report, doc_id=doc_id, seq_counter=self._seq_counter,
            )

        blocks: List[Block] = []
        for el in elements:
            page = pdf_doc[el.page_no - 1]
            recognizer: BaseRecognizer = recognizers.get(el.type)
            if recognizer is None:
                continue  # 页眉页脚等无需提取

            # 流程图：若启用 LLM 通道则优先尝试
            if el.type == "flowchart" and llm_flow is not None:
                llm_block = llm_flow.recognize(el, page)    # ★ LLM 识别在此执行
                if llm_block is not None:
                    blocks.append(llm_block)
                    continue    # 失败 → 回退规则通道
                # LLM 失败 → 回退规则通道（下方执行）

            try:
                block = recognizer.recognize(el, page)  # FlowchartRecognizer（规则）
                # 识别器返回 None = 该元素无有效内容（如"空表格"），直接跳过
                if block is None:
                    continue
                # ---- 缺陷5：规则通道低置信度自动 LLM 复核（默认关）----
                # 规则识别置信度 < 0.6（节点/边过少）且启用 LLM 通道时，
                # 用 LLM 对流程图区域图片复核并替换规则结果，弥补规则盲区。
                if (el.type == "flowchart" and llm_flow is not None
                        and self.config.flow_llm_auto_recheck
                        and getattr(block, "confidence", 1.0) < 0.6):
                    recheck_block = llm_flow.recognize(el, page)
                    if recheck_block is not None:
                        # 标记来源：规则低置信度 → LLM 复核替换
                        recheck_block.data["rechecked_by_llm"] = True
                        recheck_block.data["rule_confidence"] = block.confidence
                        print(f"[DOCSTRUCT][PARSE] 流程图低置信度 LLM 复核通过: "
                              f"第 {el.page_no} 页 (规则置信度={block.confidence})")
                        blocks.append(recheck_block)
                    else:
                        blocks.append(block)  # LLM 复核失败保留规则结果
                else:
                    blocks.append(block)
            except Exception as exc:
                logger.warning(f"第 {el.page_no} 页 {el.type} 提取失败: {exc}")
                self.report.add(ParseIssue(
                    level="warning", code="W_RECOGNIZE_FAILED",
                    message=f"第 {el.page_no} 页 {el.type} 提取失败: {exc}",
                    page_no=el.page_no,
                ))
        return blocks

    def _build_recognizers(self, doc_id: str) -> dict:
        """
        创建全部 Recognizer 实例（共享序号计数器）。

        参数:
            doc_id: 文档标识

        返回:
            {BlockType: BaseRecognizer}
        """
        from docstruct.extract.text_recognizer import TextRecognizer
        from docstruct.extract.checkbox_recognizer import CheckboxRecognizer
        from docstruct.extract.image_recognizer import ImageRecognizer

        return {
            "heading": TextRecognizer(doc_id, self.report, self._seq_counter),
            "paragraph": TextRecognizer(doc_id, self.report, self._seq_counter),
            "table": TableRecognizer(doc_id, self.report, self._seq_counter),
            "flowchart": FlowchartRecognizer(
                doc_id, self.report, self._seq_counter,
                output_dir=self.config.flow_image_dir),
            "image": ImageRecognizer(doc_id, self.report, self._seq_counter,
                                     output_dir=self.config.output_dir,
                                     llm_client=self.config.image_llm_client),
            "checkbox": CheckboxRecognizer(doc_id, self.report, self._seq_counter),
        }

    def _persist(self, sd: StructuredDocument) -> None:
        """
        把 md / json 中间产物落盘（便于调试与人工抽检）。

        参数:
            sd: 结构化文档
        """
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            base = os.path.join(self.config.output_dir, sd.doc_id)
            with open(base + ".md", "w", encoding="utf-8") as f:
                f.write(sd.markdown)
            with open(base + ".json", "w", encoding="utf-8") as f:
                import json
                json.dump(sd.json, f, ensure_ascii=False, indent=2, default=_jsonable)
            print(f"[DOCSTRUCT][PARSE] 中间产物已落盘: {base}.md / {base}.json")
        except Exception as exc:
            logger.warning(f"中间产物落盘失败: {exc}")
