# -*- coding: utf-8 -*-
"""
llm_flow_recognizer.py - 流程图 LLM 识别通道（可选）
================================================

适用场景：
    当流程图不是矢量图（而是扫描件/位图，无法用 get_drawings 分析）时，
    把流程图区域渲染成图片，交给多模态 LLM 识别，返回结构化的
    nodes / edges / description。

开关控制：
    ParserConfig.flow_llm_enabled（默认 False，本 PDF 的矢量流程图
    用规则分析已足够，本通道属扩展能力）。

LLM 客户端接口约定（调用方提供）：
    client.recognize_image(image_path: str) -> dict
    返回格式：
    {
        "nodes": ["服务对象", "建立居民健康档案", ...],   # 节点文本列表
        "edges": [{"from": "服务对象", "to": "建立居民健康档案", "label": "预约"}],
        "description": "一段话描述该流程……",            # 语义化描述
    }

失败处理：
    - 客户端缺失 / 调用异常 / 返回格式非法 → 记录 W_FLOW_LLM_FAILED
    - 调用方（ParserGate）回退到规则分析结果
"""

from typing import List, Optional

import fitz  # PyMuPDF

from docstruct.schemas import Block, LayoutElement, ParseIssue
from docstruct.report import W_FLOW_LLM_FAILED
from docstruct.utils.logger import logger


class LLMFlowRecognizer:
    """
    流程图 LLM 识别通道。

    参数:
        llm_client: 多模态 LLM 客户端（需实现 recognize_image 方法）
        output_dir: 渲染图片保存目录
        report: 质量报告
        doc_id: 文档标识
        seq_counter: 共享块序号计数器
    """

    def __init__(self, llm_client=None, output_dir: str = "",
                 report=None, doc_id: str = "", seq_counter: list = None):
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.report = report
        self.doc_id = doc_id
        self._seq_counter = seq_counter if seq_counter is not None else [0]

    def recognize(self, element: LayoutElement, page) -> Optional[Block]:
        """
        渲染流程图区域为图片，调用 LLM 识别。

        参数:
            element: flowchart 版面元素
            page: fitz.Page 对象

        返回:
            Block: 识别成功返回 Block；失败返回 None（调用方回退规则结果）
        """
        if self.llm_client is None:
            logger.warning("LLM 流程图通道未配置客户端，跳过")
            return None

        # ---- 1. 渲染流程图内容区域为图片 ----
        try:
            #  内容包围盒裁剪图（flow_p{页号}.png，2 倍缩放），供 LLM 识别
            image_path = self._render_region(page, element)
        except Exception as exc:
            logger.error(f"流程图区域渲染失败: {exc}")
            return None

        # ---- 2. 调用 LLM ----
        try:
            # 调用多模态 LLM，要求返回 {nodes, edges, description}
            raw = self.llm_client.recognize_image(image_path)
            nodes, edges = self._validate(raw) #校验格式
        except Exception as exc:
            logger.warning(f"LLM 流程图识别失败: {exc}")
            if self.report is not None:
                self.report.add(ParseIssue(
                    level="warning", code=W_FLOW_LLM_FAILED,
                    message=f"第 {element.page_no} 页 LLM 流程图识别失败: {exc}",
                    page_no=element.page_no,
                ))
            return None

        # ---- 3. 构造 Block ----
        description = raw.get("description", "") if isinstance(raw, dict) else ""
        path_text = self._edges_to_text(nodes, edges)
        content = f"【流程图】{description}\n{path_text}"

        block = Block(
            block_id=f"{self.doc_id}__p{element.page_no}__{self._next_seq()}",
            type="flowchart",
            page_no=element.page_no,
            bbox=element.bbox,
            content=content,
            data={
                "title": description[:30],
                "nodes": [{"id": f"n{i}", "text": t} for i, t in enumerate(nodes)],
                "edges": edges,
                "path_text": path_text,
                "mermaid": "",
                "source_llm": True,
            },
            confidence=0.8,
            source="llm",
            image_path=image_path,
        )
        print(f"[DOCSTRUCT][EXTRACT] LLM 识别流程图成功: {len(nodes)}节点, {len(edges)}边")
        return block

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _next_seq(self) -> int:
        seq = self._seq_counter[0]
        self._seq_counter[0] += 1
        return seq

    def _render_region(self, page, element: LayoutElement) -> str:
        """
        把流程图的内容区域裁剪渲染为 PNG 图片。

        与规则通道 FlowchartRecognizer 共用同一套"内容包围盒"裁剪：
        先用矢量分析定位节点矩形，以节点并集为基准并把邻近文本 span 纳入
        边界，最后外扩 15pt —— 避免 LayoutDetector 的 flowchart bbox
        覆盖整页时把整页渲染进图片（传给 LLM 的输入图与保存的
        flow_001.png 都应是流程图本身，而非整页）。规则分析不可用时
        降级为 element.bbox。

        参数:
            page: fitz.Page
            element: 流程图版面元素

        返回:
            str: 图片文件路径
        """
        from docstruct.extract.flowchart_recognizer import FlowchartRecognizer

        # 复用规则通道的节点/文本分析来计算内容包围盒（不依赖 LLM 通道开关）
        rule = FlowchartRecognizer(
            output_dir=self.output_dir, report=self.report,
            doc_id=self.doc_id, seq_counter=self._seq_counter,
        )
        try:
            area = element.bbox
            spans = rule._collect_spans(page, area)
            nodes = rule._collect_nodes(page, area, spans)
            # 与规则通道同路径（flow_p{页号}.png）：内容包围盒裁剪图
            return rule._render_region(page, element, nodes=nodes, spans=spans)
        except Exception as exc:
            logger.warning(f"流程图内容包围盒计算失败，降级 element.bbox 渲染: {exc}")
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            clip = fitz.Rect(element.bbox.x0 - 10, element.bbox.y0 - 10,
                             element.bbox.x1 + 10, element.bbox.y1 + 10)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)  # 2 倍缩放提升清晰度
            filepath = os.path.join(self.output_dir,
                                    f"flow_p{element.page_no}.png")
            pix.save(filepath)
            return filepath

    @staticmethod
    def _validate(raw) -> tuple:
        """
        校验并规范化 LLM 返回的数据结构。

        参数:
            raw: LLM 返回的原始对象

        返回:
            (nodes: List[str], edges: List[FlowEdge])

        异常:
            ValueError: 返回格式非法时抛出
        """
        from docstruct.schemas import FlowEdge

        if not isinstance(raw, dict):
            raise ValueError(f"LLM 返回格式非法: {type(raw)}")

        nodes_raw = raw.get("nodes", [])
        edges_raw = raw.get("edges", [])
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            raise ValueError("nodes/edges 必须是列表")

        nodes = [str(n) for n in nodes_raw if str(n).strip()]
        edges = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            frm, to = e.get("from"), e.get("to")
            if not frm or not to:
                continue
            edges.append(FlowEdge(from_node=str(frm), to_node=str(to),
                                  label=str(e.get("label", ""))))
        return nodes, edges

    @staticmethod
    def _edges_to_text(nodes: List[str], edges: List) -> str:
        """把 edges 序列化为路径文本（LLM 通道的简化版，不做图遍历）。"""
        lines = ["流程路径："]
        for e in edges:
            arrow = " →"
            if e.label:
                arrow = f" →[{e.label}]"
            lines.append(f"{e.from_node}{arrow} {e.to_node}")
        return "\n".join(lines)
