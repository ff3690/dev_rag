# -*- coding: utf-8 -*-
"""
flowchart_recognizer.py - 流程图识别器（矢量分析）
================================================

从流程图版面元素区域分析"节点 + 边（箭头关系）"，这是本方案
区别于普通 PDF 解析的关键能力 —— 流程图的价值在于**箭头指向关系**，
如果只转成纯文本，会丢失"什么条件下下一步做什么"的语义。

识别步骤：
    1. 节点 = 区域内的填充矩形 + 矩形内部文本
    2. 边   = 区域内连接两节点的线段（方向 = 线段起点节点 → 终点节点）
    3. 条件标签 = 线段中点附近最近的文本（如"血压高于正常值"）
    4. 生成两种文本化结果：
       - path_text：每条路径一行（DFS 展开），用于向量化检索
       - mermaid：Mermaid 图描述，用于人工查看/渲染

输出到 Block.data：
    {
        "title": 流程图标题（取区域上方最近的标题文本）,
        "nodes": [{"id": "n0", "text": "..."}],
        "edges": [FlowEdge, ...],
        "path_text": "...",
        "mermaid": "...",
    }
"""

import os
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from docstruct.schemas import (BBox, Block, BlockType, FlowEdge, LayoutElement, ParseIssue)
from docstruct.extract.base import BaseRecognizer
from docstruct.layout.heuristics import is_flow_line_segment, is_flow_node_rect
from docstruct.utils.logger import logger
from docstruct.utils.text_utils import span_text


class FlowchartRecognizer(BaseRecognizer):
    """矢量流程图识别器。"""

    def __init__(self, doc_id: str = "", report=None, seq_counter=None,
                 output_dir: str = ""):
        """
        参数:
            doc_id: 文档标识
            report: 解析质量报告
            seq_counter: 块序号共享计数器
            output_dir: 流程图区域图片保存目录。
                非空时，识别完成后会把流程图所在区域裁剪渲染为 PNG 图片，
                并记录到 Block.image_path（供 LLM 识别与 RAG 检索溯源）；
                为空则不保存图片（保持旧行为，向后兼容）。
        """
        super().__init__(doc_id, report, seq_counter)
        self.output_dir = output_dir

    def supports(self, block_type: BlockType) -> bool:
        return block_type == "flowchart"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def recognize(self, element: LayoutElement, page) -> Block:
        """
        分析流程图区域，提取节点与边。

        参数:
            element: flowchart 版面元素
            page: fitz.Page 对象

        返回:
            Block: data 见 schemas.Block 的 flowchart 约定
        """
        area = element.bbox

        # ---- 1. 收集区域内的文本 span（用于给节点/边命名） ----
        spans = self._collect_spans(page, area)

        # ---- 2. 收集节点矩形（填充矩形 + 内部文本） ----
        nodes = self._collect_nodes(page, area, spans)

        # ---- 3. 收集边（线段 + 方向 + 条件标签） ----
        edges = self._collect_edges(page, area, nodes, spans)

        # ---- 4. 构造输出 ----
        title = self._find_title(page, area, element.page_no)
        path_text = self.to_path_text(nodes, edges)
        mermaid = self.to_mermaid(nodes, edges)

        # 置信度：节点或边过少时降低置信度并打 warning
        confidence = element.confidence
        if len(nodes) < 3 or len(edges) < 2:
            confidence = 0.5
            self.report.add(ParseIssue(
                level="warning", code="W_FLOW_LOW_CONFIDENCE",
                message=f"第 {element.page_no} 页流程图识别置信度低: "
                        f"节点{len(nodes)}个, 边{len(edges)}条",
                page_no=element.page_no,
            ))

        content = f"【流程图】{title}\n" + path_text
        data = {
            "title": title,
            "nodes": [{"id": n["id"], "text": n["text"]} for n in nodes],
            "edges": edges,
            "path_text": path_text,
            "mermaid": mermaid,
        }
        print(f"[DOCSTRUCT][EXTRACT] 识别流程图: {title}, "
              f"{len(nodes)}节点, {len(edges)}边")
        block = self._make_block(element, content=content, data=data, confidence=confidence)

        # 渲染流程图区域为图片（供 LLM 识别 / RAG 检索溯源）。
        # 在规则识别阶段就完成裁剪保存，不依赖 LLM 通道开关（flow_llm_enabled）。
        if self.output_dir:
            try:
                image_path = self._render_region(page, element, nodes=nodes,
                                                 spans=spans)
                block.image_path = image_path
                block.data["image_path"] = image_path
                logger.info(f"第 {element.page_no} 页流程图区域图片: {image_path}")
            except Exception as exc:
                logger.warning(f"第 {element.page_no} 页流程图区域图片渲染失败: {exc}")
        return block

    # ------------------------------------------------------------------
    # 区域图片渲染（供 LLM 识别 / RAG 检索溯源）
    # ------------------------------------------------------------------
    def _render_region(self, page, element: LayoutElement,
                       nodes: Optional[List[Dict]] = None,
                       spans: Optional[List[Tuple[BBox, str]]] = None) -> str:
        """
        把流程图的实际内容区域裁剪渲染为 PNG 图片。

        LayoutDetector 的 flowchart 元素 bbox 有时会覆盖整页，
        因此这里以识别出的节点矩形合并出**内容包围盒**为基准，并把
        与内容包围盒邻近的文本 span（章节标题、条件标签、箭头旁文字）
        一并纳入边界——占位节点（未识别出文字的节点）的 rect 只覆盖
        文字区域，矩形框实际更大，仅用节点 rect 会把顶部边缘裁掉。
        最后外扩 15pt 容纳箭头与标签。图片文件名带页码
        （flow_p{page_no}.png），配合 Block.page_no / Chunk.metadata
        中的 page_no，可在 RAG 检索时定位到流程图对应的 PDF 页面。

        参数:
            page: fitz.Page 对象
            element: flowchart 版面元素
            nodes: 识别出的节点列表（每个含 "rect": BBox）。
                为 None 时退回使用 element.bbox。
            spans: 流程图区域内的文本 span 列表（(BBox, text)），
                用于扩展裁剪边界，避免顶部标题 / 边缘文字被裁掉。

        返回:
            str: 保存后的图片绝对路径
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if nodes:
            xs0 = min(n["rect"].x0 for n in nodes)
            ys0 = min(n["rect"].y0 for n in nodes)
            xs1 = max(n["rect"].x1 for n in nodes)
            ys1 = max(n["rect"].y1 for n in nodes)
        else:
            xs0, ys0, xs1, ys1 = (element.bbox.x0, element.bbox.y0,
                                  element.bbox.x1, element.bbox.y1)
        # 把与内容包围盒邻近的 span（章节标题 / 条件标签等）纳入边界，
        # 避免占位节点 rect 偏小导致顶部文字或标题被裁剪。
        # 注意：底部不向下扩展——底部说明文字通常已在节点区域内（节点
        # y1 已下探到说明行），若继续向下扩展会把流程图之后的正文段落
        # （如"四、服务要求"）误纳入。pad=15 已足够容纳向下箭头。
        if spans:
            margin_top = 40  # 顶部扩展（容纳标题 / 顶部箭头溢出）
            margin_lr = 30   # 左右扩展（容纳旁标、横向箭头）
            for sp_bbox, _ in spans:
                if (sp_bbox.x1 >= xs0 - margin_lr and sp_bbox.x0 <= xs1 + margin_lr
                        and sp_bbox.y1 >= ys0 - margin_top and sp_bbox.y0 <= ys1):
                    xs0 = min(xs0, sp_bbox.x0)
                    ys0 = min(ys0, sp_bbox.y0)
                    xs1 = max(xs1, sp_bbox.x1)
                    # ys1 不向下扩展（见上方注释）
        pad = 15  # 外扩像素，容纳箭头与条件标签
        clip = fitz.Rect(xs0 - pad, ys0 - pad, xs1 + pad, ys1 + pad)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)  # 2 倍缩放提升清晰度
        filepath = os.path.join(self.output_dir, f"flow_p{element.page_no}.png")
        pix.save(filepath)
        return filepath

    # ------------------------------------------------------------------
    # 路径序列化（核心输出，供切片与检索使用）
    # ------------------------------------------------------------------
    def to_path_text(self, nodes: List[Dict], edges: List[FlowEdge]) -> str:
        """
        把流程图序列化为"每条分支一行"的文本。

        例：
            流程路径：
            服务对象 →[预约或直接就诊] 建立居民健康档案
            建立居民健康档案 →[血压高于正常值] 转诊上级医院
            建立居民健康档案 →[血压正常] 纳入规范管理

        算法：
            1. 找入度为 0 的节点作为起点（没有则从任意节点开始）；
            2. 从每个起点做 DFS，每走到叶子输出一条路径；
            3. 孤立节点（无任何边）单独输出一行。

        参数:
            nodes: 节点列表 [{"id": "n0", "text": "..."}]
            edges: 边列表 [FlowEdge, ...]

        返回:
            str: 路径序列化文本
        """
        if not nodes:
            return "（流程图未识别到节点）"

        node_text = {n["id"]: n["text"] for n in nodes}
        # 邻接表：from -> [(to, label)]
        adj: Dict[str, List[Tuple[str, str]]] = {n["id"]: [] for n in nodes}
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}
        # 边标签表：(from, to) -> label，用于路径渲染时取条件文本
        edge_label: Dict[Tuple[str, str], str] = {}
        for e in edges:
            if e.from_node in adj and e.to_node in adj:
                adj[e.from_node].append((e.to_node, e.label))
                in_degree[e.to_node] += 1
                edge_label[(e.from_node, e.to_node)] = e.label

        # 起点：入度为 0；若全有入度则取邻接表键序第一个
        starts = [nid for nid, deg in in_degree.items() if deg == 0]
        if not starts and nodes:
            starts = [nodes[0]["id"]]

        lines: List[str] = []
        visited_edge = set()  # 防止循环图的死循环

        def dfs(node_id: str, path: List[str]):
            """深度优先：path 累积节点 id 序列（含当前节点），到叶子统一渲染。"""
            if not adj[node_id]:
                # 到达叶子，渲染整条路径（每个节点只出现一次）
                parts: List[str] = []
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    label = edge_label.get((a, b), "")
                    arrow = f" →[{label}]" if label else " →"
                    parts.append(f"{node_text[a]}{arrow}")
                parts.append(node_text[path[-1]])
                lines.append("".join(parts))
                return
            for to_id, label in adj[node_id]:
                edge_key = (node_id, to_id)
                if edge_key in visited_edge:
                    continue  # 有环图中避免重复
                visited_edge.add(edge_key)
                dfs(to_id, path + [to_id])

        for start in starts:
            dfs(start, [start])

        # 孤立节点（没有任何边的节点）
        connected = set()
        for e in edges:
            connected.add(e.from_node)
            connected.add(e.to_node)
        for n in nodes:
            if n["id"] not in connected:
                lines.append(f"{node_text[n['id']]}（独立节点，无后续流程）")

        text = "流程路径：\n" + "\n".join(lines) if lines else "（流程图无有效路径）"
        return text

    # ------------------------------------------------------------------
    # Mermaid 序列化（人工查看）
    # ------------------------------------------------------------------
    def to_mermaid(self, nodes: List[Dict], edges: List[FlowEdge]) -> str:
        """
        生成 Mermaid flowchart 描述，便于渲染查看。

        参数:
            nodes: 节点列表
            edges: 边列表

        返回:
            str: Mermaid 代码
        """
        lines = ["```mermaid", "flowchart TD"]
        for n in nodes:
            # 节点文本含特殊字符时用引号包裹
            txt = n["text"].replace('"', "'")
            lines.append(f'    {n["id"]}["{txt}"]')
        for e in edges:
            label = f'|{e.label}|' if e.label else ""
            lines.append(f"    {e.from_node}{label} --> {e.to_node}")
        lines.append("```")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _collect_spans(self, page, area: BBox) -> List[Tuple[BBox, str]]:
        """
        收集流程图区域内的全部文本 span。

        参数:
            page: fitz.Page
            area: 流程图区域

        返回:
            [(bbox, text), ...]
        """
        spans = []
        try:
            rawdict = page.get_text("rawdict")
        except Exception:
            return spans
        for block in rawdict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    text = span_text(sp).strip()  # 兼容不同 PyMuPDF 版本
                    if not text:
                        continue
                    r = fitz.Rect(sp["bbox"])
                    # 只保留与流程图区域相交的 span
                    if (r.x1 >= area.x0 and r.x0 <= area.x1
                            and r.y1 >= area.y0 and r.y0 <= area.y1):
                        spans.append((BBox(r.x0, r.y0, r.x1, r.y1), text))
        return spans

    def _collect_nodes(self, page, area: BBox,
                       spans: List[Tuple[BBox, str]]) -> List[Dict]:
        """
        收集流程图节点（填充矩形 + 内部文本）。

        参数:
            page: fitz.Page
            area: 流程图区域
            spans: 区域内文本 span

        返回:
            [{"id": "n0", "rect": BBox, "text": "..."}, ...]
        """
        nodes: List[Dict] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return nodes

        for item in drawings:
            if item.get("type") not in ("f", "fs", "fr", "frs"):
                continue
            if item.get("fill") is None:
                continue
            r = item.get("rect")
            if r is None:
                continue
            if not is_flow_node_rect(r.width, r.height):
                continue
            # 排除页面级背景矩形：面积接近整页时（如整页底色填充）
            # 视为"背景"而非节点。若不排除，所有线段端点都会归到它身上，
            # 导致 from == to，一条边都识别不出来。
            page_area = page.rect.width * page.rect.height
            if r.width * r.height >= page_area * 0.9:
                continue
            # 矩形必须在流程图区域内
            rect = BBox(r.x0, r.y0, r.x1, r.y1)
            if not (rect.x0 >= area.x0 - 1 and rect.y0 >= area.y0 - 1
                    and rect.x1 <= area.x1 + 1 and rect.y1 <= area.y1 + 1):
                continue

            # 节点文本 = 矩形内部的 span 拼接
            inner = []
            for sp_bbox, text in spans:
                if rect.contains(sp_bbox):
                    inner.append(text)
            node_text = "".join(inner).strip()

            # 矩形内没有文本的节点：用坐标做占位文本（可能是纯图形节点）
            if not node_text:
                node_text = f"节点({int(rect.x0)},{int(rect.y0)})"

            nodes.append({"id": f"n{len(nodes)}", "rect": rect, "text": node_text})

        # ---- 缺陷5：合并重叠矩形节点（同一节点被绘制成多个近重合矩形时）----
        # diag 实证：同一文本"您是在本辖区常住么？"被识别为两个几乎重合的
        # 矩形（n3/n4），产生 n3->n4、n4->n3 自环边和"您是在本/么？"碎片标签。
        # 规则：重叠面积 ≥ 较小矩形面积的 40%，或中心距离小于较短边一半，
        #       且文本非空且文本相同的矩形合并为一个（保留覆盖范围更大的矩形）。
        nodes = self._merge_overlapping_nodes(nodes)

        # 按阅读顺序排序（左上到右下）
        nodes.sort(key=lambda n: (round(n["rect"].y0 / 10), n["rect"].x0))
        # 排序后重新编号
        for idx, n in enumerate(nodes):
            n["id"] = f"n{idx}"
        return nodes

    @staticmethod
    def _merge_overlapping_nodes(nodes: List[Dict]) -> List[Dict]:
        """
        合并重叠矩形节点（缺陷5：重复矩形 -> 自环边 / 碎片标签）。

        参数:
            nodes: 节点列表 [{"id", "rect", "text"}]

        返回:
            合并去重后的节点列表（rect 取并集，文本取非空者优先）
        """
        if len(nodes) < 2:
            return nodes
        merged: List[Dict] = []
        for n in nodes:
            absorbed = False
            for m in merged:
                a, b = n["rect"], m["rect"]
                # 计算交叠面积
                ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
                iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
                inter = ix * iy
                a_area = (a.x1 - a.x0) * (a.y1 - a.y0)
                b_area = (b.x1 - b.x0) * (b.y1 - b.y0)
                if a_area <= 0 or b_area <= 0:
                    continue
                min_area = min(a_area, b_area)
                overlap_ratio = inter / min_area if min_area > 0 else 0.0
                # 中心距离阈值：小于较短边一半视为"几乎重合"
                cx_dist = abs((a.x0 + a.x1) / 2 - (b.x0 + b.x1) / 2)
                cy_dist = abs((a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2)
                short_edge = min(min(a.x1 - a.x0, a.y1 - a.y0),
                                 min(b.x1 - b.x0, b.y1 - b.y0))
                center_close = (cx_dist < short_edge / 2 and cy_dist < short_edge / 2)
                same_text = bool(n["text"]) and n["text"] == m["text"]
                if (overlap_ratio >= 0.4 or (center_close and same_text)):
                    # 合并：rect 取并集，文本取非空且更长者
                    m["rect"] = BBox(
                        min(a.x0, b.x0), min(a.y0, b.y0),
                        max(a.x1, b.x1), max(a.y1, b.y1),
                    )
                    if n["text"] and (not m["text"] or len(n["text"]) > len(m["text"])):
                        m["text"] = n["text"]
                    absorbed = True
                    break
            if not absorbed:
                merged.append(dict(n))
        return merged

    def _collect_edges(self, page, area: BBox, nodes: List[Dict],
                       spans: List[Tuple[BBox, str]]) -> List[FlowEdge]:
        """
        收集流程图的边（线段 + 方向 + 条件标签）。

        参数:
            page: fitz.Page
            area: 流程图区域
            nodes: 节点列表
            spans: 文本 span

        返回:
            List[FlowEdge]
        """
        edges: List[FlowEdge] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return edges

        for item in drawings:
            for sub in item.get("items", []):
                if sub[0] != "l" or len(sub) < 3:
                    continue
                p1, p2 = sub[1], sub[2]
                # 线段需在流程图区域内且有足够长度
                if not is_flow_line_segment(p1.x, p1.y, p2.x, p2.y):
                    continue
                if not (area.x0 <= p1.x <= area.x1 and area.x0 <= p2.x <= area.x1
                        and area.y0 <= p1.y <= area.y1 and area.y0 <= p2.y <= area.y1):
                    continue

                # 方向：线段起点端最近的节点 = from，终点端最近的节点 = to
                from_id, from_d = self._nearest_node(p1.x, p1.y, nodes)
                to_id, to_d = self._nearest_node(p2.x, p2.y, nodes)
                if from_id is None or to_id is None or from_id == to_id:
                    continue  # 无法确定端点归属的线忽略

                # 条件标签：线段中点最近的 span 文本
                # （排除节点矩形内部的 span——那是节点正文碎片，不是条件标签）
                mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
                node_rects = [n["rect"] for n in nodes]
                label = self._label_at(mid_x, mid_y, spans, node_rects=node_rects)

                edges.append(FlowEdge(
                    from_node=from_id, to_node=to_id, label=label,
                ))

        # 去重：同一对节点间的重复/平行线段合并为一条边。
        # 源图中一条箭头常被绘制成多条重叠线段（或折线的多段被重复采样），
        # 直接保留会产生完全相同的边；这里按 (from, to) 合并，
        # 优先保留标签更长的边（更长通常意味着更完整的条件文本）。
        dedup: Dict[Tuple[str, str], FlowEdge] = {}
        for e in edges:
            key = (e.from_node, e.to_node)
            cur = dedup.get(key)
            if cur is None or len(e.label or "") > len(cur.label or ""):
                dedup[key] = e
        return list(dedup.values())

    @staticmethod
    def _nearest_node(x: float, y: float, nodes: List[Dict]) -> Tuple[Optional[str], float]:
        """
        找距离点 (x, y) 最近的节点（同时返回距离，便于阈值过滤）。

        参数:
            x, y: 点坐标
            nodes: 节点列表

        返回:
            (节点id, 距离)；无节点返回 (None, 很大)
        """
        best_id, best_dist = None, 1e9
        for n in nodes:
            rect = n["rect"]
            # 点到矩形边界的最短距离
            dx = max(rect.x0 - x, 0, x - rect.x1)
            dy = max(rect.y0 - y, 0, y - rect.y1)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_id, best_dist = n["id"], dist
        # 若距离过大（超过 60pt），视为无归属节点
        if best_dist > 60:
            return None, best_dist
        return best_id, best_dist

    @staticmethod
    def _label_at(x: float, y: float, spans: List[Tuple[BBox, str]],
                  node_rects: Optional[List[BBox]] = None,
                  max_dist: float = 45.0) -> str:
        """
        取离点 (x, y) 最近的 span 文本作为条件标签。

        策略:
            - 排除"中心位于任一节点矩形内部"的 span。节点矩形内部的 span
              是节点正文（如"填写各相关服务记录表"被 PyMuPDF 拆成多个 span
              时，中点附近的碎片"表""面"会被误判为条件标签），必须过滤掉，
              只允许节点外部的文本作为标签。

        参数:
            x, y: 中点坐标
            spans: 文本 span
            node_rects: 节点矩形列表（用于过滤节点内部文本）
            max_dist: 最大匹配距离（超出则返回空串）

        返回:
            str: 标签文本（可能为空）
        """
        best_text, best_dist = "", 1e9
        for sp_bbox, text in spans:
            # span 中心到点的距离
            cx = (sp_bbox.x0 + sp_bbox.x1) / 2
            cy = (sp_bbox.y0 + sp_bbox.y1) / 2
            # 中心落在节点矩形内部 = 节点正文碎片，跳过
            if node_rects:
                pt = BBox(cx, cy, cx, cy)
                if any(r.contains(pt) for r in node_rects):
                    continue
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_text, best_dist = text, dist
        # 单字碎片过滤："重""人""面"这类孤立单字多半是节点文本被拆分后的残片，
        # 不足以构成条件标签；"是""否""有""无"等常见判定词除外。
        if len(best_text) == 1 and best_text not in ("是", "否", "有", "无"):
            return ""
        return best_text if best_dist <= max_dist else ""

    def _find_title(self, page, area: BBox, page_no: int) -> str:
        """
        查找流程图标题：区域上方最近的文本行。

        参数:
            page: fitz.Page
            area: 流程图区域
            page_no: 页码

        返回:
            str: 标题文本
        """
        try:
            text_blocks = page.get_text("dict")["blocks"]
        except Exception:
            return f"流程图(第{page_no}页)"

        above = []
        for block in text_blocks:
            if block.get("type") != 0:
                continue
            r = fitz.Rect(block["bbox"])
            if r.y1 <= area.y0 and r.y0 >= max(0, area.y0 - 100):
                text = "".join(
                    span_text(sp) for line in block.get("lines", []) for sp in line.get("spans", []))
                above.append((r.y1, text.strip()))
        if not above:
            return f"流程图(第{page_no}页)"
        above.sort(key=lambda item: item[0], reverse=True)
        return above[0][1]
