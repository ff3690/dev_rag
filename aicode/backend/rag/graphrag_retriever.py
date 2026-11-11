"""
GraphRAG 双路融合检索器 - GraphRAG Dual-Path Retriever

将向量检索（语义匹配）与知识图谱检索（实体关系）融合，
提供增强的检索能力。支持实体链接、子图提取和双路上下文融合。

融合策略：
1. 向量路：语义检索 Top-K 文本块（已有能力）
2. 图谱路：实体匹配 → BFS 子图提取 → 结构化上下文
3. 融合：两路结果合并，LLM 综合生成回答

架构：组合模式，不修改现有 VectorRetriever 和 GraphService 的任何代码。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from backend.rag.retriever import VectorRetriever, RetrievedDocument
from backend.graph.graph_service import graph_service as _graph_service


@dataclass
class MatchedEntity:
    """查询中匹配到的图谱实体"""
    node_id: int
    name: str
    node_type: str
    description: str
    match_score: float          # 匹配置信度 0-1


@dataclass
class SubGraphNode:
    """子图节点"""
    id: int
    name: str
    node_type: str
    description: str = ""


@dataclass
class SubGraphEdge:
    """子图边"""
    id: int
    source_id: int
    target_id: int
    relation: str
    source_name: str = ""
    target_name: str = ""


@dataclass
class SubGraph:
    """提取的子图"""
    nodes: List[SubGraphNode] = field(default_factory=list)
    edges: List[SubGraphEdge] = field(default_factory=list)
    matched_entities: List[MatchedEntity] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.nodes) == 0


@dataclass
class GraphRAGResult:
    """GraphRAG 融合检索结果"""
    vector_results: List[RetrievedDocument]     # 向量检索结果
    subgraph: SubGraph                           # 图谱检索子图
    graph_context_text: str                      # 图谱上下文文本（可直接注入Prompt）
    query: str = ""


class EntityLinker:
    """
    实体链接器

    将用户查询文本中的实体与知识图谱节点进行匹配。
    采用双重策略：精确名称匹配 + 关键词包含匹配。
    """

    # 实体类型中文关键词映射（辅助从query中识别实体类型意图）
    TYPE_KEYWORDS = {
        "技术": ["技术", "框架", "库", "工具", "语言", "平台", "中间件", "数据库", "协议"],
        "概念": ["概念", "模式", "架构", "设计", "原则", "规范", "制度", "流程"],
        "工具": ["工具", "软件", "系统", "插件", "组件"],
        "项目": ["项目", "模块", "服务", "应用", "产品"],
        "流程": ["流程", "步骤", "阶段", "环节", "周期"],
        "角色": ["角色", "人员", "岗位", "团队", "部门"],
    }

    def link(self, query: str, all_nodes: List[Dict[str, Any]]) -> List[MatchedEntity]:
        """
        将查询链接到知识图谱实体

        参数:
            query: 用户查询文本
            all_nodes: 图谱中所有节点 [{"id": 1, "name": "xxx", "type": "xxx", ...}]

        返回:
            List[MatchedEntity]: 匹配到的实体列表（按匹配度降序）
        """
        matched = []
        query_lower = query.lower()

        for node in all_nodes:
            name = node.get("name", "")
            name_lower = name.lower()

            score = 0.0

            # 策略1：精确名称匹配（最高权重）
            if name_lower in query_lower:
                score = 1.0
            # 策略2：名称片段匹配
            elif len(name) >= 2 and name_lower in query_lower:
                score = 0.8
            # 策略3：查询包含节点名称的关键部分（拆分匹配）
            else:
                name_parts = name_lower.replace("-", " ").replace("_", " ").split()
                matched_parts = sum(1 for p in name_parts if len(p) >= 2 and p in query_lower)
                if matched_parts > 0 and len(name_parts) > 0:
                    score = 0.6 * (matched_parts / len(name_parts))

            if score > 0:
                matched.append(MatchedEntity(
                    node_id=node["id"],
                    name=name,
                    node_type=node.get("type", "未知"),
                    description=node.get("description", ""),
                    match_score=score
                ))

        # 按匹配分数降序排列
        matched.sort(key=lambda x: x.match_score, reverse=True)
        return matched


class GraphRAGRetriever:
    """
    GraphRAG 双路融合检索器

    整合 VectorRetriever（向量路）和 GraphService（图谱路），
    提供增强检索能力。

    使用示例：
        retriever = GraphRAGRetriever(vector_retriever)
        result = retriever.retrieve("FastAPI 依赖哪些库？", db=session)
    """

    def __init__(self, vector_retriever: VectorRetriever):
        """
        初始化 GraphRAG 检索器

        参数:
            vector_retriever: VectorRetriever 实例
        """
        self.vector_retriever = vector_retriever
        self.entity_linker = EntityLinker()

    def retrieve(
        self,
        query: str,
        db: Session,
        top_k: int = 5,
        graph_hops: int = 1,
        include_graph: bool = True,
        use_rerank: bool = False
    ) -> GraphRAGResult:
        """
        双路检索主入口

        参数:
            query: 用户查询文本
            db: 数据库会话（用于访问图谱数据）
            top_k: 向量检索 Top-K
            graph_hops: 图谱 BFS 跳数（1-2）
            include_graph: 是否启用图谱检索
            use_rerank: 向量路是否启用 BGE-Reranker 精排（默认关闭，仅粗排）

        返回:
            GraphRAGResult: 融合检索结果
        """
        import time as _time

        # ── 第1路：向量检索（use_rerank=True 时走 粗排+精排，详细日志见 retriever.py）──
        print(f"  ┌─ [GraphRAG] 向量检索路 开始 (top_k={top_k}, use_rerank={use_rerank})")
        _t0 = _time.perf_counter()
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k, use_rerank=use_rerank)
        _vec_ms = (_time.perf_counter() - _t0) * 1000
        print(f"  ├─ [GraphRAG] 向量检索路 结束 耗时: {_vec_ms:.1f}ms (返回 {len(vector_results)} 条)")

        # ── 第2路：图谱检索 ──
        subgraph = SubGraph()
        if include_graph:
            print(f"  ┌─ [GraphRAG] 图谱检索路 开始 (graph_hops={graph_hops})")
            _t1 = _time.perf_counter()
            try:
                # 获取全量图谱数据
                _t_gd = _time.perf_counter()
                graph_data = _graph_service.get_graph_data(db)
                all_nodes = graph_data.get("nodes", [])
                all_edges = graph_data.get("edges", [])
                print(f"  │  图谱数据加载: {len(all_nodes)} 节点, {len(all_edges)} 边, "
                      f"耗时: {(_time.perf_counter() - _t_gd)*1000:.1f}ms")

                if all_nodes:
                    # 实体链接
                    _t_el = _time.perf_counter()
                    matched = self.entity_linker.link(query, all_nodes)
                    print(f"  │  实体链接: 匹配 {len(matched)} 个实体, "
                          f"耗时: {(_time.perf_counter() - _t_el)*1000:.1f}ms")
                    for me in matched:
                        print(f"  │    - {me.name} (score={me.match_score:.2f})")

                    # 子图提取：从匹配节点出发 BFS
                    _t_bfs = _time.perf_counter()
                    subgraph = self._extract_subgraph(
                        matched, all_nodes, all_edges, hops=graph_hops
                    )
                    print(f"  │  BFS子图提取: {len(subgraph.nodes)} 节点, {len(subgraph.edges)} 边, "
                          f"耗时: {(_time.perf_counter() - _t_bfs)*1000:.1f}ms")
                else:
                    print(f"  │  图谱无数据（nodes=0），跳过实体链接和子图提取")
            except Exception as e:
                # 图谱检索失败不影响向量路结果
                print(f"  ├─ [GraphRAG] 图谱检索路 异常: {e}")

            _graph_ms = (_time.perf_counter() - _t1) * 1000
            print(f"  ├─ [GraphRAG] 图谱检索路 结束 耗时: {_graph_ms:.1f}ms")
        else:
            print(f"  ├─ [GraphRAG] 图谱检索路 跳过（include_graph=False）")

        # 构建图谱上下文文本
        _t_fmt = _time.perf_counter()
        graph_context_text = self._format_graph_context(subgraph)
        print(f"  │  图谱上下文格式化 耗时: {(_time.perf_counter() - _t_fmt)*1000:.1f}ms, "
              f"文本长度: {len(graph_context_text)} 字符")

        return GraphRAGResult(
            vector_results=vector_results,
            subgraph=subgraph,
            graph_context_text=graph_context_text,
            query=query
        )

    def _extract_subgraph(
        self,
        matched_entities: List[MatchedEntity],
        all_nodes: List[Dict],
        all_edges: List[Dict],
        hops: int = 1
    ) -> SubGraph:
        """
        从匹配实体出发 BFS 提取子图

        参数:
            matched_entities: 匹配到的实体
            all_nodes: 所有节点
            all_edges: 所有边
            hops: BFS 跳数

        返回:
            SubGraph: 提取的子图
        """
        if not matched_entities:
            return SubGraph()

        # 构建邻接表
        adjacency = {}
        for edge in all_edges:
            s = edge.get("source_node_id") or edge.get("source")
            t = edge.get("target_node_id") or edge.get("target")
            if s and t:
                adjacency.setdefault(s, []).append(t)
                adjacency.setdefault(t, []).append(s)

        # 构建节点字典（id -> node dict）
        node_dict = {n["id"]: n for n in all_nodes}

        # BFS 提取
        seed_ids = set()
        node_to_entity = {}
        for entity in matched_entities:
            seed_ids.add(entity.node_id)
            node_to_entity[entity.node_id] = entity

        visited = set()
        queue = list(seed_ids)
        for sid in seed_ids:
            visited.add(sid)

        # BFS
        for _ in range(hops + 1):  # +1 包括种子节点自身
            next_queue = []
            for node_id in queue:
                neighbors = adjacency.get(node_id, [])
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in node_dict:
                        visited.add(neighbor)
                        next_queue.append(neighbor)
            queue = next_queue

        # 构建子图节点
        subgraph_nodes = []
        for nid in visited:
            nd = node_dict[nid]
            subgraph_nodes.append(SubGraphNode(
                id=nid,
                name=nd.get("name", ""),
                node_type=nd.get("type", "未知"),
                description=nd.get("description", "")
            ))

        # 构建子图边（仅保留两端都在子图内的边）
        subgraph_edges = []
        visited_set = set(visited)
        for edge in all_edges:
            s = edge.get("source_node_id") or edge.get("source")
            t = edge.get("target_node_id") or edge.get("target")
            if s in visited_set and t in visited_set:
                subgraph_edges.append(SubGraphEdge(
                    id=edge.get("id", 0),
                    source_id=s,
                    target_id=t,
                    relation=edge.get("relation", "关联"),
                    source_name=node_dict.get(s, {}).get("name", ""),
                    target_name=node_dict.get(t, {}).get("name", "")
                ))

        return SubGraph(
            nodes=subgraph_nodes,
            edges=subgraph_edges,
            matched_entities=matched_entities
        )

    def _format_graph_context(self, subgraph: SubGraph) -> str:
        """
        将子图格式化为可注入 LLM Prompt 的结构化文本

        参数:
            subgraph: 提取的子图

        返回:
            str: 格式化的图谱上下文
        """
        if subgraph.is_empty:
            return "（知识图谱中未找到相关实体）"

        lines = ["## 知识图谱结构化上下文", ""]

        # 匹配的实体
        if subgraph.matched_entities:
            lines.append("### 匹配实体")
            for entity in subgraph.matched_entities:
                desc = f" - {entity.description}" if entity.description else ""
                lines.append(f"- **{entity.name}** (类型: {entity.node_type}){desc}")
            lines.append("")

        # 实体关系
        if subgraph.edges:
            lines.append("### 实体关系")
            for edge in subgraph.edges:
                lines.append(f"- {edge.source_name} **{edge.relation}** {edge.target_name}")
            lines.append("")

        # 所有相关实体汇总
        lines.append("### 相关实体列表")
        for node in subgraph.nodes:
            lines.append(f"- [{node.node_type}] {node.name}")

        return "\n".join(lines)


# ==================== 全局单例 ====================

# 注意：graphrag_retriever 依赖 VectorRetriever 实例，在应用启动时初始化
graphrag_retriever: Optional[GraphRAGRetriever] = None


def init_graphrag_retriever(vector_retriever: VectorRetriever):
    """初始化全局 GraphRAG 检索器"""
    global graphrag_retriever
    graphrag_retriever = GraphRAGRetriever(vector_retriever)
