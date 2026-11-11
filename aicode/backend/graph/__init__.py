"""
知识图谱模块 - Knowledge Graph Module

该模块提供知识图谱的构建、存储和查询功能。
基于 DeepSeek LLM 自动抽取文档中的实体和关系，使用 SQLite 存储图数据。
"""

from backend.graph.models import KnowledgeNode, KnowledgeEdge
from backend.graph.graph_service import GraphService
from backend.graph.routes import router as graph_router

__all__ = ["KnowledgeNode", "KnowledgeEdge", "GraphService", "graph_router"]
