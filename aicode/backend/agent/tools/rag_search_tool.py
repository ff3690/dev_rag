"""
RAG 知识库检索工具 - RAG Search Tool

Agent 可通过此工具在研发知识库中搜索相关文档和知识图谱信息。
这是 Agent 最核心的工具，几乎所有知识问答场景都需要它。
"""

from typing import Optional
from sqlalchemy.orm import Session

from backend.rag.retriever import VectorRetriever
from backend.rag.graphrag_retriever import GraphRAGRetriever, GraphRAGResult


# 工具描述（供 Agent 的 LLM 决策时参考）
TOOL_DESCRIPTION = """
工具名称: rag_search
功能: 在研发团队知识库中搜索相关文档和知识图谱信息
参数:
  - query (str, 必填): 搜索查询关键词，请用自然语言描述你想查找的内容
  - top_k (int, 可选): 返回结果数量，默认5
返回: 检索到的文档内容和知识图谱关联信息
使用场景:
  - 查找技术文档、项目规范、制度文件
  - 查询技术栈、依赖关系
  - 获取项目相关信息
"""


def rag_search_tool(
    query: str,
    top_k: int = 5,
    db: Optional[Session] = None,
    vector_retriever: Optional[VectorRetriever] = None,
    graphrag_retriever: Optional[GraphRAGRetriever] = None
) -> str:
    """
    在研发知识库中检索相关文档

    优先使用 GraphRAGRetriever（如果可用），融合向量检索和知识图谱检索；
    否则使用纯向量检索。

    参数:
        query: 搜索查询
        top_k: 返回文档数量
        db: 数据库会话
        vector_retriever: 向量检索器实例
        graphrag_retriever: GraphRAG 检索器实例

    返回:
        str: 格式化的检索结果文本
    """
    if not vector_retriever:
        return "[工具错误] 检索器未初始化"

    # 尝试使用 GraphRAG 双路检索
    if graphrag_retriever and db:
        try:
            result: GraphRAGResult = graphrag_retriever.retrieve(
                query=query, db=db, top_k=top_k, include_graph=True
            )

            parts = []

            # 向量检索结果
            if result.vector_results:
                parts.append("## 知识库文档检索结果\n")
                for i, doc in enumerate(result.vector_results, 1):
                    doc_id = (doc.metadata or {}).get("document_id", "未知")
                    parts.append(f"### 文档{i} (ID: {doc_id}, 相关度: {doc.score:.2f})")
                    # 截取内容前800字符
                    content = doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
                    parts.append(content)
                    parts.append("")

            # 知识图谱上下文
            if result.graph_context_text and result.graph_context_text != "（知识图谱中未找到相关实体）":
                parts.append(result.graph_context_text)

            if not parts:
                return "未找到与查询相关的知识库内容。"

            return "\n".join(parts)

        except Exception as e:
            print(f"[rag_search_tool] GraphRAG 检索失败，降级为纯向量检索: {e}")

    # 降级：纯向量检索
    results = vector_retriever.retrieve(query, top_k=top_k)

    if not results:
        return "未找到与查询相关的知识库内容。"

    parts = ["## 知识库文档检索结果\n"]
    for i, doc in enumerate(results, 1):
        doc_id = (doc.metadata or {}).get("document_id", "未知")
        parts.append(f"### 文档{i} (ID: {doc_id}, 相关度: {doc.score:.2f})")
        content = doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
        parts.append(content)
        parts.append("")

    return "\n".join(parts)
