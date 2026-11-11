"""
检索器模块 - Retriever Module

基于 LlamaIndex 的向量检索器，使用真实 Embedding 模型进行查询向量化。
支持上下文感知检索、结果重排序和来源追踪。

模块职责：
1. VectorRetriever: 向量检索器，执行核心检索逻辑
2. Reranker: 结果重排序器，优化检索结果质量（旧版简单排序，已由 RerankerService 取代）
3. SourceTracker: 来源追踪器，格式化来源信息

检索流程（集成精排）：
用户查询 -> Embedding 向量化 -> 向量粗排(Top-N) -> BGE-Reranker精排(Top-K) -> 来源追踪 -> 返回结果

精排集成说明：
- VectorRetriever 可选注入 RerankerService 实例
- 启用精排时：粗排返回 N 个候选，精排后返回 K 个最终结果
- 关闭精排时：直接返回粗排 Top-K 结果（与原逻辑一致）
"""

from typing import List, Optional
from dataclasses import dataclass

from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import LlamaVectorStore, SearchResult
from backend.rag.text_splitter import TextSplitter
from backend.config.settings import settings


@dataclass
class RetrievedDocument:
    """
    检索结果文档数据类

    封装检索返回的文档信息，是 RAG 流程中检索阶段的标准输出。
    传递给 LLM 作为上下文，用于生成最终回答。

    属性说明：
        document_id: 原始文档 ID（对应数据库中的文档记录）
        chunk_id: 文本块 ID（格式: {document_id}_{index}）
        content: 文本块内容（传递给 LLM 的实际内容）
        score: 主分数 (0-1)，分数越高表示越相关。
               经过精排时为精排分数（Cross-Encoder sigmoid 输出），
               未精排/精排降级时为粗排向量相似度分数
        metadata: 元数据字典，包含分类、部门等溯源信息
        rerank_score: 精排分数。仅真正执行精排时非 None，
                      用于区分精排/粗排两种不同量纲的分数
        original_score: 粗排原始分数（向量相似度），精排后可对比展示

    使用场景：
    - 传递给 LLM 作为 prompt 的上下文
    - 展示给用户作为来源引用
    - 用于日志记录和调试
    """
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Optional[dict] = None
    rerank_score: Optional[float] = None
    original_score: Optional[float] = None


class VectorRetriever:
    """
    基于 LlamaIndex 的向量检索器

    负责将用户查询转换为向量，并从向量存储中检索最相关的文档。
    是 RAG 系统中"R"（Retrieval）的核心实现。

    检索策略：
    - 语义检索：使用 Embedding 模型计算语义相似度
    - Top-K 返回：返回相似度最高的 K 个文档块
    - 无上下文感知：默认独立处理每个查询

    与旧版的区别：
    - 旧版使用零向量 + SQL 模糊匹配
    - 新版使用真实 Embedding + 向量余弦相似度
    - 检索质量大幅提升

    使用示例：
        retriever = VectorRetriever(vector_store)
        results = retriever.retrieve("如何配置系统？", top_k=3)
        for doc in results:
            print(f"[{doc.score:.2f}] {doc.content[:50]}")
    """

    def __init__(
        self,
        vector_store: LlamaVectorStore,
        splitter: Optional[TextSplitter] = None,
        reranker_service=None
    ):
        """
        初始化向量检索器

        参数:
            vector_store: LlamaIndex 向量存储实例
                         必须已初始化且包含索引数据
            splitter: 文本分割器实例（预留用于上下文扩展）
                      默认使用 TextSplitter 标准实现
            reranker_service: RerankerService 实例（可选）
                             注入后启用精排功能，流程变为：粗排 → 精排 → 返回
                             为 None 时仅使用粗排（与原逻辑一致）

        精排集成说明：
        - 启用精排时，retrieve() 方法的 top_k 参数含义变为"精排后返回数量"
        - 粗排会返回 top_k * 4 个候选（受 RERANKER_INPUT_TOP_K 限制）
        - 精排后返回 top_k 个最终结果
        """
        self.vector_store = vector_store
        self.splitter = splitter or TextSplitter()
        self.reranker = reranker_service

    def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> List[RetrievedDocument]:
        """
        检索相似文档（主要接口）

        完整的检索流程（集成精排）：
        1. 将查询文本通过 Embedding 模型转换为向量
        2. 在向量存储中计算与所有文档块的余弦相似度（粗排）
        3. 如果启用精排：对粗排结果进行 Cross-Encoder 精排重排序
        4. 返回最相似的 top_k 个文档块

        参数:
            query: 用户查询文本（如 "如何配置数据库连接？"）
            top_k: 返回的最大结果数量，默认 5
                   - 启用精排时：精排后返回的数量
                   - 关闭精排时：粗排直接返回的数量
                   - 推荐范围: 3-10
            use_rerank: 是否启用精排（默认 True）
                       - True: 粗排 + 精排
                       - False: 仅粗排（与原逻辑一致）

        返回:
            List[RetrievedDocument]: 检索结果列表，按相关性降序排列
                                      如果没有匹配结果，返回空列表

        精排流程说明：
        1. 粗排阶段：向量检索返回 max(top_k, RERANKER_INPUT_TOP_K) 个候选
        2. 精排阶段：BGE-Reranker 对候选进行 Cross-Encoder 打分
        3. 返回阶段：按精排分数降序返回 top_k 个结果

        性能（CPU 环境）：
        - 粗排（向量化+检索）: ~20-50ms
        - 精排（Cross-Encoder）: ~500-1500ms（5个文档）
        - 总计: ~1-2秒

        降级行为：
        - 向量存储不可用: 返回空列表 []
        - Embedding 模型不可用: 返回空列表 []
        - 查询为空字符串: 返回空列表 []
        - Reranker 不可用: 返回粗排结果（降级到原逻辑）
        """
        import time as _time

        # 精排开启时，扩大粗排召回数量
        if use_rerank and self.reranker is not None and self.reranker.is_available():
            # 粗排召回数量 = max(请求的 top_k, 配置的 input_top_k)
            # 确保粗排返回足够多的候选供精排筛选
            recall_k = max(top_k, self.reranker.input_top_k)
            print(f"  ┌─ [粗排] 开始 (recall_k={recall_k}, use_rerank=True)")
        else:
            # 不启用精排，直接返回 top_k 个结果
            recall_k = top_k
            if not use_rerank:
                print(f"  ┌─ [粗排] 开始 (recall_k={recall_k}, 精排未启用 use_rerank=False)")
            else:
                print(f"  ┌─ [粗排] 开始 (recall_k={recall_k}, 精排服务不可用，降级为仅粗排)")

        # Phase 1: 向量粗排
        _t0 = _time.perf_counter()
        results = self.vector_store.search(query, top_k=recall_k)  #粗排，返回 dict 列表
        _coarse_ms = (_time.perf_counter() - _t0) * 1000
        print(f"  ├─ [粗排] 结束 耗时: {_coarse_ms:.1f}ms (返回 {len(results)} 条)")

        # 将 Dict 结果转换为 RetrievedDocument 对象
        retrieved_docs = [
            RetrievedDocument(
                document_id=result.get("metadata", {}).get("document_id", ""),
                chunk_id=result.get("id", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata")
            )
            for result in results
        ]

        # Phase 2: 精排重排序（如果启用且可用）
        if use_rerank and self.reranker is not None and self.reranker.is_available():
            try:
                print(f"  ┌─ [精排] 开始 (候选 {len(retrieved_docs)} 条, top_k={top_k})")
                _t1 = _time.perf_counter()
                retrieved_docs = self.reranker.rerank_retrieved_documents(
                    query=query,
                    documents=retrieved_docs, #把粗排结果整体传给精排
                    top_k=top_k
                )
                _rerank_ms = (_time.perf_counter() - _t1) * 1000
                print(f"  ├─ [精排] 结束 耗时: {_rerank_ms:.1f}ms (返回 {len(retrieved_docs)} 条)")
            except Exception as e:
                # 精排异常，降级到粗排结果
                _rerank_ms = (_time.perf_counter() - _t1) * 1000 if '_t1' in dir() else 0
                print(f"  ├─ [精排] 异常 耗时: {_rerank_ms:.1f}ms ({e})，降级到粗排结果")
                retrieved_docs = retrieved_docs[:top_k]
        else:
            # 未启用精排，直接返回 top_k 个
            if not use_rerank:
                print(f"  ├─ [精排] 跳过（use_rerank=False）")
            else:
                print(f"  ├─ [精排] 跳过（reranker 服务不可用）")
            retrieved_docs = retrieved_docs[:top_k]

        # 打印分数概览（便于排查精排/粗排分数量纲问题）
        # 注意：doc.score 始终是"主分数"（精排时=精排分，降级时=粗排分）
        #       doc.original_score 始终是原始粗排分数
        #       doc.rerank_score 仅真正精排时非 None
        for i, doc in enumerate(retrieved_docs):
            rr = getattr(doc, "rerank_score", None)
            orig = getattr(doc, "original_score", None)
            if rr is not None:
                # 真正执行了精排
                if orig is not None:
                    # 精排分相对于粗排分的升降：
                    #  UP = 精排分 > 粗排分（精排认为该文档更相关，排名上升，这说明 BGE-Reranker 模型确实加载成功并完成了推理，不是降级路径。）
                    #  DOWN = 精排分 < 粗排分（精排认为该文档不太相关，排名下降）
                    #  SAME = 两者相等
                    # 精排把粗排中排名的文档判定为几乎不相关，这说明粗排召回了噪声文档，精排有效过滤了它们。
                    # 但是 精排也不是完美的 — BGE-Reranker 给出的评分判定为"不太相关"，但不代表"完全无用"。被过滤的文档可能包含背景知识、定义说明等辅助信息。
                    # 所以还需要引入“补充低分文档”的逻辑
                    change_str = "UP" if rr > orig else ("DOWN" if rr < orig else "SAME")
                    print(f"  │  结果 {i}: 精排分 {rr:.3f} | 粗排分 {orig:.3f} ({change_str})")
                else:
                    print(f"  │  结果 {i}: 精排分 {rr:.3f} | 粗排分 (未知)")
            else:
                # 降级/粗排模式
                print(f"  │  结果 {i}: 粗排分 {doc.score:.3f} (未精排)")

        return retrieved_docs

    def retrieve_with_context(self, query: str, context: List[str], top_k: int = 5, use_rerank: bool = True) -> List[RetrievedDocument]:
        """
        结合上下文检索相似文档（多轮对话场景）

        将历史对话和当前查询合并后进行检索，
        使检索结果能够考虑对话上下文。

        参数:
            query: 当前用户查询
            context: 历史对话上下文列表（按时间顺序）
                    每个元素是一段对话文本
            top_k: 返回的最大结果数量
            use_rerank: 是否启用精排（默认 True）

        返回:
            List[RetrievedDocument]: 检索结果列表

        使用场景：
        - 多轮对话系统：用户可能使用代词指代之前提到的内容
        - 需要上下文才能理解的查询

        限制：
        - 简单拼接策略，可能引入噪声
        - 上下文过长会稀释当前查询的权重
        - 建议上下文长度不超过 2000 字符
        """
        # 将上下文和查询拼接成一个完整查询
        combined_query = " ".join(context + [query])
        return self.retrieve(combined_query, top_k=top_k, use_rerank=use_rerank)

    def retrieve_by_vector(self, query_vector: List[float], top_k: int = 5) -> List[RetrievedDocument]:
        """
        使用预计算的向量进行检索（高级接口）

        绕过文本向量化步骤，直接使用预计算的向量。
        适用于缓存查询或需要自定义向量化策略的场景。

        参数:
            query_vector: 预计算的查询向量（512 维）
                         必须与索引使用的 Embedding 模型一致
            top_k: 返回的最大结果数量

        返回:
            List[RetrievedDocument]: 检索结果列表

        使用场景：
        - 查询缓存：同一查询的向量可复用
        - 自定义 Embedding：使用不同的模型生成向量
        - 批量处理：预先计算多个查询的向量
        """
        results = self.vector_store.search_vectors(query_vector, top_k=top_k)

        # 将 SearchResult 转换为 RetrievedDocument
        return [
            RetrievedDocument(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata
            )
            for result in results
        ]


class Reranker:
    """
    结果重排序器类

    对初步检索结果进行重排序，进一步优化结果质量。
    可以集成更复杂的重排序模型（如 cross-encoder）。

    当前实现：
    - 基于相似度分数降序排序
    - 简单但有效，适合作为基础实现

    未来扩展：
    - 可引入 BGE-Reranker 等专用重排序模型
    - 可结合查询意图进行加权排序
    - 可实现多样性约束，避免返回相似文档

    使用示例：
        reranker = Reranker()
        results = retriever.retrieve("配置说明", top_k=10)
        reranked = reranker.rerank("配置说明", results)
        top_5 = reranked[:5]
    """

    def rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        重排序检索结果

        当前基于分数降序排序，未来可实现更复杂的策略。

        参数:
            query: 用户查询文本（预留用于上下文感知重排序）
            documents: 待排序的文档列表

        返回:
            List[RetrievedDocument]: 重排序后的文档列表
        """
        return sorted(documents, key=lambda x: x.score, reverse=True)


class SourceTracker:
    """
    来源追踪器类

    将检索结果格式化为便于展示的来源信息。
    用于前端展示和溯源功能。

    功能：
    1. 提取关键信息（ID、分数、内容预览）
    2. 截断过长内容为预览
    3. 合并元数据信息

    使用场景：
    - API 返回给前端展示
    - 用户查看答案来源
    - 审计和追溯文档引用
    """

    def format_sources(self, documents: List[RetrievedDocument]) -> List[dict]:
        """
        格式化检索结果为来源信息列表

        为每个检索结果创建一个来源字典，
        包含可展示的预览内容。

        参数:
            documents: 检索结果文档列表

        返回:
            List[dict]: 格式化后的来源信息列表
                       每个字典包含:
                       - document_id: 文档 ID
                       - chunk_id: 文本块 ID
                       - score: 相似度分数
                       - content_preview: 内容预览（前 200 字符）
                       - 其他元数据字段（如有）
        """
        sources = []
        for doc in documents:
            # 截断内容为预览，避免返回过长文本
            content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content

            source_info = {
                "document_id": doc.document_id,
                "chunk_id": doc.chunk_id,
                "score": doc.score,
                "content_preview": content_preview
            }

            # 附带精排分数/粗排原始分数（若有），便于前端区分两种量纲
            if getattr(doc, "rerank_score", None) is not None:
                source_info["rerank_score"] = doc.rerank_score
            if getattr(doc, "original_score", None) is not None:
                source_info["original_score"] = doc.original_score

            # 合并元数据（如果有）
            if doc.metadata:
                source_info.update(doc.metadata)

            # Markdown 图片数据契约：图片相关字段始终存在，
            # 保证前端接入时字段稳定（无图块返回 False + 空列表）
            source_info.setdefault("has_image", False)
            source_info.setdefault("image_urls", [])
            source_info.setdefault("enhanced_by", "")

            sources.append(source_info)

        return sources