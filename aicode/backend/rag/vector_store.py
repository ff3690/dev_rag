"""
向量存储模块 - Vector Store Module

基于 LlamaIndex 的 SimpleVectorStore 实现向量存储和检索。
支持索引持久化、增量更新和批量操作。

架构设计：
1. LlamaVectorStore: 新版向量存储，使用 LlamaIndex VectorStoreIndex
   - SimpleVectorStore: 内存中的简单向量存储，持久化到磁盘
   - VectorStoreIndex: 上层索引，封装向量化和检索逻辑
   - StorageContext: 存储上下文，管理索引的持久化

2. MilvusVectorStore: 旧版向量存储（保留用于兼容）
   - 基于 SQLite 的简单实现
   - 建议迁移到 LlamaVectorStore

为什么选择 SimpleVectorStore？
- 轻量级：无需外部服务（如 Milvus、Pinecone）
- 零依赖：纯 Python 实现，易于部署
- 持久化：支持保存到磁盘，重启后可恢复
- 适合小规模：文档量 < 10万条时性能充足
- 未来可迁移：LlamaIndex 支持平滑迁移到其他向量数据库
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import os
import json

from backend.config.settings import settings
from backend.rag.embedding_service import embedding_service


@dataclass
class SearchResult:
    """
    向量搜索结果数据类

    封装向量搜索返回的结果信息，用于兼容旧版接口。
    LlamaVectorStore.search() 返回的是 Dict 格式，
    此数据类用于 search_vectors() 方法的返回。

    属性说明：
        chunk_id: 文本块唯一标识，格式为 {document_id}_{index}
        document_id: 原始文档 ID，用于溯源
        content: 文本块内容
        score: 相似度分数 (0-1)，越高越相关
        metadata: 元数据字典，包含分类、部门等信息
    """
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Optional[dict] = None


class LlamaVectorStore:
    """
    基于 LlamaIndex 的向量存储类

    核心组件：
    - _vector_store: SimpleVectorStore 实例，存储向量数据
    - _storage_context: StorageContext，管理存储配置
    - _index: VectorStoreIndex 实例，提供检索能力

    工作流程：
    1. 初始化时从磁盘加载索引，不存在则创建空索引
    2. 添加文档时自动调用 Embedding 服务进行向量化
    3. 检索时使用查询引擎进行相似度搜索
    4. 所有修改操作后自动持久化到磁盘

    使用示例：
        vs = LlamaVectorStore()
        vs.add_documents([{"content": "Hello", "metadata": {"id": "1"}}])
        results = vs.search("Hello", top_k=3)
    """

    def __init__(self, index_path: Optional[str] = None):
        """
        初始化向量存储

        参数:
            index_path: 索引存储路径
                       默认: vector_data/llama_index/
                       可指定自定义路径用于测试

        索引目录结构：
            llama_index/
            ├── default__vector_store.json    # 向量存储数据
            ├── default__index_store.json     # 索引元数据
            └── ...                           # 其他 LlamaIndex 文件
        """
        self.index_path = index_path or os.path.join(settings.VECTOR_STORE_PATH, "llama_index")
        self._index = None
        self._init_index()

    def _init_index(self):
        """
        初始化或加载索引

        策略：
        1. 确保索引目录存在
        2. 检查是否存在持久化数据
        3. 如果存在，从 docstore 加载文档，使用 from_documents 重建索引
        4. 如果不存在，创建空索引

        注意：
        - SimpleVectorStore 只存储向量，不存储文本
        - 因此需要从 docstore 加载文档，再用 from_documents 重建索引
        - 重建索引会重新向量化，但确保数据一致性
        """
        os.makedirs(self.index_path, exist_ok=True)

        try:
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.core.storage.docstore import SimpleDocumentStore
            from llama_index.core.vector_stores import SimpleVectorStore

            # 检查是否存在持久化数据
            docstore_path = os.path.join(self.index_path, "docstore.json")
            vector_store_path = os.path.join(self.index_path, "default__vector_store.json")
            
            has_existing_data = (
                os.path.exists(docstore_path) and 
                os.path.exists(vector_store_path)
            )

            if has_existing_data:
                # 尝试从磁盘加载已有索引
                try:
                    # 1. 加载文档存储（包含文本内容）
                    docstore = SimpleDocumentStore.from_persist_dir(self.index_path)
                    existing_docs = list(docstore.docs.values())
                    
                    if len(existing_docs) > 0:
                        # 2. 使用 from_documents 重建索引
                        #    这会重新向量化并创建新的向量存储
                        print(f"正在加载 {len(existing_docs)} 个文档并重建索引...")
                        
                        self._vector_store = SimpleVectorStore()
                        self._storage_context = StorageContext.from_defaults(
                            vector_store=self._vector_store
                        )
                        
                        self._index = VectorStoreIndex.from_documents(
                            existing_docs,
                            storage_context=self._storage_context,
                            embed_model=embedding_service.embed_model
                        )
                        
                        # 持久化新索引
                        self._persist_index()
                        
                        print(f"✓ 已成功加载索引: {self.index_path}")
                        print(f"  文档数: {len(existing_docs)}")
                    else:
                        print(f"⚠ 文档存储为空，创建空索引")
                        self._create_empty_index()
                        
                except Exception as e:
                    print(f"⚠ 加载已有索引失败: {str(e)}")
                    print(f"  将创建新的空索引")
                    self._create_empty_index()
            else:
                # 没有已有索引，创建新的
                print(f"📁 未发现已有索引，创建新索引")
                self._create_empty_index()
                
        except ImportError:
            print("警告: llama-index 未安装，向量存储功能不可用")
            print("请运行: pip install llama-index-core")
            self._index = None
        except Exception as e:
            print(f"警告: 索引初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self._index = None

    def _create_empty_index(self):
        """
        创建空索引

        当无法从磁盘加载时，创建一个全新的空索引。
        """
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.vector_stores import SimpleVectorStore
        
        # 创建 SimpleVectorStore - 内存中存储向量
        self._vector_store = SimpleVectorStore()

        # 创建存储上下文 - 管理向量存储和索引存储
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        # 创建空索引
        self._index = VectorStoreIndex.from_documents(
            [],
            storage_context=self._storage_context,
            embed_model=embedding_service.embed_model
        )
        print(f"✓ 已创建新的空索引: {self.index_path}")

    def add_documents(self, documents: List[Dict]):
        """
        添加文档到向量存储

        自动调用 Embedding 服务将文本转换为向量，
        然后存储到 SimpleVectorStore 中。

        参数:
            documents: 文档列表，每个文档包含:
                - content: str, 文本内容（必填）
                - metadata: dict, 元数据字典（可选）
                           常用字段: document_id, chunk_id, category, department

        流程：
        1. 将 Dict 转换为 LlamaDocument 对象
        2. 调用 insert_nodes 插入索引
        3. 自动持久化到磁盘

        向量化：
        - 由 VectorStoreIndex 内部自动完成
        - 使用初始化时传入的 embed_model（即 EmbeddingService）
        - 文本会被转换为 512 维向量
        """
        if self._index is None:
            print("警告: 向量存储未初始化")
            return

        try:
            from llama_index.core import Document as LlamaDocument
            from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

            # 转换为 LlamaIndex Document 格式
            # LlamaDocument 是 LlamaIndex 的标准文档格式
            llama_docs = []
            for doc in documents:
                doc_id = doc.get("metadata", {}).get("document_id")
                ld = LlamaDocument(
                    text=doc["content"],           # 文本内容
                    metadata=doc.get("metadata", {})  # 元数据（用于溯源和过滤）
                )
                # 关键修复：设置 SOURCE 关系，使 node.ref_doc_id == document_id
                # 否则 SimpleVectorStore.add 会通过 node_to_metadata_dict 把
                # document_id/doc_id/ref_doc_id 强制覆盖为 "None"
                if doc_id:
                    ld.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                        node_id=doc_id
                    )
                llama_docs.append(ld)

            # 插入到索引 - 自动触发向量化
            self._index.insert_nodes(llama_docs)

            # 持久化到磁盘 - 确保数据不丢失
            self._persist_index()
            print(f"已添加 {len(documents)} 个文档")
        except Exception as e:
            print(f"添加文档失败: {str(e)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        向量检索（主要接口）

        使用 LlamaIndex 的 VectorIndexRetriever 进行纯向量检索，
        不依赖 LLM，仅返回相似文档块。

        参数:
            query: 查询文本（用户输入的问题）
            top_k: 返回最相似的 k 个文档块，默认 5

        返回:
            List[Dict]: 检索结果列表，每个结果包含:
                - content: str, 文本内容
                - score: float, 相似度分数 (0-1)
                - metadata: dict, 元数据
                - id: str, 节点 ID

        检索流程：
        1. 将 query 文本通过 Embedding 模型转换为向量
        2. 在向量存储中计算余弦相似度
        3. 返回 top_k 个最相似的结果

        注意：
        - 分数范围取决于模型，bge 系列通常在 0.3-0.8 之间
        - 如需更高精度，可使用 reranker 进行重排序
        - 使用 VectorIndexRetriever 而非 query_engine，避免 LLM 依赖
        """
        if self._index is None:
            return []

        # 空查询处理 - 直接返回空结果
        if not query or not query.strip():
            return []

        try:
            # 使用 VectorIndexRetriever 进行纯检索
            from llama_index.core.retrievers import VectorIndexRetriever

            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
                embed_model=embedding_service.embed_model
            )

            # 执行检索 - 只返回相关文档，不生成回答
            nodes = retriever.retrieve(query)

            # 提取节点信息
            results = []
            for node in nodes:
                results.append({
                    "content": node.get_content(),
                    "score": getattr(node, 'score', 0.0),
                    "metadata": node.metadata,
                    "id": getattr(node, 'node_id', '')
                })

            return results
        except Exception as e:
            print(f"检索失败: {str(e)}")
            return []

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        向量相似度搜索（底层接口）

        直接使用预计算的向量进行搜索，绕过文本向量化步骤。
        适用于已有向量的场景（如缓存查询）。

        参数:
            query_vector: 查询向量（512 维）
            top_k: 返回数量

        返回:
            List[SearchResult]: 搜索结果列表

        实现说明：
        - 手动计算余弦相似度
        - 遍历所有存储的向量进行比较
        - 时间复杂度 O(n)，适合小规模数据
        - 大规模数据建议使用 FAISS 等专用索引
        """
        if self._index is None:
            return []

        try:
            import numpy as np

            query_embedding = np.array(query_vector)
            results = []

            # 遍历所有存储的节点
            for node in self._index.vector_store.to_dict().get("nodes", []):
                node_embedding = np.array(node.get("embedding", []))
                if len(node_embedding) == 0:
                    continue

                # 计算余弦相似度: cos(θ) = (A·B) / (|A| * |B|)
                norm = np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
                if norm == 0:
                    score = 0.0
                else:
                    score = float(np.dot(query_embedding, node_embedding) / norm)

                results.append(SearchResult(
                    chunk_id=node.get("node_id", ""),
                    document_id=node.get("metadata", {}).get("document_id", ""),
                    content=node.get("text", ""),
                    score=score,
                    metadata=node.get("metadata", {})
                ))

            # 按相似度降序排列
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"向量搜索失败: {str(e)}")
            return []

    def delete_document(self, document_id: str):
        """
        删除指定文档的所有向量

        根据 document_id 元数据过滤并删除对应的向量数据，
        同时清理 docstore 中的文档记录，确保重启后不会“复活”。

        参数:
            document_id: 文档 ID（在 metadata 中存储的标识）

        实现说明：
        - 优先使用官方 delete_ref_doc：同时清理 docstore / vector_store / index_store
        - 兜底方式：直接操作 vector_store 的 embedding/metadata 结构删除残留节点
        - 删除后必须重新持久化以确保下次启动时数据一致

        注意：
        - 操作不可逆，删除后无法恢复
        - 建议在删除前备份索引
        """
        print(f"[delete_vector] ===== 开始删除文档向量，document_id={document_id} =====")
        if self._index is None:
            print(f"[delete_vector] ✗ 向量存储未初始化(self._index is None)，无法删除，请检查 LlamaIndex 是否正常加载")
            return

        try:
            vector_store = self._index.vector_store
            deleted_count = 0

            # 0. 打印当前索引状态，便于分析
            count_before = None
            try:
                count_before = self.get_vector_count()
                print(f"[delete_vector] 删除前向量总数: {count_before}")
                print(f"[delete_vector] 目标 document_id: {document_id}，类型: {type(document_id).__name__}")
            except Exception as e:
                print(f"[delete_vector] 获取删除前向量总数失败: {e}")

            # 1. 收集属于该 document_id 的所有节点 ID
            #    （从 vector_store 的 metadata_dict 以及 docstore 的 ref_doc_info 中获取）
            target_node_ids = set()
            vs_dict = None
            try:
                vs_dict = vector_store.to_dict()
                print(f"[delete_vector] vector_store.to_dict() 返回的 key: {list((vs_dict or {}).keys())}")
                metadata_dict = (vs_dict or {}).get("metadata_dict", {}) or {}
                print(f"[delete_vector] metadata_dict 节点数: {len(metadata_dict)}")
                # 打印一个样本节点，确认 metadata 结构
                sample_keys = list(metadata_dict.keys())[:2]
                for sk in sample_keys:
                    print(f"[delete_vector] 样本节点 {sk} 的 metadata: {metadata_dict[sk]}")
                for node_id, node_meta in metadata_dict.items():
                    if (node_meta or {}).get("document_id") == document_id:
                        target_node_ids.add(node_id)
                print(f"[delete_vector] 从 metadata_dict 匹配到节点数: {len(target_node_ids)}")
            except Exception as e:
                print(f"[delete_vector] ✗ 读取 vector_store 元数据失败: {e}")

            # 2. 优先使用官方 delete_ref_doc（同时清理 docstore / vector_store / index_store）
            matched_ref_ids = []
            try:
                docstore = self._index.docstore
                ref_doc_infos = getattr(docstore, "ref_doc_info", None) or {}
                print(f"[delete_vector] docstore ref_doc_info 总数: {len(ref_doc_infos)}")
                for ref_id, info in ref_doc_infos.items():
                    md = getattr(info, "metadata", None) or {}
                    if md.get("document_id") == document_id:
                        matched_ref_ids.append(ref_id)
                        node_ids = getattr(info, "node_ids", None) or []
                        target_node_ids.update(node_ids)
                print(f"[delete_vector] 从 docstore 匹配到 ref_doc_id: {matched_ref_ids}")
            except Exception as e:
                print(f"[delete_vector] ✗ 读取 docstore ref_doc_info 失败: {e}")

            print(f"[delete_vector] 最终待删除节点(node_id)集合: {target_node_ids}")

            delete_ref_doc = getattr(self._index, "delete_ref_doc", None)
            if not delete_ref_doc:
                print(f"[delete_vector] ⚠ 当前 LlamaIndex 版本不支持 delete_ref_doc，改用直接操作 vector_store")
            for ref_id in matched_ref_ids:
                if delete_ref_doc:
                    try:
                        # 关键：delete_from_docstore=True 才会同时清理 docstore（docstore.json）
                        # 若为 False（默认），delete_ref_doc 只会删 vector_store 中的向量，
                        # docstore 中的 ref_doc 与 node 不会被删除，导致删除后 docstore.json 残留
                        delete_ref_doc(ref_id, delete_from_docstore=True)
                        deleted_count += 1
                        print(f"[delete_vector] ✓ delete_ref_doc({ref_id}, delete_from_docstore=True) 调用成功")
                    except Exception as e:
                        print(f"[delete_vector] ✗ delete_ref_doc({ref_id}) 失败: {e}")

            # 3. 兜底：若仍有节点未删除，直接操作 vector_store 内部结构
            if vs_dict is not None and target_node_ids:
                try:
                    embedding_dict = dict((vs_dict or {}).get("embedding_dict", {}) or {})
                    metadata_dict = dict((vs_dict or {}).get("metadata_dict", {}) or {})
                    text_id_map = dict((vs_dict or {}).get("text_id_to_ref_doc_id", {}) or {})
                    print(f"[delete_vector] 兜底删除前: embedding_dict={len(embedding_dict)}, metadata_dict={len(metadata_dict)}, text_id_map={len(text_id_map)}")

                    removed = 0
                    not_in_embedding = []
                    for nid in list(target_node_ids):
                        if nid in embedding_dict:
                            del embedding_dict[nid]
                            removed += 1
                        else:
                            not_in_embedding.append(nid)
                        metadata_dict.pop(nid, None)
                        text_id_map.pop(nid, None)

                    print(f"[delete_vector] 兜底删除: 从 embedding_dict 删除 {removed} 个，未在 embedding_dict 中的节点: {not_in_embedding}")

                    if removed > 0:
                        # 写回 SimpleVectorStore 内部数据（兼容不同版本的属性命名）
                        data = getattr(vector_store, "_data", None)
                        if data is not None:
                            data.embedding_dict = embedding_dict
                            data.metadata_dict = metadata_dict
                            data.text_id_to_ref_doc_id = text_id_map
                            print(f"[delete_vector] ✓ 已写回 vector_store._data")
                        else:
                            print(f"[delete_vector] ⚠ vector_store 没有 _data 属性，需通过公开属性同步")
                        # 同步常见公开属性，确保 to_dict()/persist() 读取到最新数据
                        for attr, val in (
                            ("embedding_dict", embedding_dict),
                            ("metadata_dict", metadata_dict),
                            ("text_id_to_ref_doc_id", text_id_map),
                        ):
                            if hasattr(vector_store, attr):
                                setattr(vector_store, attr, val)
                                print(f"[delete_vector] ✓ 已同步公开属性 {attr}")

                        if deleted_count == 0:
                            deleted_count = removed
                except Exception as e:
                    print(f"[delete_vector] ✗ 直接操作 vector_store 删除失败: {e}")

            # 4. 彻底清理 docstore 中该文档的节点与 ref_doc，避免重启后从 docstore 重建时“复活”
            #    注意：delete_ref_doc 只有在 delete_from_docstore=True 时才会清理 docstore，
            #    且需按 ref_doc_id 匹配；这里作为兜底，直接按 document_id 找到所有 ref_doc，
            #    并逐个删除其 node 与 ref_doc 记录，保证 docstore.json 被真正清干净。
            try:
                docstore = self._index.docstore
                delete_doc = getattr(docstore, "delete_document", None)
                delete_ref_doc_store = getattr(docstore, "delete_ref_doc", None)
                ref_doc_infos = getattr(docstore, "ref_doc_info", None) or {}

                # 4.1 找出所有属于该 document_id 的 ref_doc，收集其 node_ids
                target_ref_ids = [
                    rid for rid, info in ref_doc_infos.items()
                    if (getattr(info, "metadata", None) or {}).get("document_id") == document_id
                ]
                doc_node_ids = set()
                for rid in target_ref_ids:
                    info = ref_doc_infos.get(rid)
                    node_ids = getattr(info, "node_ids", None) or []
                    doc_node_ids.update(node_ids)
                # 若已通过 vector_store 匹配到节点，也纳入清理范围
                doc_node_ids.update(target_node_ids)

                print(f"[delete_vector] 待清理 docstore: ref_doc_ids={target_ref_ids}, node_ids 共 {len(doc_node_ids)} 个")

                # 4.2 删除每个 node（含 kvstore 中的 node 与 metadata 记录）
                if delete_doc:
                    for nid in doc_node_ids:
                        try:
                            delete_doc(nid, raise_error=False)
                        except Exception as e:
                            print(f"[delete_vector] ✗ docstore.delete_document({nid}) 失败: {e}")
                    print(f"[delete_vector] ✓ 已从 docstore 删除 {len(doc_node_ids)} 个 node")

                # 4.3 删除 ref_doc 记录（优先用官方 delete_ref_doc，否则直接弹出 ref_doc_info）
                for rid in target_ref_ids:
                    if delete_ref_doc_store:
                        try:
                            delete_ref_doc_store(rid, raise_error=False)
                            print(f"[delete_vector] ✓ docstore.delete_ref_doc({rid}) 成功")
                            continue
                        except Exception as e:
                            print(f"[delete_vector] ✗ docstore.delete_ref_doc({rid}) 失败: {e}")
                    # 兜底：直接从 ref_doc_info 中移除该 ref_doc
                    try:
                        ref_doc_infos.pop(rid, None)
                        print(f"[delete_vector] ✓ 已从 ref_doc_info 移除 {rid}")
                    except Exception as e:
                        print(f"[delete_vector] ✗ 从 ref_doc_info 移除 {rid} 失败: {e}")

                if target_ref_ids or doc_node_ids:
                    deleted_count = max(deleted_count, 1)

                # 4.4 最终校验：确保 docstore 中已无该 document_id 的残留
                ref_doc_infos = getattr(docstore, "ref_doc_info", None) or {}
                still_exist = [
                    rid for rid, info in ref_doc_infos.items()
                    if (getattr(info, "metadata", None) or {}).get("document_id") == document_id
                ]
                print(f"[delete_vector] 清理后 docstore 中该 document 的残留 ref_doc: {still_exist}")
            except Exception as e:
                print(f"[delete_vector] ✗ 清理 docstore 失败: {e}")

            # 5. 持久化到磁盘
            try:
                self._persist_index()
                print(f"[delete_vector] ✓ 索引已持久化到磁盘")
            except Exception as e:
                print(f"[delete_vector] ✗ 索引持久化失败: {e}")

            # 6. 删除后验证
            try:
                count_after = self.get_vector_count()
                print(f"[delete_vector] 删除后向量总数: {count_after}（删除前 {count_before if count_before is not None else '未知'}）")
            except Exception as e:
                print(f"[delete_vector] 获取删除后向量总数失败: {e}")

            if deleted_count > 0:
                print(f"[delete_vector] ✔ 删除成功：文档 {document_id} 的 {deleted_count} 条向量/引用已删除")
            else:
                print(f"[delete_vector] ✘ 未找到文档 {document_id} 的向量数据（deleted_count=0，请检查 document_id 与 metadata.document_id 是否一致）")

        except Exception as e:
            print(f"[delete_vector] ✗ 删除文档失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _persist_index(self):
        """
        持久化索引到磁盘

        将内存中的索引状态保存到文件系统，
        确保应用重启后可以恢复数据。

        持久化内容：
        - 向量数据 (default__vector_store.json)
        - 索引元数据 (default__index_store.json)

        调用时机：
        - add_documents 之后
        - delete_document 之后
        - 任何修改索引的操作之后
        """
        if self._index is not None:
            try:
                self._index.storage_context.persist(self.index_path)
            except Exception as e:
                print(f"索引持久化失败: {str(e)}")

            # 确保 docstore 被显式持久化到 docstore.json
            # （storage_context.persist 通常已包含 docstore，但显式写一次可兜底，
            #   尤其当 index.docstore 与 storage_context.docstore 不是同一实例时）
            try:
                docstore = getattr(self._index, "docstore", None)
                if docstore is not None and hasattr(docstore, "persist"):
                    os.makedirs(self.index_path, exist_ok=True)
                    docstore_path = os.path.join(self.index_path, "docstore.json")
                    docstore.persist(persist_path=docstore_path)
            except Exception as e:
                print(f"docstore 显式持久化失败: {str(e)}")

    def get_vector_count(self) -> int:
        """
        获取向量总数

        用于监控和验证索引状态。

        返回:
            int: 存储的向量数量（即文档块数量）
        """
        if self._index is None:
            return 0

        try:
            vector_store_dict = self._index.vector_store.to_dict()
            # 优先读取 embedding_dict（SimpleVectorStore 实际存储向量的字段）
            embedding_dict = (vector_store_dict or {}).get("embedding_dict", {})
            if isinstance(embedding_dict, (dict, list)):
                return len(embedding_dict)
            # 兼容旧的 nodes 结构
            nodes = vector_store_dict.get("nodes", {})
            if isinstance(nodes, dict):
                return len(nodes)
            elif isinstance(nodes, list):
                return len(nodes)
            return 0
        except Exception:
            return 0

    def is_available(self) -> bool:
        """
        检查向量存储是否可用

        返回:
            bool: 索引是否已初始化且可用
        """
        return self._index is not None


class MilvusVectorStore:
    """
    旧版向量存储类（保留用于兼容和数据迁移）

    基于 SQLite 的简单向量存储实现。
    已被 LlamaVectorStore 取代，保留原因：
    1. 兼容旧代码中的引用
    2. 数据迁移时需要读取旧格式数据
    3. 作为 SimpleVectorStore 的备选方案

    建议：
    - 新功能使用 LlamaVectorStore
    - 旧数据可通过 scripts/migrate_vectors.py 迁移
    - 未来版本可能移除此类
    """

    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self.vector_store_path = os.path.join("vector_data", f"{collection_name}.db")
        self._init_database()

    def _init_database(self):
        import sqlite3
        os.makedirs("vector_data", exist_ok=True)
        
        with sqlite3.connect(self.vector_store_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id ON vectors(document_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_id ON vectors(chunk_id)
            """)
            conn.commit()

    def insert_vectors(self, vectors: List[List[float]], metadata: List[dict]):
        import sqlite3
        import struct
        
        def _vector_to_blob(vector):
            return struct.pack(f'{len(vector)}f', *vector)
        
        with sqlite3.connect(self.vector_store_path) as conn:
            cursor = conn.cursor()
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                cursor.execute("""
                    INSERT OR REPLACE INTO vectors (chunk_id, document_id, vector, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    meta.get("chunk_id", f"chunk_{i}"),
                    meta.get("document_id", ""),
                    _vector_to_blob(vector),
                    meta.get("content", ""),
                    json.dumps(meta)
                ))
            conn.commit()

    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        import sqlite3
        import struct
        
        def _blob_to_vector(blob):
            count = len(blob) // 4
            return list(struct.unpack(f'{count}f', blob))
        
        def _dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))
        
        def _norm(vector):
            return sum(x * x for x in vector) ** 0.5
        
        results = []
        with sqlite3.connect(self.vector_store_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, chunk_id, document_id, vector, content, metadata FROM vectors")
            
            for row in cursor.fetchall():
                stored_vector = _blob_to_vector(row[3])
                norm = _norm(query_vector) * _norm(stored_vector)
                score = _dot_product(query_vector, stored_vector) / norm if norm != 0 else 0.0
                
                results.append(SearchResult(
                    chunk_id=row[1],
                    document_id=row[2],
                    content=row[4],
                    score=score,
                    metadata=json.loads(row[5]) if row[5] else None
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete_by_document_id(self, document_id: str):
        import sqlite3
        with sqlite3.connect(self.vector_store_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE document_id = ?", (document_id,))
            conn.commit()

    def get_vector_count(self) -> int:
        import sqlite3
        with sqlite3.connect(self.vector_store_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]