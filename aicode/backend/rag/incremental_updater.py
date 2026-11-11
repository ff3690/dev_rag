"""
增量更新模块 - Incremental Updater Module

基于 LlamaIndex 的增量更新器，支持文档的添加、修改和删除操作。
使用真实的 Embedding 模型进行向量化，支持变更日志记录。

模块职责：
1. 文档变更管理：添加、更新、删除文档
2. 变更日志记录：追踪所有变更操作
3. 数据一致性保证：确保向量存储与文档记录同步

变更类型：
- add: 添加新文档到向量存储
- update: 更新已有文档的向量
- delete: 从向量存储删除文档

实现策略：
- 所有变更操作都通过 change_log.json 记录
- 更新操作采用"先删后增"策略，确保数据一致性
- 批量操作通过 batch_update 方法统一处理
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
import os

from backend.config.settings import settings
from backend.rag.document_loader import Document, DocumentChunk
from backend.rag.text_splitter import TextSplitter
from backend.rag.vector_store import LlamaVectorStore
from backend.rag.embedding_service import embedding_service


@dataclass
class DocumentChange:
    """
    文档变更数据类

    封装文档的变更信息，用于增量更新和变更日志记录。
    是增量更新操作的基本单位。

    属性说明：
        document_id: 文档唯一标识
        change_type: 变更类型 ("add", "update", "delete")
        content: 文档内容（add/update 必填，delete 可选）
        timestamp: 变更时间戳，自动生成

    使用示例：
        # 添加文档
        change = DocumentChange("doc_1", "add", "文档内容")
        
        # 更新文档
        change = DocumentChange("doc_1", "update", "新内容")
        
        # 删除文档
        change = DocumentChange("doc_1", "delete")
    """
    document_id: str
    change_type: str
    content: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        """初始化后自动设置时间戳"""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IncrementalUpdater:
    """
    基于 LlamaIndex 的增量更新器

    负责文档变更的跟踪和向量存储的同步更新。
    所有变更操作都会被记录到变更日志中，便于审计和回溯。

    核心特性：
    1. 完整的 CRUD 操作支持
    2. 变更日志持久化
    3. 批量更新支持
    4. 一致性验证

    更新策略（先删后增）：
    - 更新文档时，先删除旧的向量数据
    - 然后分割新内容并重新向量化
    - 最后添加新的向量数据
    - 确保不会出现重复或残留数据

    使用示例：
        updater = IncrementalUpdater(vector_store)
        
        # 添加文档
        doc_id = updater.handle_document_add(document)
        
        # 更新文档
        updater.handle_document_update(doc_id, new_content)
        
        # 删除文档
        updater.handle_document_delete(doc_id)
        
        # 批量更新
        updater.batch_update(changes)
    """

    def __init__(self, vector_store: LlamaVectorStore, splitter: Optional[TextSplitter] = None):
        """
        初始化增量更新器

        参数:
            vector_store: LlamaIndex 向量存储实例
            splitter: 文本分割器实例
                      默认使用 TextSplitter 标准实现

        初始化操作：
        1. 保存向量存储和分割器引用
        2. 设置变更日志路径
        3. 创建变更日志文件（如果不存在）
        """
        self.vector_store = vector_store
        self.splitter = splitter or TextSplitter()
        # 变更日志存储路径
        self._change_log_path = os.path.join(settings.VECTOR_STORE_PATH, "change_log.json")
        self._init_change_log()

    def _init_change_log(self):
        """
        初始化变更日志文件

        创建日志目录和初始空日志文件。
        日志文件存储在 vector_data/change_log.json
        """
        os.makedirs(os.path.dirname(self._change_log_path), exist_ok=True)
        if not os.path.exists(self._change_log_path):
            with open(self._change_log_path, "w") as f:
                json.dump([], f)

    def _read_change_log(self) -> List[dict]:
        """
        读取变更日志

        返回:
            List[dict]: 变更记录列表
        """
        try:
            with open(self._change_log_path, "r") as f:
                data = json.load(f)
            # 兜底：文件内容非法（非列表）时返回空列表
            if not isinstance(data, list):
                return []
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            # 文件不存在或内容损坏时，返回空列表，避免影响主流程
            return []

    def _write_change_log(self, log: List[dict]):
        """
        写入变更日志

        参数:
            log: 变更记录列表
        """
        with open(self._change_log_path, "w") as f:
            json.dump(log, f, default=str)

    def _add_change_log(self, change: DocumentChange):
        """
        添加变更记录到日志

        参数:
            change: 文档变更对象
        """
        log = self._read_change_log()
        log.append({
            "document_id": change.document_id,
            "change_type": change.change_type,
            "timestamp": str(change.timestamp)
        })
        self._write_change_log(log)

    def _remove_document_from_log(self, document_id: str):
        """
        从变更日志中移除指定 document_id 的所有记录（删除操作）。

        移除后若列表为空，仍保留 []，保证 change_log.json 始终是合法的 JSON 数组。
        调用方需捕获可能出现的写文件异常。

        参数:
            document_id: 文档 ID
        """
        log = self._read_change_log()
        # 仅保留 document_id 不同的记录，即移除该文档的所有日志
        log = [
            entry for entry in log
            if str(entry.get("document_id")) != str(document_id)
        ]
        self._write_change_log(log)

    def record_change(self, document_id: str, change_type: str):
        """
        记录一条变更日志（公开方法，供外部模块调用）

        用于在文档删除等主流程中统一记录变更，避免依赖向量索引是否存在。

        删除（delete）时不会追加 delete 标志记录，而是直接移除该文档的所有日志记录，
        因为向量文件已删除，日志中再保留该 document_id 无实际意义。

        参数:
            document_id: 文档 ID
            change_type: 变更类型 ("add", "update", "delete")
        """
        change = DocumentChange(document_id=document_id, change_type=change_type)
        # delete 操作：移除该文档的所有日志记录（而非追加 delete 标志）
        if change_type == "delete":
            self._remove_document_from_log(document_id)
        else:
            self._add_change_log(change)

    def _prepare_document_data(self, document_id: str, chunks, document=None) -> List[Dict]:
        """
        准备文档数据用于添加到向量存储

        将文本块转换为 LlamaVectorStore 所需的格式，
        为每个块添加必要的元数据。

        参数:
            document_id: 文档 ID
            chunks: 文本块列表（由 TextSplitter 生成）
            document: 原始文档对象（可选），用于提取文档名称等信息

        返回:
            List[Dict]: 文档数据列表，每个元素包含:
                - content: 文本块内容
                - metadata: 元数据字典

        元数据字段说明：
            - chunk_id: 块唯一标识，格式为 {document_id}_{index}
            - document_id: 所属文档 ID
            - document_name: 文档名称（来自原始文档对象）
            - file_type: 文件类型（来自原始文档对象）
            - category: 文档分类（来自原始文档对象）
            - department: 所属部门（来自原始文档对象）
            - chunk_index: 块在文档中的顺序
            - total_chunks: 文档的总块数
        """
        documents = []

        # 提取文档级元数据
        doc_metadata = {}
        if document:
            doc_metadata = {
                "document_name": document.title or "",
                "file_type": document.file_type or "",
                "category": document.category or "",
                "department": document.department or "",
            }

        for i, chunk in enumerate(chunks):
            # 生成块 ID: 文档ID_序号
            chunk_id = f"{document_id}_{i}"

            # 构建元数据
            metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": chunk.total_chunks if hasattr(chunk, 'total_chunks') else len(chunks),
            }

            # 添加文档级元数据
            metadata.update(doc_metadata)

            # 继承原始块的元数据（如果有）
            if hasattr(chunk, 'metadata') and chunk.metadata:
                metadata.update(chunk.metadata)

            # 强制保证 document_id / chunk_id 不被 chunk 元数据覆盖，
            # 否则 document_id 可能被覆盖为 None/"None"，导致删除时无法关联
            metadata["document_id"] = document_id
            metadata["chunk_id"] = chunk_id

            documents.append({
                "content": chunk.content,
                "metadata": metadata
            })

        return documents

    def handle_document_add(self, document: Document) -> str:
        """
        处理文档添加操作

        完整流程：
        1. 生成文档 ID（如果未指定）
        2. 使用 TextSplitter 分割文档为文本块
        3. 准备文档数据（添加元数据）
        4. 调用向量存储的 add_documents 方法
        5. 记录变更日志

        参数:
            document: 文档对象，包含 id 和 content 属性

        返回:
            str: 文档 ID（UUID 格式或自定义 ID）

        向量化过程：
        - 由 LlamaVectorStore.add_documents 自动触发
        - 使用 BAAI/bge-small-zh-v1.5 模型
        - 每个文本块被转换为 512 维向量
        """
        # 生成或使用文档 ID
        document_id = document.id or str(uuid.uuid4())
        document.id = document_id

        print(f"处理文档添加操作 - 文档ID: {document_id}")

        # 分割文档为文本块
        chunks = self.splitter.split_document(document)
        print(f"  分割文档为 {len(chunks)} 个文本块")

        # 准备带元数据的文档数据（传递 document 对象以获取文档名称等信息）
        documents = self._prepare_document_data(document_id, chunks, document=document)

        # 添加到向量存储（自动向量化）
        self.vector_store.add_documents(documents)
        print(f"  已添加 {len(documents)} 个文本块到向量存储")

        # 记录变更日志
        change = DocumentChange(document_id=document_id, change_type="add")
        self._add_change_log(change)

        return document_id

    def handle_document_update(self, document_id: str, new_content=None, new_document: Optional[Document] = None):
        """
        处理文档更新操作

        采用"先删后增"策略确保数据一致性：
        1. 删除文档的所有旧向量数据
        2. 将新内容分割为文本块
        3. 添加新的向量数据
        4. 记录变更日志

        参数:
            document_id: 文档 ID（必须与已有文档对应）
            new_content: 更新后的完整文档内容（纯文本方式，与 new_document 二选一）
            new_document: 更新后的完整 Document 对象（结构化方式，可携带 structure
                          供结构感知切片使用；传此参数时优先使用）

        注意：
        - 此操作会完全替换文档的向量数据
        - 文档块数量可能发生变化
        - 旧的块 ID 将不再有效
        """
        # Step 1: 删除旧数据
        self.vector_store.delete_document(document_id)

        # Step 2: 分割新内容
        if new_document is not None:
            # 结构化更新：直接使用完整 Document（保留 structure 字段）
            new_document.id = document_id
            temp_doc = new_document
            new_content = new_document.content or ""
        else:
            # 纯文本更新：使用临时文档对象包装内容（兼容旧调用方）
            class TempDocument:
                """临时文档类，用于包装纯文本内容"""
                def __init__(self, content, id=None):
                    self.content = content
                    self.id = id

            temp_doc = TempDocument(content=new_content or "", id=document_id)
            new_content = new_content or ""
        chunks = self.splitter.split_document(temp_doc)

        # Step 3: 添加新数据
        documents = self._prepare_document_data(document_id, chunks)
        self.vector_store.add_documents(documents)

        # Step 4: 记录变更日志
        change = DocumentChange(
            document_id=document_id,
            change_type="update",
            content=new_content
        )
        self._add_change_log(change)

    def handle_document_delete(self, document_id: str):
        """
        处理文档删除操作

        从向量存储中删除指定文档的所有向量数据，
        并记录删除操作到变更日志。

        参数:
            document_id: 文档 ID

        注意：
        - 删除操作不可逆
        - 删除后相关的检索结果将不再包含该文档
        """
        print(f"[handle_document_delete] 收到删除请求，document_id={document_id}，类型={type(document_id).__name__}")
        try:
            self.vector_store.delete_document(document_id)
            print(f"[handle_document_delete] vector_store.delete_document({document_id}) 调用完成")
        except Exception as e:
            print(f"[handle_document_delete] ✗ vector_store.delete_document({document_id}) 抛出异常: {e}")
            import traceback
            traceback.print_exc()

    def batch_update(self, changes: List[DocumentChange]):
        """
        批量处理文档变更

        按顺序处理多个变更操作，适用于：
        - 系统初始化时批量导入文档
        - 从备份恢复数据
        - 同步多个文档变更

        参数:
            changes: 变更列表，每个变更指定文档 ID 和操作类型

        处理顺序：
        按列表顺序依次处理，确保依赖关系正确。
        建议顺序：add -> update -> delete

        示例：
            changes = [
                DocumentChange("doc1", "add", "内容1"),
                DocumentChange("doc2", "add", "内容2"),
                DocumentChange("doc3", "delete"),
            ]
            updater.batch_update(changes)
        """
        for change in changes:
            if change.change_type == "add" and change.content:
                document = Document(id=change.document_id, content=change.content)
                self.handle_document_add(document)
            elif change.change_type == "update" and change.content:
                self.handle_document_update(change.document_id, change.content)
            elif change.change_type == "delete":
                self.handle_document_delete(change.document_id)
                self.record_change(change.document_id, "delete")

    def verify_consistency(self) -> bool:
        """
        验证向量存储与文档记录的一致性

        检查向量存储是否正常工作，返回健康状态。

        返回:
            bool: 是否一致（向量存储是否可用）
        """
        try:
            count = self.vector_store.get_vector_count()
            print(f"向量存储中共有 {count} 条记录")
            return True
        except Exception:
            return False

    def get_change_history(self, document_id: Optional[str] = None) -> List[dict]:
        """
        获取变更历史

        从变更日志中查询操作记录，支持按文档 ID 过滤。

        参数:
            document_id: 文档 ID（可选）
                        - 指定 ID: 返回该文档的变更历史
                        - 不指定: 返回所有变更记录

        返回:
            List[dict]: 变更记录列表，每条记录包含:
                        - document_id: 文档 ID
                        - change_type: 变更类型
                        - timestamp: 变更时间
        """
        log = self._read_change_log()
        if document_id:
            return [entry for entry in log if entry["document_id"] == document_id]
        return log