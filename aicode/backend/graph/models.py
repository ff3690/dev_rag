"""
知识图谱数据库模型 - Knowledge Graph Database Models

定义知识图谱的节点和关系边表结构。
- KnowledgeNode: 实体节点（技术术语、概念、工具、项目等）
- KnowledgeEdge: 实体间的关系边（依赖、使用、包含等）

设计决策：
1. 使用 SQLAlchemy ORM 与现有数据库框架保持一致
2. name + type 联合唯一约束防止重复实体
3. document_id 关联到 documents 表，支持来源追溯
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from datetime import datetime
from backend.db.database import Base


class KnowledgeNode(Base):
    """
    知识图谱节点模型
    
    表示从文档中抽取的实体，如技术术语、概念、工具、项目等。
    
    属性：
    - id: 节点唯一标识
    - name: 实体名称
    - type: 实体类型（技术/概念/工具/项目/流程/角色）
    - description: 实体描述
    - document_id: 来源文档ID
    - created_at: 创建时间
    """
    __tablename__ = "knowledge_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)
    description = Column(String(1000))
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("name", "type", name="uq_node_name_type"),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class KnowledgeEdge(Base):
    """
    知识图谱关系边模型
    
    表示两个实体之间的关系，如"Git 依赖 OpenSSL"、"Docker 属于 容器技术"。
    
    属性：
    - id: 边唯一标识
    - source_node_id: 源节点ID
    - target_node_id: 目标节点ID
    - relation: 关系类型（依赖/使用/包含/属于/关联/前置）
    - document_id: 来源文档ID
    - created_at: 创建时间
    """
    __tablename__ = "knowledge_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    source_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    target_node_id = Column(Integer, ForeignKey("knowledge_nodes.id"), nullable=False, index=True)
    relation = Column(String(100), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("source_node_id", "target_node_id", "relation", name="uq_edge_triple"),
    )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation": self.relation,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
