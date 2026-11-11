"""
知识图谱核心服务 - Knowledge Graph Core Service

该模块提供知识图谱的实体抽取、存储和查询功能。
使用 DeepSeek LLM 自动从文档中抽取实体和关系，并管理图数据的增删改查。

设计决策：
1. 复用现有 llm_service 调用 DeepSeek API
2. 通过自定义 Prompt 让 LLM 输出结构化 JSON
3. 使用 SQLite INSERT OR IGNORE + 内存去重策略避免重复节点和边
4. 长文档分段处理，每段不超过 3000 字
5. 抽取失败不抛异常，仅记录日志
"""

import json
import re
import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.graph.models import KnowledgeNode, KnowledgeEdge
from backend.rag.llm_service import LLMService
from backend.config.settings import settings

logger = logging.getLogger(__name__)


# ==================== 图谱专用 LLM 实例 ====================
# 知识图谱构建（实体/关系抽取）使用独立配置的 LLM 实例，与检索生成的 llm_service 解耦，
# 可在 settings 中通过 GRAPH_LLM_* 配置不同的模型/API，未配置时自动 fallback 到 DEEPSEEK_*。
graph_llm = LLMService(
    api_key=settings.GRAPH_LLM_API_KEY or None,
    base_url=settings.GRAPH_LLM_API_BASE or None,
    model=settings.GRAPH_LLM_MODEL or None,
    temperature=settings.GRAPH_LLM_TEMPERATURE if settings.GRAPH_LLM_TEMPERATURE is not None else None,
    max_tokens=settings.GRAPH_LLM_MAX_TOKENS or None,
    timeout=settings.GRAPH_LLM_TIMEOUT or None,
)


# ==================== 实体抽取 Prompt ====================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的知识图谱构建助手。你的任务是从给定的文档内容中提取关键实体和它们之间的关系。

请严格遵循以下规则：
1. 实体类型只能是以下六种之一：技术、概念、工具、项目、流程、角色
2. 关系类型只能是以下六种之一：依赖、使用、包含、属于、关联、前置
3. 只提取文档中明确出现的实体，不要编造
4. 每个实体需要提供简短描述（20字以内）
5. 只提取有意义的关系，避免过于琐碎的关联

请以 JSON 格式输出，格式如下：
{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"source": "源实体名称", "target": "目标实体名称", "relation": "关系类型"}
  ]
}

注意：
- entities 数组中的 name 必须在文档中出现过
- relations 中的 source 和 target 必须是 entities 中已有的 name
- 关系是有方向的，source -> target 表示 source 对 target 的关系"""

ENTITY_EXTRACTION_USER_PROMPT = """请从以下文档内容中提取实体和关系。

## 文档标题
{title}

## 文档内容
{content}

请输出 JSON 格式的实体和关系列表。"""


# ==================== 图谱服务类 ====================

class GraphService:
    """
    知识图谱核心服务
    
    提供从文档抽取实体关系、图谱数据查询、全量重建等功能。
    
    使用示例：
        service = GraphService()
        result = service.extract_and_save(doc_content, doc_title, document_id, db)
    """
    
    # 实体类型颜色映射
    TYPE_COLORS = {
        "技术": "#00D4FF",
        "概念": "#7B61FF",
        "工具": "#FF6B6B",
        "项目": "#FFD93D",
        "流程": "#6BCB77",
        "角色": "#FF8C42"
    }
    
    # 关系类型颜色映射
    RELATION_COLORS = {
        "依赖": "#FF6B6B",
        "使用": "#00D4FF",
        "包含": "#7B61FF",
        "属于": "#FFD93D",
        "关联": "#6BCB77",
        "前置": "#FF8C42"
    }
    
    def extract_and_save(
        self,
        doc_content: str,
        doc_title: str,
        document_id: int,
        db: Session
    ) -> dict:
        """
        从文档中抽取实体和关系并保存到数据库
        
        参数：
            doc_content: 文档文本内容
            doc_title: 文档标题
            document_id: 文档数据库ID
            db: 数据库会话
        
        返回：
            dict: {"nodes_added": int, "edges_added": int, "error": str or None}
        """
        print(f"[图谱] extract_and_save 开始 - document_id={document_id}, title={doc_title}, 内容长度={len(doc_content) if doc_content else 0}")
        try:
            # 分段处理长文档
            segments = self._split_content(doc_content, max_chars=3000)
            print(f"[图谱] 文档分段完成 - 共 {len(segments)} 段")
            
            all_entities = []
            all_relations = []
            
            for i, segment in enumerate(segments):
                print(f"[图谱] 正在处理第 {i+1}/{len(segments)} 段 (长度={len(segment)})，调用 LLM 抽取实体...")
                entities, relations = self._extract_from_segment(segment, doc_title)
                print(f"[图谱] 第 {i+1} 段抽取结果 - 实体 {len(entities)} 个, 关系 {len(relations)} 条")
                all_entities.extend(entities)
                all_relations.extend(relations)
            
            # 去重实体
            unique_entities = self._deduplicate_entities(all_entities)
            print(f"[图谱] 实体去重完成 - 原始 {len(all_entities)} 个, 去重后 {len(unique_entities)} 个, 关系共 {len(all_relations)} 条")
            
            # 保存到数据库
            print("[图谱] 开始保存实体到数据库...")
            nodes_added = self._save_entities(unique_entities, document_id, db)
            print(f"[图谱] 实体保存完成 - 新增节点 {nodes_added} 个")
            
            print("[图谱] 开始保存关系到数据库...")
            edges_added = self._save_relations(all_relations, db, document_id)
            print(f"[图谱] 关系保存完成 - 新增边 {edges_added} 条")
            
            logger.info(
                f"知识图谱抽取完成 - 文档: {doc_title}, "
                f"新增节点: {nodes_added}, 新增边: {edges_added}"
            )
            print(f"[图谱] extract_and_save 完成 - document_id={document_id}, 新增节点: {nodes_added}, 新增边: {edges_added}")
            
            return {
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "error": None
            }
            
        except Exception as e:
            import traceback
            print(f"[图谱] 异常: extract_and_save 失败 - document_id={document_id}, 错误: {str(e)}")
            traceback.print_exc()
            logger.error(f"知识图谱抽取失败 - 文档: {doc_title}, 错误: {str(e)}")
            return {
                "nodes_added": 0,
                "edges_added": 0,
                "error": str(e)
            }
    
    def get_graph_data(self, db: Session) -> dict:
        """
        获取完整图谱数据（节点+边）
        
        参数：
            db: 数据库会话
        
        返回：
            dict: {"nodes": [...], "edges": [...]}
        """
        # 查询所有节点
        nodes = db.query(KnowledgeNode).all()
        nodes_data = []
        for node in nodes:
            node_dict = node.to_dict()
            node_dict["color"] = self.TYPE_COLORS.get(node.type, "#8B949E")
            nodes_data.append(node_dict)
        
        # 查询所有边
        edges = db.query(KnowledgeEdge).all()
        edges_data = []
        for edge in edges:
            edge_dict = edge.to_dict()
            edge_dict["color"] = self.RELATION_COLORS.get(edge.relation, "#8B949E")
            # 查找 source 和 target 节点名称
            source_node = db.query(KnowledgeNode).filter(
                KnowledgeNode.id == edge.source_node_id
            ).first()
            target_node = db.query(KnowledgeNode).filter(
                KnowledgeNode.id == edge.target_node_id
            ).first()
            if source_node and target_node:
                edge_dict["source_name"] = source_node.name
                edge_dict["target_name"] = target_node.name
            edges_data.append(edge_dict)
        
        return {
            "nodes": nodes_data,
            "edges": edges_data
        }
    
    def rebuild_all(self, db: Session) -> dict:
        """
        全量重建知识图谱
        
        清空现有图谱数据，重新从所有已入库文档中抽取。
        
        参数：
            db: 数据库会话
        
        返回：
            dict: 重建结果统计
        """
        from backend.auth.models import Document as DB_Document
        from backend.rag.document_loader import DocumentLoader
        
        # 清空现有图谱数据
        db.query(KnowledgeEdge).delete()
        db.query(KnowledgeNode).delete()
        db.commit()
        
        # 获取所有已入库文档
        documents = db.query(DB_Document).all()
        
        if not documents:
            return {
                "success": True,
                "total_documents": 0,
                "total_nodes": 0,
                "total_edges": 0,
                "errors": []
            }
        
        loader = DocumentLoader()
        total_nodes = 0
        total_edges = 0
        errors = []
        
        for doc in documents:
            try:
                # 读取文档内容
                loaded_doc = loader.load(doc.file_path)
                content = loaded_doc.content
                
                # 抽取并保存
                result = self.extract_and_save(content, doc.title, doc.id, db)
                total_nodes += result["nodes_added"]
                total_edges += result["edges_added"]
                
                if result["error"]:
                    errors.append({
                        "document_id": doc.id,
                        "title": doc.title,
                        "error": result["error"]
                    })
                    
            except Exception as e:
                logger.error(f"重建图谱失败 - 文档: {doc.title}, 错误: {str(e)}")
                errors.append({
                    "document_id": doc.id,
                    "title": doc.title,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_documents": len(documents),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "errors": errors
        }
    
    def get_node_detail(self, node_id: int, db: Session) -> dict:
        """
        获取节点详情
        
        参数：
            node_id: 节点ID
            db: 数据库会话
        
        返回：
            dict: 节点详情，包含关联的边和其他节点信息
        """
        node = db.query(KnowledgeNode).filter(KnowledgeNode.id == node_id).first()
        if not node:
            return None
        
        # 查询以该节点为源或目标的边
        outgoing_edges = db.query(KnowledgeEdge).filter(
            KnowledgeEdge.source_node_id == node_id
        ).all()
        incoming_edges = db.query(KnowledgeEdge).filter(
            KnowledgeEdge.target_node_id == node_id
        ).all()
        
        # 获取关联节点
        related_nodes = []
        for edge in outgoing_edges + incoming_edges:
            related_id = (
                edge.target_node_id if edge.source_node_id == node_id
                else edge.source_node_id
            )
            related_node = db.query(KnowledgeNode).filter(
                KnowledgeNode.id == related_id
            ).first()
            if related_node:
                related_nodes.append({
                    "node": related_node.to_dict(),
                    "relation": edge.relation,
                    "direction": "outgoing" if edge.source_node_id == node_id else "incoming"
                })
        
        # 查询来源文档信息
        doc_info = None
        if node.document_id:
            from backend.auth.models import Document as DB_Document
            doc = db.query(DB_Document).filter(DB_Document.id == node.document_id).first()
            if doc:
                doc_info = {
                    "id": doc.id,
                    "title": doc.title,
                    "file_type": doc.file_type
                }
        
        return {
            "node": node.to_dict(),
            "node_color": self.TYPE_COLORS.get(node.type, "#8B949E"),
            "related_nodes": related_nodes,
            "document": doc_info,
            "total_relations": len(outgoing_edges) + len(incoming_edges)
        }
    
    def delete_by_document(self, document_id: int, db: Session) -> dict:
        """
        删除指定文档关联的所有图谱数据
        
        参数：
            document_id: 文档ID
            db: 数据库会话
        
        返回：
            dict: 删除统计
        """
        # 找到该文档关联的所有边
        edges = db.query(KnowledgeEdge).filter(
            KnowledgeEdge.document_id == document_id
        ).all()
        edges_count = len(edges)
        for edge in edges:
            db.delete(edge)
        
        # 找到该文档关联的所有节点
        nodes = db.query(KnowledgeNode).filter(
            KnowledgeNode.document_id == document_id
        ).all()
        nodes_count = len(nodes)
        for node in nodes:
            # 先删除该节点相关的所有边
            related_edges = db.query(KnowledgeEdge).filter(
                or_(
                    KnowledgeEdge.source_node_id == node.id,
                    KnowledgeEdge.target_node_id == node.id
                )
            ).all()
            for e in related_edges:
                db.delete(e)
            db.delete(node)
        
        db.commit()
        return {
            "deleted_nodes": nodes_count,
            "deleted_edges": edges_count
        }
    
    # ==================== 私有方法 ====================
    
    def _split_content(self, content: str, max_chars: int = 3000) -> List[str]:
        """
        将长文档按段落分割
        
        参数：
            content: 文档内容
            max_chars: 每段最大字符数
        
        返回：
            List[str]: 分段后的文本列表
        """
        if len(content) <= max_chars:
            return [content]
        
        segments = []
        paragraphs = content.split("\n\n")
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) > max_chars and current:
                segments.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        
        if current.strip():
            segments.append(current.strip())
        
        return segments
    
    def _extract_from_segment(
        self,
        content: str,
        doc_title: str
    ) -> tuple:
        """
        调用 LLM 从文本段中抽取实体和关系
        
        参数：
            content: 文本段内容
            doc_title: 文档标题
        
        返回：
            tuple: (entities_list, relations_list)
        """
        # 构建 Prompt
        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ENTITY_EXTRACTION_USER_PROMPT.format(
                    title=doc_title,
                    content=content
                )
            }
        ]
        
        # 调用 LLM
        try:
            # 检查图谱专用 LLM 是否可用；未配置 API Key 则跳过本段，避免空转
            if not graph_llm.is_available:
                logger.warning("[图谱] 图谱 LLM 未配置 API Key，跳过实体抽取")
                return [], []

            print(f"[图谱]   → 开始调用 LLM 抽取实体 (model={graph_llm.model}, base_url={graph_llm.base_url})...")
            response = graph_llm._call_api(messages)
            answer = response["choices"][0]["message"]["content"]
            print(f"[图谱]   → LLM 返回成功, 响应长度={len(answer)}, 内容前200字符: {answer[:200]}")
            
            # 解析 JSON
            data = self._parse_json_response(answer)
            
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            print(f"[图谱]   → JSON 解析完成 - 实体 {len(entities)} 个, 关系 {len(relations)} 条")
            
            # 验证实体类型
            valid_types = {"技术", "概念", "工具", "项目", "流程", "角色"}
            valid_relations = {"依赖", "使用", "包含", "属于", "关联", "前置"}
            
            entities = [
                e for e in entities
                if e.get("type") in valid_types
            ]
            relations = [
                r for r in relations
                if r.get("relation") in valid_relations
            ]
            
            return entities, relations
            
        except Exception as e:
            import traceback
            print(f"[图谱]   → 异常: LLM 实体抽取调用失败: {str(e)}")
            traceback.print_exc()
            logger.error(f"LLM 实体抽取调用失败: {str(e)}")
            return [], []
    
    def _parse_json_response(self, text: str) -> dict:
        """
        从 LLM 响应中解析 JSON
        
        处理 LLM 可能返回的各种格式：
        - 纯 JSON
        - 被 markdown 代码块包裹的 JSON
        - 文本中嵌入的 JSON
        
        参数：
            text: LLM 响应文本
        
        返回：
            dict: 解析后的 JSON 对象
        """
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # 尝试提取 JSON 对象
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试修复常见的 JSON 格式问题
            text = text.replace("'", '"')  # 单引号替换双引号
            text = re.sub(r',\s*}', '}', text)  # 移除尾部逗号
            text = re.sub(r',\s*]', ']', text)  # 移除数组尾部逗号
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"无法解析 LLM JSON 响应: {text[:200]}")
                return {"entities": [], "relations": []}
    
    def _deduplicate_entities(self, entities: List[dict]) -> List[dict]:
        """
        实体去重
        
        按 (name, type) 组合去重，保留首次出现的。
        
        参数：
            entities: 实体列表
        
        返回：
            List[dict]: 去重后的实体列表
        """
        seen = set()
        unique = []
        for entity in entities:
            key = (entity["name"], entity.get("type", "概念"))
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique
    
    def _save_entities(
        self,
        entities: List[dict],
        document_id: int,
        db: Session
    ) -> int:
        """
        保存实体到数据库

        使用 SQLite INSERT OR IGNORE + 条件 DO UPDATE 策略：
        1. 新实体直接插入
        2. 已存在实体：仅当新描述更详细时才原子更新，避免额外 SELECT 查询

        参数：
            entities: 实体列表
            document_id: 文档ID
            db: 数据库会话

        返回：
            int: 写入行数（新增或更新均计入）
        """
        from sqlalchemy import case, func as sa_func

        count = 0
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue

            entity_type = entity.get("type", "概念")
            new_desc = entity.get("description", "")[:1000].strip() or None

            node_tbl = KnowledgeNode.__table__

            if new_desc is None:
                # 无描述：仅插入，冲突时静默忽略
                stmt = sqlite_insert(node_tbl).values(
                    name=name,
                    type=entity_type,
                    description=None,
                    document_id=document_id
                ).on_conflict_do_nothing(
                    index_elements=[node_tbl.c.name, node_tbl.c.type]
                )
            else:
                # 有描述：原子条件更新（仅新描述更长时才覆盖）
                # 注：on_conflict_do_update 的赋值必须放在 set_ 字典中
                stmt = sqlite_insert(node_tbl).values(
                    name=name,
                    type=entity_type,
                    description=new_desc,
                    document_id=document_id
                ).on_conflict_do_update(
                    index_elements=[node_tbl.c.name, node_tbl.c.type],
                    set_={
                        "description": case(
                            (
                                node_tbl.c.description.is_(None) |
                                (sa_func.length(new_desc) > sa_func.length(node_tbl.c.description)),
                                new_desc
                            ),
                            else_=node_tbl.c.description
                        )
                    }
                )

            result = db.execute(stmt)
            # rowcount 可能为 -1（未知）或 0（未改变），> 0 视为有行被写入
            if (result.rowcount or 0) > 0:
                count += 1

        try:
            db.commit()
            print(f"[图谱]   → 实体入库提交成功，本次处理 {count} 个节点")
        except Exception as e:
            import traceback
            print(f"[图谱]   → 异常: 实体入库 commit 失败: {str(e)}")
            traceback.print_exc()
            raise
        return count
    
    def _save_relations(
        self,
        relations: List[dict],
        db: Session,
        document_id: int
    ) -> int:
        """
        保存关系到数据库

        使用 INSERT OR IGNORE 策略避免重复边，不再依赖 autoflush 下的"先查后插"。
        先在内存中按 (source, target, relation) 去重，再用 SQLite
        on_conflict_do_nothing 确保数据库层面不会触发唯一约束冲突。

        参数：
            relations: 关系列表
            db: 数据库会话
            document_id: 文档ID

        返回：
            int: 新增边数
        """
        # 第一步：内存去重，防止同批次 LLM 返回的重复关系
        seen = set()
        unique_relations = []
        for relation in relations:
            source_name = relation.get("source", "").strip()
            target_name = relation.get("target", "").strip()
            rel_type = relation.get("relation", "").strip()

            if not source_name or not target_name or not rel_type:
                continue

            key = (source_name, target_name, rel_type)
            if key in seen:
                continue
            seen.add(key)
            unique_relations.append(relation)

        count = 0
        for relation in unique_relations:
            source_name = relation.get("source", "").strip()
            target_name = relation.get("target", "").strip()
            rel_type = relation.get("relation", "").strip()

            # 查找节点（节点已在 _save_entities 中提交，可查询到）
            source_node = db.query(KnowledgeNode).filter(
                KnowledgeNode.name == source_name
            ).first()
            target_node = db.query(KnowledgeNode).filter(
                KnowledgeNode.name == target_name
            ).first()

            if not source_node or not target_node:
                continue

            # 使用 SQLite INSERT OR IGNORE，命中唯一约束时静默跳过
            # 直接指定列名作为冲突目标，避免约束名解析问题
            edge_tbl = KnowledgeEdge.__table__
            stmt = sqlite_insert(edge_tbl).values(
                source_node_id=source_node.id,
                target_node_id=target_node.id,
                relation=rel_type,
                document_id=document_id
            ).on_conflict_do_nothing(
                index_elements=[
                    edge_tbl.c.source_node_id,
                    edge_tbl.c.target_node_id,
                    edge_tbl.c.relation
                ]
            )

            result = db.execute(stmt)
            # rowcount 可能为 -1（未知）或 0（被忽略），仅 > 0 视为成功插入
            if (result.rowcount or 0) > 0:
                count += 1

        try:
            db.commit()
            print(f"[图谱]   → 关系入库提交成功，本次新增 {count} 条新边")
        except Exception as e:
            import traceback
            print(f"[图谱]   → 异常: 关系入库 commit 失败: {str(e)}")
            traceback.print_exc()
            raise
        return count


# ==================== 全局单例 ====================

graph_service = GraphService()
