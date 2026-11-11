"""
知识图谱 API 路由 - Knowledge Graph API Routes

该模块定义知识图谱的 RESTful API 接口。
提供图谱数据查询、全量重建和节点详情查询功能。

设计决策：
1. 使用 APIRouter 前缀 /graph 组织路由
2. 依赖注入数据库会话和用户认证
3. GET /data 返回 D3.js 可直接消费的图数据结构
4. POST /rebuild 支持手动全量重建
5. GET /node-detail/{id} 支持节点详情查询
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional

from backend.db.database import get_db
from backend.auth import models, dependencies
from backend.graph.graph_service import graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/data")
async def get_graph_data(
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取完整知识图谱数据
    
    返回所有节点和边数据，供前端 D3.js 力导向图渲染。
    节点包含颜色信息（按实体类型），边包含颜色和节点名称信息。
    
    参数：
        current_user: 当前登录用户（通过认证依赖注入）
        db: 数据库会话
    
    返回：
        dict: {
            "nodes": [{"id", "name", "type", "description", "color", ...}],
            "edges": [{"id", "source_node_id", "target_node_id", "relation", "color", "source_name", "target_name", ...}]
        }
    """
    data = graph_service.get_graph_data(db)
    
    # 添加统计信息
    type_counts = {}
    for node in data["nodes"]:
        t = node.get("type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    return {
        "success": True,
        "data": data,
        "stats": {
            "total_nodes": len(data["nodes"]),
            "total_edges": len(data["edges"]),
            "type_distribution": type_counts
        }
    }


@router.post("/rebuild")
async def rebuild_graph(
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    全量重建知识图谱
    
    清空现有图谱数据，重新从所有已入库文档中抽取实体和关系。
    该操作在后台执行，立即返回"重建已启动"响应。
    
    参数：
        background_tasks: FastAPI 后台任务
        current_user: 当前登录用户
        db: 数据库会话
    
    返回：
        dict: 包含重建状态消息
    """
    # 注意：全量重建可能耗时较长，建议在生产环境中使用后台任务队列
    result = graph_service.rebuild_all(db)
    return result


@router.get("/node-detail/{node_id}")
async def get_node_detail(
    node_id: int,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取节点详情
    
    查询指定节点的详细信息，包括关联的边、相邻节点和来源文档。
    
    参数：
        node_id: 节点ID
        current_user: 当前登录用户
        db: 数据库会话
    
    返回：
        dict: 节点详情，包含关联关系
    
    异常：
        HTTPException: 节点不存在时返回 404
    """
    detail = graph_service.get_node_detail(node_id, db)
    
    if not detail:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    
    return {
        "success": True,
        "data": detail
    }


@router.delete("/document/{document_id}")
async def delete_document_graph(
    document_id: int,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除指定文档关联的图谱数据
    
    当文档被删除时，清理相关的节点和边。
    
    参数：
        document_id: 文档ID
        current_user: 当前登录用户
        db: 数据库会话
    
    返回：
        dict: 删除统计
    """
    result = graph_service.delete_by_document(document_id, db)
    return {
        "success": True,
        "data": result
    }


@router.get("/stats")
async def get_graph_stats(
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取知识图谱统计信息
    
    返回节点数、边数、类型分布等统计。
    
    参数：
        current_user: 当前登录用户
        db: 数据库会话
    
    返回：
        dict: 统计信息
    """
    data = graph_service.get_graph_data(db)
    
    # 统计
    type_counts = {}
    for node in data["nodes"]:
        t = node.get("type", "未知")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    relation_counts = {}
    for edge in data["edges"]:
        r = edge.get("relation", "未知")
        relation_counts[r] = relation_counts.get(r, 0) + 1
    
    return {
        "success": True,
        "stats": {
            "total_nodes": len(data["nodes"]),
            "total_edges": len(data["edges"]),
            "type_distribution": type_counts,
            "relation_distribution": relation_counts
        }
    }
