"""
Agent API 路由 - Agent Routes

提供智能体对话接口，支持多轮对话、流式响应和工具调用。

端点:
    POST /agent/chat     - Agent 对话（支持多轮）
    GET  /agent/tools    - 列出可用工具
    POST /agent/reset    - 重置对话历史
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.auth import models as auth_models
from backend.db.database import get_db
from backend.config.settings import settings
from backend.agent import (
    agent_graph,
    AgentState,
    AgentPhase,
    set_tool_context,
    ToolContext,
    TOOL_REGISTRY,
    get_tool_descriptions,
)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ==================== 请求/响应模型 ====================

class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    message: str = Field(..., description="用户消息", min_length=1)
    conversation_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    system_prompt: Optional[str] = Field(None, description="自定义系统提示词")
    stream: bool = Field(False, description="是否启用流式响应")


class AgentChatResponse(BaseModel):
    """Agent 对话响应"""
    answer: str = Field(..., description="Agent 回复")
    tool_calls_made: list = Field(default_factory=list, description="执行过的工具调用")
    conversation_id: str = Field(..., description="会话ID")
    phase: str = Field("finished", description="执行阶段")


class ToolInfo(BaseModel):
    """工具信息"""
    name: str
    description: str


# ==================== 会话管理 ====================

# 简单内存会话存储（生产环境应使用 Redis）
_conversations: dict = {}


def get_or_create_conversation(conversation_id: Optional[str] = None) -> tuple:
    """获取或创建会话"""
    if conversation_id and conversation_id in _conversations:
        cid = conversation_id
        history = _conversations[cid]
    else:
        cid = str(uuid.uuid4())[:8]
        history = []
        _conversations[cid] = history
    return cid, history


# ==================== API 端点 ====================

@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    current_user: auth_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Agent 对话端点

    支持多轮对话，Agent 会自主决定是否调用工具。

    使用示例 - 请假场景:
    POST /agent/chat
    {
        "message": "我要请事假，8月5号到8月6号",
        "conversation_id": "abc123"
    }

    使用示例 - 办公用品场景:
    POST /agent/chat
    {
        "message": "帮我根据上个月消耗预测本月办公用品需求，生成领用申请"
    }
    """
    from backend.rag.routes import retriever as vector_retriever, graphrag as gr

    # 设置工具上下文（注入当前用户三要素：id→余额查询隔离、部门→邮件抄送、姓名→落款）
    set_tool_context(ToolContext(
        db=db,
        vector_retriever=vector_retriever,
        graphrag_retriever=gr,
        current_user_id=current_user.id,
        current_user_department=current_user.department,
        current_user_name=current_user.name,
    ))

    # 获取或创建会话
    cid, history = get_or_create_conversation(request.conversation_id)

    # 添加用户消息到历史
    history.append({"role": "user", "content": request.message})

    # ---- 请假流程追踪日志：第1步 收到用户问题 ----
    _LEAVE_KEYWORDS = {"请假", "事假", "病假", "年假", "婚假", "产假", "休假", "销假", "调休"}
    if getattr(settings, "AGENT_LEAVE_TRACE", False) and any(
        kw in request.message for kw in _LEAVE_KEYWORDS
    ):
        msg_preview = request.message[:80].replace("\n", " ")
        print(f"【agent请假】第1步 收到用户问题 user={current_user.name} msg_preview=\"{msg_preview}\"")

    # 构建初始状态
    initial_state: AgentState = {
        "messages": list(history),
        "tool_calls": [],
        "tool_results": [],
        "context_docs": [],
        "graph_context": "",
        "final_answer": "",
        "phase": AgentPhase.DECIDING.value,
        "iteration": 0,
        "max_iterations": 5,
        "system_prompt": request.system_prompt or "",
        "conversation_id": cid,
        "current_user_id": current_user.id,
        "current_user_name": current_user.name or "",
        "current_user_department": current_user.department or "",
    }

    # 运行 Agent 图
    try:
        final_state = agent_graph.invoke(initial_state)

        # 提取调用过的工具
        tool_calls_made = []
        for tr in final_state.get("tool_results", []):
            tool_calls_made.append({
                "name": tr.get("tool_name", ""),
                "args": tr.get("tool_args", {}),
            })

        # 保存助手回复到历史
        answer = final_state.get("final_answer", "")
        if answer:
            history.append({"role": "assistant", "content": answer})

        return AgentChatResponse(
            answer=answer,
            tool_calls_made=tool_calls_made,
            conversation_id=cid,
            phase=final_state.get("phase", AgentPhase.FINISHED.value),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行异常: {str(e)}")


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """
    列出所有可用工具

    返回每个工具的名称和描述，帮助用户了解 Agent 的能力。
    """
    tools = []
    for name, tool in TOOL_REGISTRY.items():
        tools.append(ToolInfo(
            name=name,
            description=tool.description,
        ))
    return tools


@router.post("/reset")
async def reset_conversation(conversation_id: str):
    """
    重置指定会话

    清除对话历史，开始新的对话。
    """
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return {"message": f"会话 {conversation_id} 已重置", "status": "ok"}
    return {"message": f"会话 {conversation_id} 不存在", "status": "not_found"}
