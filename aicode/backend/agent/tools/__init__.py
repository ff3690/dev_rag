"""
Agent 工具注册模块

提供工具注册、描述生成和执行能力。
所有 Agent 工具统一在此注册，便于管理和扩展。

添加新工具只需：
1. 创建 tool_xxx.py 文件
2. 导入到此模块的 TOOL_REGISTRY 中
3. 无需修改 Agent 框架任何代码
"""

from typing import Dict, Callable, Any, Optional
from sqlalchemy.orm import Session
from dataclasses import dataclass, field

from backend.rag.retriever import VectorRetriever
from backend.rag.graphrag_retriever import GraphRAGRetriever
from backend.agent.tools.rag_search_tool import (
    rag_search_tool,
    TOOL_DESCRIPTION as RAG_TOOL_DESC
)
from backend.agent.tools.send_email_tool import (
    send_email_tool,
    TOOL_DESCRIPTION as EMAIL_TOOL_DESC
)
from backend.agent.tools.predict_consumption_tool import (
    predict_consumption_tool,
    TOOL_DESCRIPTION as PREDICT_TOOL_DESC
)
from backend.agent.tools.submit_application_tool import (
    submit_application_tool,
    TOOL_DESCRIPTION as SUBMIT_TOOL_DESC
)
from backend.agent.tools.query_leave_balance_tool import (
    query_leave_balance_tool,
    TOOL_DESCRIPTION as BALANCE_TOOL_DESC,
)
from backend.agent.tools.check_leave_compliance_tool import (  # 紧随 P5 新建
    check_leave_compliance_tool,
    TOOL_DESCRIPTION as COMPLIANCE_TOOL_DESC,
)


@dataclass
class RegisteredTool:
    """已注册的工具"""
    name: str
    description: str
    func: Callable
    requires_db: bool = False


# ==================== 工具注册表 ====================

TOOL_REGISTRY: Dict[str, RegisteredTool] = {
    "rag_search": RegisteredTool(
        name="rag_search",
        description=RAG_TOOL_DESC.strip(),
        func=rag_search_tool,
        requires_db=True,
    ),
    "send_email": RegisteredTool(
        name="send_email",
        description=EMAIL_TOOL_DESC.strip(),
        func=send_email_tool,
        requires_db=True,
    ),
    "predict_consumption": RegisteredTool(
        name="predict_consumption",
        description=PREDICT_TOOL_DESC.strip(),
        func=predict_consumption_tool,
        requires_db=False,
    ),
    "submit_application": RegisteredTool(
        name="submit_application",
        description=SUBMIT_TOOL_DESC.strip(),
        func=submit_application_tool,
        requires_db=False,
    ),
    "query_leave_balance": RegisteredTool(
        name="query_leave_balance",
        description=BALANCE_TOOL_DESC.strip(),
        func=query_leave_balance_tool,
        requires_db=True,
    ),
    "check_leave_compliance": RegisteredTool(
        name="check_leave_compliance",
        description=COMPLIANCE_TOOL_DESC.strip(),
        func=check_leave_compliance_tool,
        requires_db=True,
    ),
}


def get_tool_descriptions() -> str:
    """生成所有工具的描述文本（供 LLM 决策参考）"""
    descriptions = []
    for tool in TOOL_REGISTRY.values():
        descriptions.append(f"### {tool.name}\n{tool.description}")
    return "\n\n".join(descriptions)


def get_tool_names() -> list:
    """获取所有已注册工具名称列表"""
    return list(TOOL_REGISTRY.keys())


# ==================== 工具执行上下文 ====================

@dataclass
class ToolContext:
    """工具执行上下文，保存依赖注入的实例"""
    db: Optional[Session] = None
    vector_retriever: Optional[VectorRetriever] = None
    graphrag_retriever: Optional[GraphRAGRetriever] = None
    # 当前登录用户信息：部门用于邮件抄送，姓名用于落款，id 用于余额查询按用户隔离
    current_user_id: Optional[int] = None
    current_user_department: Optional[str] = None
    current_user_name: Optional[str] = None


# 全局工具上下文
_tool_context: Optional[ToolContext] = None


def set_tool_context(ctx: ToolContext):
    """设置全局工具执行上下文"""
    global _tool_context
    _tool_context = ctx


def get_tool_context() -> Optional[ToolContext]:
    """获取全局工具执行上下文"""
    return _tool_context


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    执行指定的工具

    参数:
        tool_name: 工具名称
        arguments: 工具参数字典

    返回:
        str: 工具执行结果文本
    """
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return f"未知工具: {tool_name}，可用工具: {', '.join(get_tool_names())}"

    ctx = _tool_context

    try:
        # 根据工具需求注入上下文
        kwargs = dict(arguments)

        # 解析工具函数签名，仅注入该工具实际接受的参数，避免无关参数引发
        # "unexpected keyword argument" 错误（例如 send_email 不接受 vector_retriever）
        import inspect
        try:
            sig = inspect.signature(tool.func)
            accepted = set(sig.parameters.keys())
        except (TypeError, ValueError):
            accepted = None  # 签名解析失败时不限制注入

        def _inject(name, value):
            if accepted is None or name in accepted:
                kwargs[name] = value

        if tool.requires_db and ctx and ctx.db:
            # 需要数据库的工具按各自签名注入 db 与检索器
            _inject("db", ctx.db)
            if ctx.vector_retriever:
                _inject("vector_retriever", ctx.vector_retriever)
            if ctx.graphrag_retriever:
                _inject("graphrag_retriever", ctx.graphrag_retriever)

        # 向邮件/余额/合规工具注入当前用户信息（按签名匹配，无需则不注入）
        if ctx and (
            ctx.current_user_id is not None
            or ctx.current_user_department is not None
            or ctx.current_user_name is not None
        ):
            _inject("current_user_id", ctx.current_user_id)
            _inject("current_user_department", ctx.current_user_department)
            _inject("current_user_name", ctx.current_user_name)

        # ---- 针对 send_email：调用前校验收件人参数 to ----
        # 需求：调用发送邮件工具前必须验证 to 参数非空。
        # 背景：to 被设计为"可由系统自动补全"，LLM 可能信任系统而不传 to。
        #       若不前置校验，将直接进入工具内部；若部门也查不到主管邮箱，
        #       即使 to 改为可选也会发出空收件人邮件（或依赖工具内部降级）。
        # 处理流程：
        #   1) LLM 已提供 to -> 直接放行，交给工具执行。
        #   2) LLM 未提供 to -> 尝试根据用户部门从 dept 表自动补全主管邮箱。
        #   3) 补全后仍为空 -> 提前返回降级提示，要求用户提供收件人邮箱。
        if tool_name == "send_email":
            to_provided = (kwargs.get("to") or "").strip()
            if not to_provided:
                dept = (ctx.current_user_department if ctx else None) or ""
                # 部门已知且存在数据库会话时，尝试自动补全本部门主管邮箱
                if dept and ctx and ctx.db:
                    from backend.agent.tools.send_email_tool import _collect_manager_emails
                    auto_to, _ = _collect_manager_emails(ctx.db, dept)
                    if auto_to:
                        kwargs["to"] = auto_to
                # 自动补全失败（部门查不到主管邮箱）且 LLM 未提供收件人 -> 提前降级
                if not (kwargs.get("to") or "").strip():
                    dept_hint = (
                        f"根据您的所属部门「{dept}」未能从 dept 表查询到本部门负责人邮箱(manager_mail)"
                        if dept else "系统未获取到您的所属部门信息"
                    )
                    return (
                        f"send_email 缺少必需参数 to（收件人）。{dept_hint}。\n"
                        f"为避免误发，请在回复中向用户询问收件人邮箱地址，"
                        f"收到后再重新调用 send_email 工具发送。"
                    )

        result = tool.func(**kwargs)
        return str(result)

    except TypeError as e:
        return f"工具参数错误: {str(e)}。工具 {tool_name} 需要: {tool.description}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"工具执行异常: {str(e)}"
