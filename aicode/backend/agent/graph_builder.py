"""
LangGraph Agent 图编排器 - Graph Builder

使用 LangGraph 构建 Agent 的任务编排图。
包含三个核心节点：decide（决策）、tool（执行）、generate（生成）。

流程图:
    START → decide_node ──→ tool_node → decide_node (循环)
                  │                           │
                  └────→ generate_node ─────→ END

decide_node: LLM 分析用户意图 → 决定 respond 还是 call_tool
tool_node:   执行工具调用，将结果写入状态
generate_node: 综合所有信息生成最终回答
"""

import json
import uuid
from typing import Literal
from langgraph.graph import StateGraph, END

from backend.config.settings import settings
from backend.rag.llm_service import llm_service
from backend.agent.agent_state import (
    AgentState,
    AgentPhase,
    AGENT_DECIDE_PROMPT,
    AGENT_GENERATE_PROMPT,
    DEFAULT_AGENT_SYSTEM_PROMPT,
)
from backend.agent.tools import (
    execute_tool,
    get_tool_descriptions,
)


# ==================== 请假流程追踪日志 ====================
# 请假相关工具集合：仅这三个工具触发【agent请假】追踪 print
_LEAVE_TRACE_TOOLS = {"query_leave_balance", "check_leave_compliance", "send_email"}

# 工具中文名（用于追踪日志可读性）
_LEAVE_TOOL_LABELS = {
    "query_leave_balance": "查询请假余额",
    "check_leave_compliance": "请假合规检查",
    "send_email": "发送邮件",
}


def _leave_trace_enabled() -> bool:
    """检查请假追踪日志是否开启"""
    return getattr(settings, "AGENT_LEAVE_TRACE", False)


# ==================== 节点函数 ====================

def decide_node(state: AgentState) -> AgentState:
    """
    决策节点：分析用户意图，决定下一步行动

    输入: AgentState（含 messages）
    输出: 更新后的 AgentState（含 tool_calls 或 final_answer）
    """
    state["phase"] = AgentPhase.DECIDING.value
    state["iteration"] = state.get("iteration", 0) + 1

    # 防止无限循环
    max_iter = state.get("max_iterations", 5)
    if state["iteration"] > max_iter:
        state["final_answer"] = "抱歉，处理超时。请简化您的问题或稍后再试。"
        state["phase"] = AgentPhase.FINISHED.value
        return state

    # 获取最后一条用户消息
    messages = state.get("messages", [])
    if not messages:
        state["final_answer"] = "您好！请问有什么可以帮助您的？"
        state["phase"] = AgentPhase.FINISHED.value
        return state

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        state["final_answer"] = "请问您有什么问题？"
        state["phase"] = AgentPhase.FINISHED.value
        return state

    # 构建决策 Prompt
    history = _format_history(messages[:-1]) if len(messages) > 1 else "（无历史对话）"
    tool_descs = get_tool_descriptions()
    system_prompt = _build_system_prompt(state.get("system_prompt", ""))
    known_info = _build_known_info(state)

    decide_prompt = AGENT_DECIDE_PROMPT.format(
        user_message=user_message,
        history=history,
        tool_descriptions=tool_descs,
        known_info=known_info,
    )

    # 调用 LLM 做决策
    try:
        decision_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": decide_prompt},
        ]
        print(">>>>>> [Agent] decide_node 调用 LLM 进行意图决策 <<<<<<")
        response = llm_service._call_api(decision_messages)
        decision_text = response["choices"][0]["message"]["content"].strip()

        # 解析 JSON 决策
        # 清理可能的前后缀
        decision_text = _clean_json_text(decision_text)

        try:
            decision = json.loads(decision_text)
        except json.JSONDecodeError:
            # JSON 解析失败，默认直接回复
            state["final_answer"] = decision_text
            state["phase"] = AgentPhase.FINISHED.value
            return state

        action = decision.get("action", "respond")

        if action == "call_tool" and decision.get("tool_calls"):
            tool_calls = decision["tool_calls"]
            # 确保是列表
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            state["tool_calls"] = tool_calls
            # ---- 请假流程追踪日志：第2步/第5步/第8步 意图识别 ----
            if _leave_trace_enabled():
                first_tool = tool_calls[0].get("name", "") if tool_calls else ""
                if first_tool in _LEAVE_TRACE_TOOLS:
                    if state["iteration"] == 1:
                        print(
                            f"【agent请假】第2步 意图识别 "
                            f"action=call_tool tool={first_tool} "
                            f"iteration={state['iteration']}"
                        )
                    elif state["iteration"] == 2:
                        print(
                            f"【agent请假】第5步 意图识别 "
                            f"action=call_tool tool={first_tool} "
                            f"iteration={state['iteration']}"
                        )
                    elif state["iteration"] == 3:
                        print(
                            f"【agent请假】第8步 意图识别 "
                            f"action=call_tool tool={first_tool} "
                            f"iteration={state['iteration']}"
                        )
            state["phase"] = AgentPhase.EXECUTING.value
            # 保存用户的回复文本（用于最终生成回答）
            if decision.get("response"):
                state["messages"].append({
                    "role": "assistant",
                    "content": decision["response"]
                })
        else:
            # 直接回复
            answer = decision.get("response", decision_text)
            state["final_answer"] = answer
            state["tool_calls"] = []
            state["phase"] = AgentPhase.GENERATING.value

    except Exception as e:
        print(f"[Agent decide] LLM 决策失败: {e}")
        # 降级：直接回复
        fallback_msg = {
            "role": "system",
            "content": "你是一个研发团队助手，直接回复用户的问题。"
        }
        try:
            print(">>>>>> [Agent] decide_node 决策失败，调用 LLM 降级回复 <<<<<<")
            resp = llm_service._call_api([fallback_msg, {"role": "user", "content": user_message}])
            answer = resp["choices"][0]["message"]["content"]
        except Exception:
            answer = "您好！我是研发团队智能助手，请问有什么可以帮助您的？"
        state["final_answer"] = answer
        state["phase"] = AgentPhase.GENERATING.value

    return state


def tool_node(state: AgentState) -> AgentState:
    """
    工具执行节点：执行工具调用

    输入: AgentState（含 tool_calls）
    输出: 更新后的 AgentState（含 tool_results）

    特殊逻辑：合规检查通过后自动追加 send_email 调用
      - 在 tool_node 内部（而非条件路由函数中）执行，因为条件路由函数
        不能修改 state，在 route_after_tool 中注入 tool_calls 不会生效。
      - 合规检查通过 → 从合规结果提取信息构造邮件主题/正文 → 执行 send_email
      - 合规检查不通过 → 不追加 send_email
    """
    state["phase"] = AgentPhase.EXECUTING.value

    tool_calls = state.get("tool_calls", [])
    # 改为追加模式：保留之前轮次的工具结果，不覆盖
    tool_results = state.get("tool_results", []) or []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if not name:
            continue

        # ---- 请假流程追踪日志：第3步/第6步 工具执行 / 最后一步 发送邮件 ----
        if _leave_trace_enabled() and name in _LEAVE_TRACE_TOOLS:
            iteration = state.get("iteration", 1)
            label = _LEAVE_TOOL_LABELS.get(name, name)
            if name == "send_email":
                # 最后一步：不写"第N步"，写"最后一步"
                to_preview = str(args.get("to", ""))[:40]
                subject_preview = str(args.get("subject", ""))[:40]
                print(
                    f"【agent请假】最后一步 发送邮件 "
                    f"to={to_preview} subject=\"{subject_preview}\""
                )
            elif iteration == 1:
                # 第3步：查询请假余额
                ctx_parts = [f"leave_type={args.get('leave_type', 'ALL')}"]
                ctx_str = " ".join(ctx_parts)
                print(
                    f"【agent请假】第3步 {label} "
                    f"tool={name} {ctx_str}"
                )
            elif iteration == 2:
                # 第6步：请假合规检查
                ctx_parts = [
                    f"leave_type={args.get('leave_type', '')}",
                    f"days={args.get('leave_days', '')}",
                    f"start={args.get('start_date', '')}",
                ]
                ctx_str = " ".join(ctx_parts)
                print(
                    f"【agent请假】第6步 {label} "
                    f"tool={name} {ctx_str}"
                )

        # 执行工具
        result = execute_tool(name, args)

        tool_results.append({
            "tool_name": name,
            "tool_args": args,
            "result": result,
        })

        # 添加工具调用消息到历史
        state["messages"].append({
            "role": "assistant",
            "content": f"[调用工具 {name}]"
        })
        state["messages"].append({
            "role": "tool",
            "content": result,
            "tool_name": name,
        })

        # ---- 确定性规则：合规检查通过后，在同一轮自动追加 send_email ----
        # 不能在 route_after_tool（条件路由函数）中修改 state，
        # 因为 LangGraph 条件路由函数对 state 的修改不会被保留。
        # 在 tool_node（节点函数）中修改 state 是安全的。
        if name == "check_leave_compliance" and "【合规检查结论】通过" in str(result):
            subject, body = _build_email_from_compliance_result(str(result))
            # ---- 请假流程追踪日志：第7步 路由判断（合规通过→自动追加send_email） ----
            if _leave_trace_enabled():
                print(
                    f"【agent请假】第7步 路由判断 "
                    f"末工具=check_leave_compliance(合规通过) → 自动追加send_email"
                )
            # ---- 请假流程追踪日志：最后一步 发送邮件 ----
            if _leave_trace_enabled():
                print(
                    f"【agent请假】最后一步 发送邮件 "
                    f'to=(系统自动补全) subject="{subject[:40]}"'
                )
            email_result = execute_tool("send_email", {"subject": subject, "body": body})
            tool_results.append({
                "tool_name": "send_email",
                "tool_args": {"subject": subject, "body": body},
                "result": email_result,
            })
            state["messages"].append({
                "role": "assistant",
                "content": "[调用工具 send_email]"
            })
            state["messages"].append({
                "role": "tool",
                "content": email_result,
                "tool_name": "send_email",
            })
            print(f"[tool_node] 合规检查通过 → 自动追加执行 send_email 完成")

    state["tool_results"] = tool_results

    return state


def generate_node(state: AgentState) -> AgentState:
    """
    生成节点：综合所有信息生成最终回答

    输入: AgentState（含 tool_results, context_docs, graph_context）
    输出: 更新后的 AgentState（含 final_answer）
    """
    state["phase"] = AgentPhase.GENERATING.value

    # 如果决策节点已经设置了 final_answer，直接使用
    if state.get("final_answer"):
        state["phase"] = AgentPhase.FINISHED.value
        return state

    # 获取用户消息
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # 构建工具结果文本
    tool_results = state.get("tool_results", [])
    tool_results_text = ""
    if tool_results:
        for tr in tool_results:
            tool_results_text += f"\n### {tr['tool_name']} 结果\n{tr['result']}\n"
    else:
        tool_results_text = "（未使用工具）"

    # 构建上下文
    history = _format_history(messages[:-1]) if len(messages) > 1 else "（无历史对话）"

    context_docs = state.get("context_docs", [])
    context_docs_text = ""
    if context_docs:
        for i, doc in enumerate(context_docs, 1):
            content = doc.get("content", "")[:600]
            context_docs_text += f"\n- 文档{i}: {content}\n"

    graph_context = state.get("graph_context", "")

    # 构建生成 Prompt
    generate_prompt = AGENT_GENERATE_PROMPT.format(
        user_message=user_message,
        tool_results=tool_results_text,
        context_docs=context_docs_text or "（无检索结果）",
        graph_context=graph_context or "（无图谱上下文）",
        history=history,
        known_info=_build_known_info(state),
    )

    # 调用 LLM 生成回答
    try:
        system_prompt = _build_system_prompt(state.get("system_prompt", ""))
        gen_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generate_prompt},
        ]
        print(">>>>>> [Agent] generate_node 调用 LLM 生成最终回答 <<<<<<")
        response = llm_service._call_api(gen_messages)
        answer = response["choices"][0]["message"]["content"]

        state["final_answer"] = answer
    except Exception as e:
        print(f"[Agent generate] LLM 生成失败: {e}")
        # 降级：使用工具结果作为回答
        if tool_results_text:
            state["final_answer"] = f"以下是根据工具查询到的信息:\n\n{tool_results_text}"
        else:
            state["final_answer"] = "抱歉，处理过程中出现了问题，请稍后重试。"

    state["phase"] = AgentPhase.FINISHED.value
    return state


# ==================== 路由函数 ====================

def route_after_decide(state: AgentState) -> Literal["tool_node", "generate_node"]:
    """决策后的路由：有工具调用 → tool_node，否则 → generate_node"""
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "tool_node"
    return "generate_node"


def route_after_tool(state: AgentState) -> Literal["decide_node", "generate_node"]:
    """
    工具执行后的启发式路由：

    目标：在不新增节点/边的前提下，支持"余额→合规→邮件"等多步链式编排。

    策略（按优先级）：
      1. 达到 max_iterations → generate_node（兜底，避免死循环）
      2. 本轮没有实际执行任何工具 → generate_node（tool_node 空跑，防止无限回环）
      3. 最末一个工具是"终端工具"（send_email/submit_application）→ generate_node
         （邮件已发送/申请已提交，任务完成，直接生成最终回复）
      4. 其余情况 → decide_node，让 LLM 根据 updated known_info(已调用工具追踪)
         判断下一步应该调谁（合规 → 邮件，依此类推）

    兼容性（办公用品场景）：
      predict_consumption → submit_application（终端）→ generate_node，正常终止，
      不受启发式影响（submit_application ∈ terminal_tools）。
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)
    if iteration >= max_iter:
        print(f"[路由route_after_tool] 达到 max_iterations={max_iter} → generate_node")
        return "generate_node"

    # ---- 最末执行的工具名（兼容 dict 与 object 两种 tool_results 存储形式） ----
    tool_results = state.get("tool_results", []) or []
    last_tool = None
    if tool_results:
        last = tool_results[-1]
        last_tool = last.get("tool_name") if isinstance(last, dict) else getattr(last, "tool_name", None)

    # ---- 本轮有没有新增工具结果：若没有 → 直接 generate（防止空回环） ----
    # iteration 从 0 起，每进一次 decide_node +1；tool_results 长度<迭代数视为空跑
    # 兜底：iteration >=1 但工具结果数 == (iteration -1) 时才说明本次 tool_node 没产出新结果
    expected_min = 1  # 至少有 1 条结果才考虑继续循环
    if len(tool_results) < expected_min and not last_tool:
        print("[路由route_after_tool] 无实际工具结果 → generate_node")
        return "generate_node"

    # ---- 终端工具判定 ----
    terminal_tools: set[str] = {"send_email", "submit_application"}
    if last_tool and last_tool in terminal_tools:
        print(f"[路由route_after_tool] 末工具={last_tool}(终端) → generate_node")
        return "generate_node"

    # ---- 合规检查不通过 → 直接 generate_node（不发送邮件） ----
    # 合规检查通过的情况已在 tool_node 中自动追加 send_email，末工具变为 send_email（终端），
    # 会走到上面的终端工具判定 → generate_node，不会到这里。
    if last_tool == "check_leave_compliance":
        print(f"[路由route_after_tool] 合规检查不通过 → generate_node（不发送邮件）")
        return "generate_node"

    # ---- 其余：循环回 decide_node 再决策 ----
    print(f"[路由route_after_tool] 末工具={last_tool}(非终端) → 回decide_node继续决策 iteration={iteration}/{max_iter}")
    # ---- 请假流程追踪日志：第4步/第7步 路由判断 ----
    if _leave_trace_enabled() and last_tool in _LEAVE_TRACE_TOOLS:
        if iteration == 1:
            print(
                f"【agent请假】第4步 路由判断 "
                f"末工具={last_tool}(非终端) → 回decide_node继续决策"
            )
        elif iteration == 2:
            print(
                f"【agent请假】第7步 路由判断 "
                f"末工具={last_tool}(非终端) → 回decide_node继续决策"
            )
    return "decide_node"


# ==================== 图构建 ====================

def build_agent_graph() -> StateGraph:
    """
    构建 LangGraph Agent 图

    返回:
        StateGraph: 编译后的 Agent 工作流图
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide_node", decide_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("generate_node", generate_node)

    # 设置入口
    workflow.set_entry_point("decide_node")

    # 添加边
    workflow.add_conditional_edges(
        "decide_node",
        route_after_decide,
        {
            "tool_node": "tool_node",
            "generate_node": "generate_node",
        }
    )

    workflow.add_conditional_edges(
        "tool_node",
        route_after_tool,
        {
            "decide_node": "decide_node",
            "generate_node": "generate_node",
        }
    )

    # 生成节点后结束
    workflow.add_edge("generate_node", END)

    # 编译图
    graph = workflow.compile()

    # 保存流程图为图片
    graph.get_graph().draw_png("my_flowchart.png")

    return graph


# ==================== 辅助函数 ====================

def _build_email_from_compliance_result(compliance_result: str) -> tuple[str, str]:
    """
    从合规检查工具的输出文本中提取申请信息，自动构造请假邮件主题和正文。

    合规结果格式示例：
        【合规检查结论】通过
        【申请概要】事假 3天，2026-09-01 ~ 2026-09-03，原因：家里有事
        ...

    返回: (subject, body)
    """
    import re

    # ---- 提取申请概要行 ----
    overview_match = re.search(r"【申请概要】(.+)", compliance_result)
    overview_text = overview_match.group(1).strip() if overview_match else ""

    # ---- 从概要中分离"原因" ----
    reason = ""
    if "，原因：" in overview_text:
        parts = overview_text.split("，原因：", 1)
        overview_text = parts[0].strip()
        reason = parts[1].strip()

    # ---- 构造邮件主题：假别+天数（如"事假3天申请"） ----
    # overview_text 形如 "事假 3天，2026-09-01 ~ 2026-09-03"
    subject_text = overview_text.split("，")[0].strip() if overview_text else "请假申请"
    subject = f"{subject_text}申请"

    # ---- 构造邮件正文 ----
    body_lines = ["您好，", ""]
    body_lines.append(f"本人因{'「' + reason + '」' if reason else '个人原因'}，特申请{overview_text}。")
    body_lines.append("")
    body_lines.append("恳请批准为盼。")
    body_lines.append("")
    body_lines.append("（此邮件由请假流程 Agent 在合规检查通过后自动生成发送）")
    body = "\n".join(body_lines)

    return subject, body


def _build_known_info(state: AgentState) -> str:
    """
    生成系统已知的用户信息文本 + 已调用工具追踪列表，注入到决策/生成 Prompt。

    目标：
      1) 避免 LLM 向用户重复询问系统已自动获取的信息（姓名/部门/余额/收件人/抄送）
      2) 让 decide_node 的 LLM 明确知道"已经调用过哪些工具"，从而按
         "余额→合规→邮件"的顺序推进下一步，不会跳步或重复调用。
    """
    name = (state.get("current_user_name") or "").strip() or "未知"
    dept = (state.get("current_user_department") or "").strip() or "未知"

    # ---- 已调用工具追踪：按出现顺序+去重，供 LLM 判断"下一步该调谁" ----
    tool_results = state.get("tool_results", []) or []
    called_tools: list[str] = []
    seen: set = set()
    for tr in tool_results:
        tname = tr.get("tool_name") if isinstance(tr, dict) else getattr(tr, "tool_name", None)
        if tname and tname not in seen:
            seen.add(tname)
            called_tools.append(tname)
    called_tools_text = (
        "、".join(called_tools) if called_tools else "（尚未调用任何工具）"
    )

    lines = [
        "## 系统已自动获取的用户信息（以下信息无需向用户询问，send_email/余额/合规工具会自动补全）",
        f"- 姓名（用于邮件落款）：{name}",
        f"- 所属部门（用于邮件主题 / 部门查询）：{dept}",
        "- 收件人（本部门负责人邮箱）：系统已根据部门自动从 dept 表查询，无需用户提供",
        "- 抄送（上级部门负责人邮箱）：系统已自动沿 superior 递归生成，无需用户提供",
        "- 请假余额（所有假别总额/已用/剩余）：系统通过 query_leave_balance 工具按当前用户+当前年度自动获取，无需用户提供",
        f"- 本对话已调用过的工具（用于判断下一步顺序，避免跳步/重复）：{called_tools_text}",
        "- 请假流程工具调用顺序：query_leave_balance → check_leave_compliance → send_email（合规不通过则不发邮件）",
    ]
    return "\n".join(lines)


def _build_system_prompt(custom_prompt: str) -> str:
    """
    构建最终的系统提示词

    若传入自定义 system_prompt，则原样返回（不调用 .format()，避免其中的
    花括号如 {请假类型}、{姓名} 被误当作 Python 占位符而抛 KeyError）；
    否则使用默认模板并注入工具描述。

    参数:
        custom_prompt: 用户自定义系统提示词（可为空）

    返回:
        str: 最终系统提示词
    """
    if custom_prompt and custom_prompt.strip():
        return custom_prompt
    return DEFAULT_AGENT_SYSTEM_PROMPT.format(
        tool_descriptions=get_tool_descriptions()
    )


def _format_history(messages: list) -> str:
    """格式化对话历史"""
    if not messages:
        return "（无历史对话）"

    lines = []
    for msg in messages[-10:]:  # 最多取最近10条
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"用户: {content[:200]}")
        elif role == "assistant":
            lines.append(f"助手: {content[:200]}")
        elif role == "tool":
            lines.append(f"[工具结果]")

    return "\n".join(lines) if lines else "（无历史对话）"


def _clean_json_text(text: str) -> str:
    """清理 LLM 返回的 JSON 文本（去除 markdown 代码块标记等）"""
    text = text.strip()

    # 去除 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip()


# ==================== 全局实例 ====================

# 编译一次，全局复用
agent_graph = build_agent_graph()
