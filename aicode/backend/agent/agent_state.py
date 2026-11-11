"""
Agent 状态定义 - AgentState Definition

定义 LangGraph 智能体的状态结构，用于在节点间传递对话上下文、
工具调用结果和最终回答。

LangGraph 中，State 是一个 TypedDict，在图的各节点间流转。
每个节点读取 State 并返回部分更新的 State。
"""

from typing import TypedDict, List, Optional, Any, Annotated
from dataclasses import dataclass, field
from enum import Enum
import operator


class AgentPhase(str, Enum):
    """Agent 执行阶段"""
    DECIDING = "deciding"       # 意图决策中
    EXECUTING = "executing"     # 工具执行中
    GENERATING = "generating"   # 回答生成中
    FINISHED = "finished"       # 已完成


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    tool_args: dict = field(default_factory=dict)
    result: str = ""
    call_id: str = ""


class AgentState(TypedDict, total=False):
    """
    LangGraph Agent 状态

    在各节点间流转的核心状态对象。

    字段说明:
        messages: 完整消息历史 [{"role": "user/assistant/system/tool", "content": "..."}]
        tool_calls: 本轮待执行的工具调用列表
        tool_results: 已执行完毕的工具调用（含结果）
        context_docs: 检索到的知识库文档列表
        graph_context: 知识图谱结构化上下文文本
        final_answer: 最终回答（generate_node 产出）
        phase: 当前执行阶段
        iteration: 当前迭代次数（防止无限循环）
        max_iterations: 最大迭代次数
        system_prompt: 可自定义的系统提示词
    """
    messages: Annotated[List[dict], operator.add]   # 用 operator.add 累积消息
    tool_calls: List[dict]
    tool_results: List[ToolCall]
    context_docs: List[dict]
    graph_context: str
    final_answer: str
    phase: str
    iteration: int
    max_iterations: int
    system_prompt: str
    conversation_id: str
    # 当前登录用户三要素：按用户隔离(余额查询)、部门抄送、邮件落款
    current_user_id: int
    current_user_name: str
    current_user_department: str


# ==================== 默认系统提示词 ====================

DEFAULT_AGENT_SYSTEM_PROMPT = """你是一个专业的研发团队智能助手。你可以使用工具来帮助用户完成任务。

## 你的能力
1. **知识库检索**：使用 rag_search 工具在团队知识库中搜索文档、制度、技术方案等信息
2. **多轮对话**：如果你需要更多信息才能帮用户，请主动提问
3. **任务拆解**：复杂的任务可以分步完成，每步使用合适的工具

## 你的行为准则
- 优先使用工具查找准确信息，不要凭空编造
- 如果知识库中没有相关信息，诚实告知用户
- 回答要专业、简洁、有结构
- 标注信息来源

## 可用工具
你可以使用以下工具（通过调用它们来完成任务）：
{tool_descriptions}

## 当前对话
请根据对话历史和用户需求，决定下一步行动。"""


AGENT_DECIDE_PROMPT = """请分析用户的最新消息，决定下一步行动。

## 用户消息
{user_message}

## 对话历史
{history}

## 可用工具
{tool_descriptions}

## 系统已知的用户信息（系统已自动获取，无需向用户询问，send_email 工具会自动补全）
{known_info}

## 决策规则
1. 如果用户只是打招呼或闲聊，直接回复即可，不需要调用工具（设置 tool_calls=[]）
2. 如果用户的问题需要查知识库才能准确回答，调用 rag_search 工具
3. 请假相关：若用户明确提及请假（请事假/病假/年假、请假日期、请假原因等），
   按以下顺序**自动连环调用**（缺失信息先向用户提问），每一步只调用一个工具：
   a) 未查询过请假余额 → 调用 query_leave_balance 工具（无需参数，系统自动取当前用户所有假别余额）
   b) 已查到余额但未做合规检查 → 调用 check_leave_compliance 工具，填入用户消息中提到的假别(leave_type)、
      请假天数(leave_days)、起止日期(start_date/end_date)、请假原因(reason)
   c) 合规检查结论为"通过" → 调用 send_email 工具，按请假邮件模板生成主题+正文；
      用户的姓名、所属部门、收件人、抄送均由系统自动补全，无需用户提供
   d) 合规检查结论为"不通过" → 不调用 send_email，直接 respond 告知不通过原因 + 合规建议
4. 如果用户询问办公用品消耗/领用预测，且你已经拿到消耗清单，调用 predict_consumption 工具
5. 如果预测完成后需要提交申请，调用 submit_application 工具
6. 如果信息不完整（如请假缺少日期、请假原因、工作交接同事姓名），先向用户提问获取信息
   （不调用工具，在 response 中提问）；但不要询问姓名、部门、收件人、抄送、假期余额等系统已知信息

请以 JSON 格式返回你的决策（只返回JSON，不要其他内容）：
{{
    "action": "respond" 或 "call_tool",
    "response": "如果 action 是 respond，这里是给用户的回复",
    "tool_calls": [
        {{
            "name": "工具名称",
            "arguments": {{"参数名": "参数值"}}
        }}
    ]
}}

如果 action 是 "respond"，tool_calls 可以为空数组。"""


AGENT_GENERATE_PROMPT = """请根据以下信息生成最终回答。

## 用户问题
{user_message}

## 工具执行结果
{tool_results}

## 知识库检索结果（如有）
{context_docs}

## 知识图谱上下文（如有）
{graph_context}

## 对话历史
{history}

## 系统已知的用户信息（系统已自动补全，用于落款、收件人、抄送等）
{known_info}

## 生成规则
1. 综合所有信息生成完整、准确的回答
2. 标注信息来源
3. 如果是请假场景，按照文档中的格式生成邮件内容
4. 如果是办公用品领用场景，按照申请格式整理
5. 回答要专业、有结构

请直接给出回答内容（不要JSON格式）："""
