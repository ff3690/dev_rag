"""
请假合规检查工具 - Check Leave Compliance Tool

在"请假余额查询 → 合规检查 → 发邮件"三步链路中承担第二步：
  对照《员工请假管理制度》校验本次请假申请是否合规。

检查策略（两层判定 + LLM 解释）：
  1. **规则引擎（主判定，确定性）**：
     - R1 余额充足：事假/病假/年假的 remaining_days >= leave_days
     - R2 提前申请天数：事假≥3天提前3天 / 年假≥5天提前7天 / 病假≤1天无需提前
     - R3 病假证明：病假≥2天 需附诊断证明（通过则标注"需补证明"）
     - R4 日期合法性：start_date <= end_date（若提供 end_date）
     - R5 跨年度提醒（提示级，不阻断）
  2. **RAG 制度检索（佐证依据）**：
     - 直接复用 VectorRetriever 检索《员工请假管理制度》文档原文，
       作为"制度依据"段落的引用（避免工具循环依赖，不通过 rag_search 工具）
  3. **LLM 解释（辅助，用户可读）**：
     - 合规结论以规则引擎为准，LLM 仅负责将不通过原因、制度依据、调整建议
       润色为自然语言文本；LLM 调用失败或返回异常时降级为纯规则文本。

降级策略（ExperienceRecall Lesson：显式标记路径，不静默绕过）：
  - RAG 检索失败 → "制度依据"段落填默认摘要（不影响结论）
  - LLM 解释失败 → "建议"段落用规则引擎输出的纯文本替代
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional
from sqlalchemy.orm import Session


TOOL_DESCRIPTION = """
工具名称: check_leave_compliance
功能: 对照请假管理制度，校验本次请假是否合规（余额、提前申请、证明、日期）
参数:
  - leave_type (str, 必填): 假别 annual / sick / personal / maternity / marriage
  - leave_days (float, 必填): 请假天数
  - start_date (str, 必填): 起始日期，格式 YYYY-MM-DD
  - end_date (str, 可选): 结束日期，格式 YYYY-MM-DD（不传则默认=start_date, leave_days=1）
  - reason (str, 可选): 请假原因（用于病假等场景提示补充材料）
返回: 合规检查结构化文本：结论(通过/不通过) + 不通过原因 + 制度依据 + 建议
使用场景:
  - 在发送请假申请邮件前做合规预检查
  - 给用户提示缺少的材料或调整建议
说明:
  - 余额查询通过 query_leave_balance 工具结果 + 内部 ORM 校验（按当前用户+当前年度）
  - 系统根据 current_user_id 自动获取用户，无需传入
"""


# ============================================================
#  假别中英文映射（与 query_leave_balance 保持一致）
# ============================================================
LEAVE_TYPE_LABELS: dict[str, str] = {
    "annual": "年假",
    "sick": "病假",
    "personal": "事假",
    "maternity": "产假",
    "marriage": "婚假",
}


def _parse_date(s: str) -> Optional[date]:
    """解析 YYYY-MM-DD 日期字符串，失败返回 None"""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _advance_days(start_date: date) -> int:
    """计算从今天到 start_date 还剩多少天（负数表示过去日期）"""
    today = date.today()
    return (start_date - today).days


# ============================================================
#  余额查询（合规工具内部 ORM 查询，避免依赖上一步工具结果解析）
# ============================================================

def _query_balance_for_type(db, user_id: int, leave_type: str) -> Optional[dict]:
    """
    查询指定假别的余额记录，失败或不存在返回 None。

    优先走 SQL 模板，失败/无记录降级 ORM。
    返回 dict: {leave_type, total_days, used_days, remaining_days}
    """
    if not db or not user_id or not leave_type:
        return None
    year = date.today().year
    rows = []
    try:
        from backend.agent.tools.sql_templates import execute_template
        rows = execute_template(
            db,
            "balance_by_user_type_year",
            {"user_id": user_id, "year": year, "leave_type": leave_type},
        )
    except Exception as e:
        print(f"[合规检查] 余额SQL失败: {e}，降级ORM")

    if rows:
        row = rows[0]
        try:
            return {
                "leave_type": row["leave_type"],
                "total_days": row["total_days"],
                "used_days": row["used_days"],
                "remaining_days": row["remaining_days"],
            }
        except (KeyError, TypeError):
            return {
                "leave_type": row[0],
                "total_days": row[1],
                "used_days": row[2],
                "remaining_days": row[3],
            }

    # ORM 降级
    from backend.auth.models import LeaveBalance
    rec = (
        db.query(LeaveBalance)
        .filter(
            LeaveBalance.user_id == user_id,
            LeaveBalance.year == year,
            LeaveBalance.leave_type == leave_type,
        )
        .first()
    )
    if not rec:
        return None
    return {
        "leave_type": rec.leave_type,
        "total_days": rec.total_days,
        "used_days": rec.used_days,
        "remaining_days": rec.total_days - rec.used_days,
    }


# ============================================================
#  规则引擎：逐条检查，返回 (passed: bool, reasons: list[str], remarks: list[str], refs: list[str])
# ============================================================

def _run_rule_engine(
    leave_type: str,
    leave_days: float,
    start_date: Optional[date],
    end_date: Optional[date],
    balance: Optional[dict],
) -> tuple[bool, list[str], list[str], list[str]]:
    """
    规则引擎核心。

    返回:
        (是否通过, 不通过原因列表, 提醒列表, 制度依据条款摘要列表)
    """
    passed: bool = True
    reasons: list[str] = []
    remarks: list[str] = []
    refs: list[str] = []

    lt_label = LEAVE_TYPE_LABELS.get(leave_type, leave_type)

    # ---------- R4: 日期合法性 ----------
    if start_date is None:
        passed = False
        reasons.append("起始日期格式不合法，应为 YYYY-MM-DD")
    else:
        if end_date is not None and end_date < start_date:
            passed = False
            reasons.append("结束日期不能早于起始日期")
        # 过去日期：允许但提醒
        if (date.today() - start_date).days > 1:
            remarks.append("请假起始日期早于今天，属于补请，需确认主管是否接受")

    # ---------- R1: 余额充足 ----------
    remaining = 0
    if balance is None:
        passed = False
        reasons.append(f"未找到您的{lt_label}余额记录，请联系管理员初始化余额")
    else:
        remaining = int(balance.get("remaining_days", 0))
        total = int(balance.get("total_days", 0))
        used = int(balance.get("used_days", 0))
        refs.append(
            f"《员工请假管理制度》第2.1条：{lt_label}余额不足不得申请"
            f"（当前总额{total}天/已用{used}天/剩余{remaining}天，申请{leave_days}天）"
        )
        if leave_days > remaining:
            passed = False
            reasons.append(
                f"{lt_label}余额不足：剩余{remaining}天，申请{leave_days}天，"
                f"差额{leave_days - remaining:g}天"
            )

    # ---------- R2: 提前申请天数 ----------
    if start_date is not None:
        advance = _advance_days(start_date)
        # 规则：事假≥3天→提前3天；年假≥5天→提前7天；病假/婚假/产假不强制提前
        need_advance = 0
        if leave_type == "personal" and leave_days >= 3:
            need_advance = 3
        elif leave_type == "annual" and leave_days >= 5:
            need_advance = 7

        if need_advance > 0:
            refs.append(
                f"《员工请假管理制度》第3.2条：{leave_days:g}天及以上{lt_label}"
                f"需提前{need_advance}天提交申请"
            )
            if advance < need_advance:
                passed = False
                reasons.append(
                    f"未满足提前申请：{leave_days:g}天{lt_label}需提前{need_advance}天申请，"
                    f"当前距请假开始仅{advance}天"
                )

    # ---------- R3: 病假证明 ----------
    if leave_type == "sick" and leave_days >= 2:
        refs.append("《员工请假管理制度》第4.3条：病假≥2天需附二级及以上医院诊断证明")
        remarks.append(f"本次{leave_days:g}天病假需附诊断证明，提交后请补附件")

    # ---------- R5: 跨年度提醒 ----------
    if start_date is not None:
        year = start_date.year
        now_year = date.today().year
        if end_date is not None and end_date.year != year:
            remarks.append("本次请假跨越多个年度，系统会按实际年度分别扣减余额，请知悉")
        elif year > now_year:
            remarks.append(f"本次请假属{year}年度，若{year}年度额度未初始化，需联系管理员提前开账")

    return passed, reasons, remarks, refs


# ============================================================
#  RAG 制度原文检索
# ============================================================

def _rag_lookup_policy(vector_retriever) -> str:
    """
    用底层 VectorRetriever 检索《员工请假管理制度》相关段落，用作"制度依据"旁证。
    检索失败或无结果 → 返回空字符串（不影响主结论）。
    """
    if not vector_retriever:
        return ""
    try:
        docs = vector_retriever.retrieve(
            "员工请假管理制度 假期余额 提前申请 病假证明",
            top_k=2,
        )
        parts = []
        for doc in docs[:2]:
            doc_id = (doc.metadata or {}).get("document_id", "未知")
            content = (doc.content or "")[:400]
            parts.append(f"  - [{doc_id}] {content}")
        if not parts:
            return ""
        return "\n".join(parts)
    except Exception as e:
        print(f"[合规检查] RAG检索制度失败: {e}")
        return ""


# ============================================================
#  LLM 解释：将规则引擎结果润色为可读建议
# ============================================================

COMPLIANCE_LLM_PROMPT = """请基于以下合规检查结果，为用户生成一段简洁、专业的"建议与解释"文本。
要求：
1. 如果不通过，建议用户具体如何调整（改期、改用其他假别、补证明等）
2. 如果通过但有"提醒"，也给出相应提醒
3. 不要编造制度条款，只使用提供的"不通过原因""提醒""制度依据"
4. 200 字以内，分点罗列

【假别与天数】：{leave_info}
【是否通过】：{passed}
【不通过原因】：{reasons_str}
【提醒事项】：{remarks_str}
【制度依据摘要】：{refs_str}

请直接给出建议文本（不要 JSON）："""


def _llm_suggestion(
    leave_info: str,
    passed: bool,
    reasons: list[str],
    remarks: list[str],
    refs: list[str],
) -> str:
    """
    调用 LLM 生成建议；任何异常返回空字符串（由调用方用规则默认建议兜底）。
    直接复用 llm_service._call_api（已封装 DeepSeek 请求）。
    """
    try:
        from backend.rag.llm_service import llm_service
    except Exception as e:
        print(f"[合规检查] 无法加载LLM服务，跳过建议润色: {e}")
        return ""

    prompt = COMPLIANCE_LLM_PROMPT.format(
        leave_info=leave_info or "无",
        passed="通过" if passed else "不通过",
        reasons_str="；".join(reasons) if reasons else "（无）",
        remarks_str="；".join(remarks) if remarks else "（无）",
        refs_str="；".join(refs) if refs else "（无）",
    )
    messages = [
        {
            "role": "system",
            "content": "你是企业HR合规顾问，擅长按制度解释结果并给出行动建议。只根据给定素材输出，不要编造。",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        resp = llm_service._call_api(messages)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[合规检查] LLM建议生成失败，使用规则建议兜底: {e}")
        return ""


# ============================================================
#  主工具函数
# ============================================================

def check_leave_compliance_tool(
    leave_type: str,
    leave_days: float,
    start_date: str,
    end_date: str = "",
    reason: str = "",
    db: Optional[Session] = None,
    vector_retriever=None,
    current_user_id: Optional[int] = None,
) -> str:
    """
    对照请假管理制度校验本次申请是否合规。

    参数:
        leave_type: 假别 annual/sick/personal/maternity/marriage（必填）
        leave_days: 请假天数（必填）
        start_date: 起始日期 YYYY-MM-DD（必填）
        end_date:   结束日期 YYYY-MM-DD（可选）
        reason:     请假原因（可选，仅提示用）
        db:         数据库会话（注入）
        vector_retriever: 向量检索器（注入，用于RAG查询制度原文）
        current_user_id: 当前登录用户ID（注入，用于按隔离查询余额）

    返回:
        str: 结构化的合规检查文本
    """
    # ---------- 基本参数校验 ----------
    if not leave_type or leave_type not in LEAVE_TYPE_LABELS:
        available = ", ".join(f"{k}({v})" for k, v in LEAVE_TYPE_LABELS.items())
        return f"参数错误：leave_type 无效({leave_type})，可用：{available}"

    try:
        leave_days = float(leave_days)
        if leave_days <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return "参数错误：leave_days 必须是正数字（天数）"

    sd = _parse_date(start_date)
    ed = _parse_date(end_date) if end_date else None
    # end_date 不传但 leave_days>1 → 给个友好提醒（不阻断，交由规则判断）
    if not ed and leave_days > 1 and sd:
        # 尝试按 leave_days 推出 end_date（仅作参考，不影响规则逻辑）
        from datetime import timedelta
        ed = sd + timedelta(days=int(leave_days - 1))

    # ---------- 余额查询 ----------
    balance = _query_balance_for_type(db, current_user_id, leave_type)

    # ---------- 规则引擎 ----------
    passed, reasons, remarks, refs = _run_rule_engine(
        leave_type=leave_type,
        leave_days=leave_days,
        start_date=sd,
        end_date=ed,
        balance=balance,
    )

    # ---------- RAG 制度原文检索（旁证，非关键路径） ----------
    rag_evidence = _rag_lookup_policy(vector_retriever)

    # ---------- LLM 建议（兜底规则建议） ----------
    lt_label = LEAVE_TYPE_LABELS.get(leave_type, leave_type)
    leave_info_text = f"{lt_label} {leave_days:g}天，{start_date} ~ {end_date or start_date}"
    llm_text = _llm_suggestion(leave_info_text, passed, reasons, remarks, refs)

    # LLM 建议兜底：LLM 失败/空返回则用规则默认建议
    default_suggestions: list[str] = []
    if not passed:
        for r in reasons:
            # 简单规则化建议
            if "余额不足" in r:
                default_suggestions.append(
                    "建议减少请假天数，或改用其他尚有余额的假别"
                )
            elif "提前申请" in r:
                default_suggestions.append("建议将请假起始日期向后调整，以满足提前申请要求")
            elif "格式" in r or "结束日期" in r:
                default_suggestions.append("请修正日期后重新提交")
            else:
                default_suggestions.append(f"请处理：{r}")
    if remarks:
        default_suggestions.extend(remarks)
    suggestions_text = llm_text or ("\n".join(f"- {s}" for s in default_suggestions) if default_suggestions else "-")

    # ---------- 组装结果 ----------
    lines = [f"【合规检查结论】{'通过' if passed else '不通过'}"]
    lines.append(f"【申请概要】{leave_info_text}" + (f"，原因：{reason}" if reason else ""))

    if balance is not None:
        lines.append(
            f"【余额校验】{lt_label}总额{balance['total_days']}天，"
            f"已用{balance['used_days']}天，剩余{balance['remaining_days']}天，"
            f"本次申请{leave_days:g}天"
        )
    else:
        lines.append("【余额校验】未找到对应假别余额记录")

    if reasons:
        lines.append("【不通过原因】")
        lines.extend(f"  - {r}" for r in reasons)
    if remarks:
        lines.append("【提醒事项】")
        lines.extend(f"  - {r}" for r in remarks)

    if refs or rag_evidence:
        lines.append("【制度依据】")
        lines.extend(f"  - {r}" for r in refs)
        if rag_evidence:
            lines.append("  - 知识库检索到的制度原文片段：")
            lines.append(rag_evidence)

    lines.append("【建议】")
    lines.append(suggestions_text)

    # ---------- 路径日志 ----------
    llm_called = bool(llm_text)
    print(
        f"[合规检查] 结论={'通过' if passed else '不通过'} "
        f"leave_type={leave_type} days={leave_days:g} "
        f"LLM_建议={'命中' if llm_called else '规则兜底'} "
        f"RAG_依据={'命中' if rag_evidence else '无'}"
    )
    return "\n".join(lines)
