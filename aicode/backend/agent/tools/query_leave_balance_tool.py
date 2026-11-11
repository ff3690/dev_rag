"""
请假余额查询工具 - Query Leave Balance Tool

Agent 可通过此工具查询当前登录用户在指定年度的各假别余额（总额/已用/剩余）。
典型用法：用户发起请假申请前，由 Agent Graph 第一步调用此工具确认剩余天数，
再交给 check_leave_compliance 工具做合规校验，最后通过 send_email 发邮件。

与 sql_templates 的集成：
  - 主路径调用 execute_template("balance_all_types_by_user_year")
  - 指定假别时调用 "balance_by_user_type_year"
  - 异常时降级为 db.query(LeaveBalance) ORM 查询
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session


TOOL_DESCRIPTION = """
工具名称: query_leave_balance
功能: 查询当前登录用户的请假余额（年假/事假/病假等总额、已用、剩余天数）
参数:
  - leave_type (str, 可选): 假别 annual / sick / personal / maternity / marriage
    不传则返回所有假别的余额明细
返回: 余额文本（按假别列出 总额/已用/剩余天数），用于合规检查和向用户展示
使用场景:
  - 在发送请假申请邮件前，自动查询余额，判断是否足够
  - 用户主动查询假期结余
说明:
  - 查询范围默认为当前自然年度（如 2026）
  - 查询对象为"当前登录用户"，系统自动从上下文获取 user_id，无需用户传入
  - 若未找到任何余额记录，会返回空余额说明，提示联系管理员初始化
"""


# 假别中文名称映射（便于用户可读输出）
LEAVE_TYPE_LABELS: dict[str, str] = {
    "annual": "年假",
    "sick": "病假",
    "personal": "事假",
    "maternity": "产假",
    "marriage": "婚假",
}


def _label_for(leave_type: str) -> str:
    """假别英文枚举 → 中文名称；未知假别原样返回（便于调试）"""
    return LEAVE_TYPE_LABELS.get(leave_type, leave_type)


def _format_result(rows) -> str:
    """
    将查询结果行（leave_type, total_days, used_days, remaining_days）
    格式化为用户可读的文本。
    """
    if not rows:
        return "未查询到您当前年度的请假余额数据，建议联系管理员初始化。"

    lines = [f"【2026年度 请假余额查询结果】"]
    for row in rows:
        try:
            ltype = row["leave_type"]
            total = row["total_days"]
            used = row["used_days"]
            remain = row["remaining_days"]
        except (KeyError, TypeError):
            ltype, total, used, remain = row[0], row[1], row[2], row[3]

        label = _label_for(ltype)
        lines.append(
            f"- {label}（{ltype}）：总额{total}天，已用{used}天，剩余{remain}天"
        )
    return "\n".join(lines)


def query_leave_balance_tool(
    leave_type: str = "",
    db: Optional[Session] = None,
    current_user_id: Optional[int] = None,
) -> str:
    """
    查询当前登录用户的请假余额。

    参数:
        leave_type: 可选，指定假别 annual/sick/personal/maternity/marriage；
                    空字符串表示查询所有假别
        db: SQLAlchemy Session（由 execute_tool 通过签名匹配注入）
        current_user_id: 当前登录用户 id（由 execute_tool 注入）

    返回:
        str: 用户可读的余额文本
    """
    # ----- 前置校验 -----
    if not db:
        return "[工具错误] 数据库会话未初始化，无法查询余额。"
    if not current_user_id:
        return "[工具错误] 未获取到当前登录用户信息，请重新登录后再试。"

    # leave_type 合法性：非空则必须在白名单里
    lt = (leave_type or "").strip()
    if lt and lt not in LEAVE_TYPE_LABELS:
        available = "、".join(f"{k}({v})" for k, v in LEAVE_TYPE_LABELS.items())
        return (
            f"参数错误：leave_type='{lt}' 不是有效的假别编码。"
            f"可用：{available}；留空则查询所有假别。"
        )

    # 默认为当前自然年度（如 2026）
    year = datetime.now().year

    # ----- 主路径：参数化 SQL 模板 -----
    sql_failed = False
    rows = []
    try:
        from backend.agent.tools.sql_templates import execute_template

        if leave_type and leave_type.strip():
            rows = execute_template(
                db,
                "balance_by_user_type_year",
                {
                    "user_id": current_user_id,
                    "year": year,
                    "leave_type": leave_type.strip(),
                },
            )
        else:
            rows = execute_template(
                db,
                "balance_all_types_by_user_year",
                {"user_id": current_user_id, "year": year},
            )
    except Exception as e:
        print(f"[余额查询][降级] SQL模板路径失败: {e}，回退到ORM查询")
        sql_failed = True

    # ----- 降级路径：ORM 查询 -----
    if sql_failed or not rows:
        from backend.auth.models import LeaveBalance

        q = db.query(LeaveBalance).filter(
            LeaveBalance.user_id == current_user_id,
            LeaveBalance.year == year,
        )
        if leave_type and leave_type.strip():
            q = q.filter(LeaveBalance.leave_type == leave_type.strip())
        items = q.order_by(LeaveBalance.leave_type).all()

        # 组装为与 SQL 模板同字段的行结构（简化为 dict 迭代）
        rows = [
            {
                "leave_type": it.leave_type,
                "total_days": it.total_days,
                "used_days": it.used_days,
                "remaining_days": it.total_days - it.used_days,
            }
            for it in items
        ]

    # ----- 路径日志（ExperienceRecall Lesson：显式标记路径） -----
    path_tag = "SQL模板" if not sql_failed else "ORM-降级"
    print(
        f"[余额查询] 路径={path_tag} user_id={current_user_id} "
        f"year={year} leave_type={leave_type or 'ALL'} rows={len(rows)}"
    )
    return _format_result(rows)
