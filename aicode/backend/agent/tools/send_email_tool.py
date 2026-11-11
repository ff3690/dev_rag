"""
邮件发送工具 - Send Email Tool

Agent 可通过此工具发送邮件，用于请假申请、通知等场景。
支持 SMTP 协议，配置通过环境变量管理。

使用方式（Agent 调用）:
    send_email(to="zhuguan@company.com", subject="事假申请", body="...")
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import datetime
import re
from backend.config.settings import settings


TOOL_DESCRIPTION = """
工具名称: send_email
功能: 发送邮件到指定收件人
参数:
  - to (str, 必填): 收件人邮箱地址
  - subject (str, 必填): 邮件主题
  - body (str, 必填): 邮件正文（支持纯文本和HTML）
  - cc (str, 可选): 手动抄送邮箱地址，多个用逗号分隔
返回: 发送结果描述
使用场景:
  - 发送请假申请邮件给主管
  - 发送通知邮件
  - 发送报告邮件
说明:
  - 邮件正文会自动加入当前真实日期（申请日期）
  - 系统会根据当前用户所属部门名称（users.department）查询 dept 表，将本部门负责人邮箱(manager_mail)自动作为 to 收件人
  - 系统会沿 superior 向上递归，将所有【上级部门】负责人邮箱自动添加为抄送人（不含本部门）
  - 邮件末尾会自动追加"发送人"落款（当前登录用户姓名）
注意: 如果邮件服务器未配置，会返回"未配置"提示，内容仍会生成供手动发送
"""


def _get_smtp_config():
    """获取 SMTP 配置（优先读取 settings（.env），回退到系统环境变量）"""
    return {
        "host": settings.SMTP_HOST or os.getenv("SMTP_HOST", ""),
        "port": settings.SMTP_PORT or int(os.getenv("SMTP_PORT", "587")),
        "username": settings.SMTP_USERNAME or os.getenv("SMTP_USERNAME", ""),
        "password": settings.SMTP_PASSWORD or os.getenv("SMTP_PASSWORD", ""),
        "from_addr": settings.SMTP_FROM or os.getenv("SMTP_FROM", "") or settings.SMTP_USERNAME or os.getenv("SMTP_USERNAME", ""),
        "use_tls": settings.SMTP_USE_TLS if settings.SMTP_HOST else os.getenv("SMTP_USE_TLS", "true").lower() == "true",
    }


def _normalize_dept_name(name: str) -> str:
    """
    部门名称归一化：去除首尾空格、统一全角空格为半角、过滤不可见字符。

    用途：users.department 与 dept.deptname 可能因输入来源不同存在细微差异
    （前后空格、全角空格、不可见字符），归一化后能提升匹配成功率。
    """
    if not name:
        return ""
    # 全角空格(U+3000) → 半角空格，再统一去除首尾空白
    return str(name).replace("\u3000", " ").strip()


def _query_dept(db, deptname: str):
    """
    按部门名称查询 dept 表。

    匹配策略（由精确到模糊逐级回退）：
        1. 归一化后的精确匹配（Dept.deptname == 归一化名称）
        2. 忽略大小写 + 去除所有空格的宽松匹配（兜底，提升健壮性）

    参数:
        db: 数据库会话
        deptname: 部门名称（users.department 字段值）

    返回:
        Dept 记录或 None
    """
    # 惰性导入，避免与 tools/__init__ 循环 import
    from backend.auth.models import Dept
    from sqlalchemy import func

    name = _normalize_dept_name(deptname)
    if not name:
        return None

    # 策略 1：归一化后的精确匹配（优先，性能最优）
    dept = db.query(Dept).filter(Dept.deptname == name).first()
    if dept:
        return dept

    # 策略 2：忽略大小写 + 去空格宽松匹配（容错，处理全角/半角/大小写差异）
    return db.query(Dept).filter(
        func.lower(func.replace(Dept.deptname, " ", ""))
        == name.replace(" ", "").lower()
    ).first()


def _collect_manager_emails_via_orm(db, department: str):
    """
    [降级路径] 现有 ORM + Python 循环递归方案。

    行为与原 _collect_manager_emails 完全一致：
      _query_dept() 查单部门 → while 循环沿 superior 每层一次查询
    参数/返回与原函数保持同签名，作为 CTE SQL 主路径失败后的兜底。

    （该函数原封不动地从旧版 _collect_manager_emails 主体搬迁而来）
    """
    dept = _query_dept(db, department) if department else None

    # 本部门负责人邮箱 -> 收件人 to
    to_email = ""
    if dept:
        to_email = (dept.manager_mail or "").strip()

    # 从上级部门开始沿 superior 向上递归收集负责人邮箱 -> 抄送 cc
    cc_mails: list[str] = []
    visited: set = set()
    depth = 0
    if dept:
        current = _normalize_dept_name(dept.superior)
    else:
        current = ""
    while current and current not in visited and depth < 20:
        visited.add(current)
        depth += 1
        sup_dept = _query_dept(db, current)
        if not sup_dept:
            break
        mail = (sup_dept.manager_mail or "").strip()
        if mail:
            cc_mails.append(mail)
        current = _normalize_dept_name(sup_dept.superior)

    # 去重（保持顺序）
    result: list = []
    seen: set = set()
    for m in cc_mails:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return to_email, result


def _collect_manager_emails_via_sql(db, department: str):
    """
    [主路径] 通过参数化 SQL 模板（CTE 递归）单次查询获取 to + cc。

    与 ORM 路径等价：
      level==0 的行 → 本部门负责人（to）
      level>0  的行 → 依次作为上级部门负责人（cc 顺序）
    CTE 内部对 deptname / superior 已做全角空格归一化 + 模糊匹配回退，
    深度上限 level<20 防 superior 循环引用，等价于 ORM 的 visited+depth<20。

    返回:
        (to_email, cc_list) 同 _collect_manager_emails_via_orm 签名

    抛出:
        Exception: SQL 执行异常 / CTE 返回结构异常（由调度器降级到 ORM 路径）
    """
    # 惰性导入：与 execute_template 同包，工具模块加载阶段即可用
    from backend.agent.tools.sql_templates import execute_template

    normalized_dept = _normalize_dept_name(department)
    # 若归一化后仍为空（department=""），直接返回空，由调用方按"部门未提供"统一处理
    if not normalized_dept:
        return "", []

    rows = execute_template(
        db,
        "recipient_superior_chain",
        # 传归一化的值（CTE 内部再次归一化做双保险）
        {"dept_name": normalized_dept},
    )

    to_email = ""
    cc_mails: list[str] = []

    for row in rows:
        # SQLAlchemy Row: 支持 row._mapping 或 row[0]/row["col"]
        try:
            mail = row["manager_mail"]
            level = row["level"]
        except (KeyError, TypeError):
            # 兜底：按列序访问
            mail, level = row[0], row[1]

        mail_clean = (mail or "").strip()
        if not mail_clean:
            continue

        if level == 0:
            # 本部门(level==0)若多行（极少），取第一个非空作 to
            if not to_email:
                to_email = mail_clean
        else:
            cc_mails.append(mail_clean)

    # cc 顺序去重（与 ORM 路径保持一致）
    cc_uniq: list = []
    seen: set = set()
    for m in cc_mails:
        if m not in seen:
            seen.add(m)
            cc_uniq.append(m)
    # 排除已进入 to 的邮箱
    cc_uniq = [m for m in cc_uniq if m != to_email]
    return to_email, cc_uniq


def _collect_manager_emails(db, department: str):
    """
    根据部门名称查询 dept 表，获取本部门负责人邮箱（to）和上级链负责人邮箱（cc）。

    调度策略（ExperienceRecall Lesson：显式标记路径，不隐式绕过）：
      1. 主路径：调用 _collect_manager_emails_via_sql  —— CTE 单次查询，高性能
      2. 降级条件：① SQL 抛 Exception ② SQL 返回 to+cc 全空
         → 降级到 _collect_manager_emails_via_orm —— Python 循环递归 ORM
      3. 日志明确标记"SQL路径命中/降级到ORM"，便于生产排查。

    参数:
        db: 数据库会话
        department: 当前用户所属部门名称（users.department 字段）

    返回:
        (to_email, cc_list): 本部门负责人邮箱(可能为空)，上级部门负责人邮箱去重列表
    """
    normalized_dept = _normalize_dept_name(department)
    if not normalized_dept:
        return "", []

    # ----- 主路径：CTE SQL 模板 -----
    to_email = ""
    cc_mails: list = []
    sql_ok = False
    try:
        to_email, cc_mails = _collect_manager_emails_via_sql(db, department)
        sql_ok = True
    except Exception as e:
        print(f"[收件人查询][降级] CTE主路径异常: {e}，回退到ORM路径")

    # 若 SQL 未抛但结果全空（可能部门不存在），仍走一次 ORM 路径确保不遗漏
    if sql_ok and not to_email and not cc_mails:
        print("[收件人查询][降级] CTE返回空结果，回退到ORM路径再尝试")
        try:
            to_email, cc_mails = _collect_manager_emails_via_orm(db, department)
        except Exception as e2:
            print(f"[收件人查询] ORM降级路径也失败: {e2}")

    # SQL 路径异常：直接走 ORM 降级
    if not sql_ok:
        try:
            to_email, cc_mails = _collect_manager_emails_via_orm(db, department)
        except Exception as e2:
            print(f"[收件人查询] ORM降级路径也失败: {e2}")

    # 路径标记日志（便于人工/自动化确认 CTE 实际生效）
    path_tag = "SQL-CTE" if sql_ok and (to_email or cc_mails) else "ORM-降级"
    print(
        f"[收件人查询] 路径={path_tag} "
        f"部门={normalized_dept!r} to={to_email!r} cc_count={len(cc_mails)}"
    )
    return to_email, cc_mails


def send_email_tool(
    to: str = "",    # 由必填改为可选：系统会根据当前用户部门自动从 dept 表补全本部门负责人邮箱作为收件人
    subject: str = "",
    body: str = "",
    cc: Optional[str] = None,
    db=None,
    current_user_department: Optional[str] = None,
    current_user_name: Optional[str] = None,
) -> str:
    """
    发送邮件

    说明:
        - to 为可选参数。若 LLM 未提供 to，系统会根据 current_user_department 查询 dept 表，
          将本部门负责人邮箱(manager_mail)自动补全为收件人。
        - 若最终仍无法确定收件人（部门查不到主管邮箱且用户未提供邮箱），
          会返回友好降级提示，引导向用户询问收件人邮箱，而不是抛 Python 异常。

    参数:
        to: 收件人邮箱（可选，系统自动补全部门主管邮箱）
        subject: 邮件主题
        body: 邮件正文（支持 HTML）
        cc: 抄送邮箱（可选，多个用逗号分隔）
        db: 数据库会话（由框架注入）
        current_user_department: 当前用户所属部门（用于自动抄送上级部门负责人）
        current_user_name: 当前用户姓名（用于邮件落款/发送人）

    返回:
        str: 发送结果描述
    """
    # ---- 申请日期：统一由系统生成，放在邮件末尾"发送人"落款的下方 ----
    # 说明：强制用系统真实今天的日期，避免 LLM 编造日期。
    #       申请日期不再插入到正文最前面，而是放到邮件末尾（发送人落款的下方），
    #       由下方的落款组装逻辑统一追加，保证格式统一、位置固定。
    today = datetime.date.today().strftime("%Y年%m月%d日")
    # 移除正文中 LLM 可能在任意位置编写的独立"申请日期"行，
    # 避免正文与落款下方的申请日期重复/不一致。
    # 仅删除"以申请日期开头"的独立行，不误删正文中夹杂的其他文字。
    body = re.sub(r"^[ \t]*申请日期[：:][^\n]*\n?", "", body, flags=re.MULTILINE)
    # 清理移除后可能残留的多余空行
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    # ---- 部门负责人查询：users.department 存的是部门名称，查询 dept 表 ----
    # dept_mail = 本部门负责人邮箱（作 to 收件人）；auto_cc = 上级部门负责人邮箱（作抄送）
    dept_mail = ""
    auto_cc: list = []
    if db is not None and current_user_department:
        dept_mail, auto_cc = _collect_manager_emails(db, current_user_department)

    # ---- 收件人 to：LLM 传入的收件人 + 本部门负责人邮箱（去重） ----
    # 说明：to 现为可选参数，此处用 (to or "") 兼容 LLM 不传 to 的情况，
    #       仅依赖系统从 dept 表补全的本部门负责人邮箱(dept_mail)。
    to_list: list = []
    for addr in [a.strip() for a in (to or "").split(",") if a.strip()]:
        if addr not in to_list:
            to_list.append(addr)
    if dept_mail and dept_mail not in to_list:
        to_list.append(dept_mail)
    final_to = ",".join(to_list)

    # ---- 收件人校验：必须至少有一个有效邮箱，否则降级要求用户提供 ----
    # 场景：LLM 未传 to，且系统根据部门也查不到本部门负责人邮箱(manager_mail)。
    # 此时不再继续发送（避免空收件人），而是返回友好提示，由 Agent 向用户询问收件人邮箱。
    if not final_to:
        dept_hint = (
            f"根据您的所属部门「{current_user_department or '未知'}」"
            f"未能从 dept 表查询到本部门负责人邮箱(manager_mail)"
            if current_user_department
            else "系统未获取到您的所属部门信息"
        )
        return (
            f"邮件发送失败：无法确定收件人。{dept_hint}。\n"
            f"为避免误发，请在回复中向用户询问并请其提供收件人邮箱地址，"
            f"收到后再重新调用 send_email 工具发送。"
        )

    # ---- 抄送 cc：手动抄送 + 上级部门负责人邮箱（去重，并排除已进入收件人的邮箱） ----
    cc_list: list = []
    if cc:
        cc_list.extend(addr.strip() for addr in cc.split(",") if addr.strip())
    cc_list.extend(m for m in auto_cc if m not in cc_list)
    cc_list = [m for m in cc_list if m not in to_list]
    final_cc = ",".join(cc_list) if cc_list else None
    print(f"final_cc: {final_cc}")
    # ---- 发送人落款 + 申请日期：落款在正文末尾，申请日期写在落款下方 ----
    sender_name = (current_user_name or "").strip()
    if not sender_name and final_to:
        sender_name = final_to.split("@")[0]

    # ---- 请假流程追踪日志：最后一步 发送邮件（邮件详情） ----
    from backend.config.settings import settings as _settings
    if getattr(_settings, "AGENT_LEAVE_TRACE", False):
        print(
            f"【agent请假】最后一步 发送邮件 "
            f"to={final_to} cc={final_cc} subject=\"{subject[:50]}\" "
            f"sender={sender_name} dept={current_user_department or ''}"
        )

    # 组装邮件末尾：先写"发送人"落款，再在落款下方写"申请日期"。
    # 即使 sender_name 为空（无发送人落款），申请日期也会追加在正文最后。
    footer_parts = []
    if sender_name:
        footer_parts.append(f"发送人：{sender_name}")
    footer_parts.append(f"申请日期：{today}")
    body = f"{body.rstrip()}\n\n" + "\n".join(footer_parts)

    config = _get_smtp_config()

    # 如果 SMTP 未配置，生成邮件预览供手动发送
    if not config["host"] or not config["username"]:
        preview = _generate_email_preview(final_to, subject, body, final_cc)
        return (
            "邮件服务器未配置（缺少 SMTP_HOST/SMTP_USERNAME 环境变量），"
            "已生成邮件内容预览，请手动发送:\n\n" + preview
        )

    try:
        # 构建邮件
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config["from_addr"]
        msg["To"] = final_to
        if final_cc:
            msg["Cc"] = final_cc

        # 添加纯文本和HTML版本
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # HTML 版本（将纯文本转换为简单HTML）
        html_body = body.replace("\n", "<br>")
        html = f"""<html><body style="font-family: Arial, sans-serif; padding: 20px;">
<p>{html_body}</p>
</body></html>"""
        msg.attach(MIMEText(html, "html", "utf-8"))

        # 发送邮件
        if config["use_tls"]:
            server = smtplib.SMTP(config["host"], config["port"])
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(config["host"], config["port"])

        server.login(config["username"], config["password"])
        all_recipients = [a.strip() for a in final_to.split(",") if a.strip()]
        if final_cc:
            all_recipients.extend(addr.strip() for addr in final_cc.split(","))
        server.sendmail(config["from_addr"], all_recipients, msg.as_string())
        server.quit()

        return f"邮件发送成功！\n- 收件人: {final_to}\n- 抄送: {final_cc}\n- 主题: {subject}"

    except Exception as e:
        error_msg = str(e)
        preview = _generate_email_preview(final_to, subject, body, final_cc)
        return (
            f"邮件发送失败: {error_msg}\n\n"
            "已生成邮件内容预览，请手动发送:\n\n" + preview
        )


def _generate_email_preview(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None
) -> str:
    """生成邮件内容预览"""
    lines = [
        "=" * 50,
        f"收件人: {to}",
    ]
    if cc:
        lines.append(f"抄送: {cc}")
    lines.extend([
        f"主题: {subject}",
        "-" * 50,
        body,
        "=" * 50,
    ])
    return "\n".join(lines)
