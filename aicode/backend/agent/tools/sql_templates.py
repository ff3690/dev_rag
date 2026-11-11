"""
参数化 SQL 模板模块 - SQL Template Module

集中管理研发团队 RAG-Agent 中所有固定结构的 SQL 查询模板。

设计原则：
1. **防注入优先**：所有变量通过 SQLAlchemy 命名参数（:param_name）绑定，
   严禁字符串拼接；调用方使用 db.execute(text(sql), params) 执行。
2. **人工编写并校验**：所有 SQL 由工程师预先审查，不依赖 LLM 生成，
   保证安全、确定性与可测试性（ExperienceRecall 教训：避免动态 SQL
   绕过不透明的 LLM 输出）。
3. **schema 白名单**：仅允许引用 dept / leave_balances 两张白名单表，
   不触碰 users.password_hash 等敏感列（ExperienceRecall Lesson）。
4. **路径日志**：execute_template 成功/失败分别带路径标记输出，
   便于生产环境快速排查是否命中模板（ExperienceRecall Lesson）。
5. **异常可诊断**：模板名不存在 → 告警+抛 ValueError；SQL 执行失败 →
   记录原始 SQL 名+参数（不含敏感值）→ 抛异常由工具层降级。

模板分类：
- recipient_*: 邮件收件人/抄送（dept 表 + CTE 递归）
- balance_*:   请假余额（leave_balances 表 + 参数化过滤）
"""

from __future__ import annotations

from typing import Any
from sqlalchemy import text


# ============================================================
#  收件人/抄送查询类模板（dept 表 + superior 递归链）
# ============================================================

QUERY_RECIPIENT_SUPERIOR_CHAIN_SQL = """
-- 查询本部门负责人(level=0 → to 收件人) + 沿 superior 递归全部上级
-- 负责人(level>0 → cc 抄送)，单次 CTE 替代原 Python N 次 ORM 查询。
--
-- 设计要点：
--   1. 锚点同时支持"归一化精确匹配 + 去空格小写模糊匹配"两种策略，
--      复刻 _query_dept 的两级回退；SQLite 的 char(12288) 即全角空格 U+3000。
--   2. 递归 JOIN 条件对 dept.deptname / dept.superior 分别做归一化，
--      与锚点策略一致，避免因空格/全角差异导致上级链断裂。
--   3. dc.level < 20：深度上限，防御 superior 循环引用导致无限递归。
WITH RECURSIVE dept_chain(deptname, manager_mail, superior, level) AS (
    SELECT
        deptname,
        manager_mail,
        superior,
        0 AS level
    FROM dept
    WHERE trim(replace(deptname, char(12288), ' ')) = :dept_name
       OR lower(replace(replace(deptname, char(12288), ' '), ' ', ''))
          = lower(replace(:dept_name, ' ', ''))
    UNION ALL
    SELECT
        d.deptname,
        d.manager_mail,
        d.superior,
        dc.level + 1
    FROM dept d
    JOIN dept_chain dc
      ON trim(replace(d.deptname, char(12288), ' '))
       = trim(replace(dc.superior, char(12288), ' '))
    WHERE dc.level < 20
)
SELECT manager_mail, level
FROM dept_chain
WHERE manager_mail IS NOT NULL
  AND manager_mail != ''
ORDER BY level
"""

QUERY_RECIPIENT_SINGLE_DEPT_SQL = """
-- 备用模板：仅查询单部门负责人（不递归上级），用于降级路径精确查询验证。
SELECT manager_mail
FROM dept
WHERE trim(replace(deptname, char(12288), ' ')) = :dept_name
   OR lower(replace(replace(deptname, char(12288), ' '), ' ', ''))
      = lower(replace(:dept_name, ' ', ''))
LIMIT 1
"""


# ============================================================
#  请假余额查询类模板（leave_balances 表）
# ============================================================

QUERY_BALANCE_ALL_TYPES_SQL = """
-- 查询指定用户在指定年度的所有假别余额（含剩余天数计算）。
SELECT
    leave_type,
    total_days,
    used_days,
    (total_days - used_days) AS remaining_days
FROM leave_balances
WHERE user_id = :user_id
  AND year    = :year
ORDER BY leave_type
"""

QUERY_BALANCE_BY_USER_TYPE_YEAR_SQL = """
-- 查询指定用户在指定年度的某个假别余额（含剩余天数计算）。
SELECT
    leave_type,
    total_days,
    used_days,
    (total_days - used_days) AS remaining_days
FROM leave_balances
WHERE user_id    = :user_id
  AND year       = :year
  AND leave_type = :leave_type
LIMIT 1
"""


# ============================================================
#  模板注册表：按名称检索（集中管理，便于扩展）
# ============================================================

SQL_TEMPLATES: dict[str, str] = {
    # 收件人/抄送类
    "recipient_superior_chain": QUERY_RECIPIENT_SUPERIOR_CHAIN_SQL,
    "recipient_single_dept":    QUERY_RECIPIENT_SINGLE_DEPT_SQL,
    # 余额查询类
    "balance_all_types_by_user_year":   QUERY_BALANCE_ALL_TYPES_SQL,
    "balance_by_user_type_year":        QUERY_BALANCE_BY_USER_TYPE_YEAR_SQL,
}


# ============================================================
#  允许的表白名单（schema 约束校验）
# ============================================================
# ExperienceRecall 教训：validate_sql 必须校验 FROM/JOIN 目标表是否在
# 允许的 schema 白名单内。此处由人工审查模板后列出实际引用的表，
# execute_template 会额外做正则化表名校验兜底（见 _check_table_whitelist）。
ALLOWED_TABLES: set[str] = {"dept", "leave_balances"}


def _check_table_whitelist(sql: str) -> None:
    """
    执行前兜底：正则提取 FROM/JOIN 后的表名，确保仅来自 ALLOWED_TABLES。

    CTE 排除（关键修复）：
      - 先扫一遍 WITH RECURSIVE / WITH 子句中定义的 CTE 别名，
        这些别名会出现在 FROM/JOIN 右侧但**不是真实表**，必须排除。
      - 匹配模式：
          WITH RECURSIVE alias1(...) AS ( ... ) , alias2 AS ( ... )
          WITH alias AS ( ... )
    """
    import re

    # -------- Step 1: 收集 CTE 别名 --------
    # 先剥掉 "WITH RECURSIVE" 前缀（若存在），然后按 ", xxx AS (" 分隔扫描
    cte_aliases: set[str] = set()
    # WITH [RECURSIVE] 开头后紧跟的第一个别名
    m_head = re.search(
        r"WITH\s+(?:RECURSIVE\s+)?([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )
    if m_head:
        cte_aliases.add(m_head.group(1).lower())
    # 第二个及以后的 CTE 别名：", <alias> [可选(...)] AS ("
    m_tail = re.findall(
        r",\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*AS\s*\(",
        sql,
        flags=re.IGNORECASE,
    )
    for a in m_tail:
        cte_aliases.add(a.lower())

    # -------- Step 2: 扫 FROM/JOIN 目标表（排除 CTE 别名） --------
    tokens = re.findall(
        r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )
    allowed_lower = {x.lower() for x in ALLOWED_TABLES}
    for t in tokens:
        tl = t.lower()
        if tl in cte_aliases:
            continue  # CTE 别名 → 不是真实表，放行
        if tl not in allowed_lower:
            raise ValueError(
                f"[SQL安全] 模板引用了白名单外的表: {t}，"
                f"允许的表: {sorted(ALLOWED_TABLES)}"
            )


# ============================================================
#  对外 API
# ============================================================

def get_template(name: str) -> str:
    """
    按模板名取 SQL 原文。

    参数:
        name: 注册表 SQL_TEMPLATES 中定义的键名

    返回:
        str: SQL 文本

    抛出:
        ValueError: 模板名不存在（调用方负责降级）
    """
    if name not in SQL_TEMPLATES:
        available = ", ".join(sorted(SQL_TEMPLATES.keys()))
        raise ValueError(
            f"[SQL模板] 不存在的模板名: {name}，可用模板: {available}"
        )
    return SQL_TEMPLATES[name]


def execute_template(
    db,
    name: str,
    params: dict[str, Any] | None = None,
) -> list:
    """
    按模板名执行参数化 SQL，返回行对象列表。

    参数:
        db:     SQLAlchemy Session（由工具执行器注入，或测试中 SessionLocal）
        name:   模板名（键名来自 SQL_TEMPLATES）
        params: 命名参数字典，值需与模板中 :param_name 一一对应；缺省为 {}

    返回:
        list: Row 对象列表，可通过 row._mapping 或 row[0]/row["col"] 访问

    抛出:
        ValueError: 模板名不存在 / 引用白名单外表
        Exception:  数据库执行异常（调用方负责降级路径）

    日志说明（ExperienceRecall Lesson）：
        - 成功：输出 [SQL模板][SQL路径] name=xxx rows=N
        - 失败：输出 [SQL模板][SQL异常] name=xxx 原因=xxx，然后重新抛出
    """
    params = params or {}
    sql = get_template(name)
    _check_table_whitelist(sql)

    try:
        rows = list(db.execute(text(sql), params))
        print(
            f"[SQL模板][SQL路径] name={name} rows={len(rows)} "
            f"params_keys={sorted(params.keys())}"
        )
        return rows
    except Exception as e:
        print(
            f"[SQL模板][SQL异常] name={name} 原因={e} "
            f"params_keys={sorted(params.keys())}"
        )
        raise
