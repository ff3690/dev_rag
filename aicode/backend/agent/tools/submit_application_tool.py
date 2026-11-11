"""
第三方系统申请提交工具 - Submit Application Tool

Agent 可通过此工具向第三方 REST 系统提交领用申请。
第三方系统是一个模拟的 REST API，接受标准格式的申请数据。

典型场景:
    Agent 预测完成 → 生成申请单 → 调本工具提交 → 返回受理结果
"""

import os
import json
import httpx
from typing import Optional

from backend.config.settings import settings


TOOL_DESCRIPTION = """
工具名称: submit_application
功能: 向第三方办公用品管理系统提交领用申请单
参数:
  - application_data (str, 必填): JSON 格式的申请数据，需包含:
    - applicant: 申请人姓名
    - department: 部门
    - items: 申请物品列表 [{name, quantity, unit, category}]
    - reason: 申请原因
    - predicted_by: 预测依据(如 "LLM预测分析")
返回: 第三方系统的受理结果
使用场景:
  - 办公用品领用申请提交
  - 物资申领
"""


# 申请单格式模板说明
APPLICATION_FORMAT = """
申请单格式要求（JSON）:
{
  "applicant": "申请人姓名",
  "department": "部门",
  "application_date": "申请日期 YYYY-MM-DD",
  "items": [
    {
      "name": "物品名称",
      "quantity": 数量,
      "unit": "单位(个/支/包/箱)",
      "category": "分类(文具/耗材/清洁/电子/其他)",
      "estimated_price": 预估单价
    }
  ],
  "total_items": 物品总类数,
  "total_quantity": 物品总数量,
  "reason": "申请原因",
  "predicted_by": "预测来源",
  "urgency": "紧急程度(normal/urgent)"
}
"""


def submit_application_tool(
    application_data: str,
    api_url: Optional[str] = None
) -> str:
    """
    向第三方系统提交领用申请

    参数:
        application_data: JSON 格式的申请数据
        api_url: 第三方系统 API 地址（可选，默认从环境变量读取）

    返回:
        str: 提交结果描述
    """
    # 解析申请数据
    try:
        if isinstance(application_data, str):
            # 尝试从文本中提取 JSON（LLM 可能在 JSON 前后加说明文字）
            data = _extract_json(application_data)
            if data is None:
                return (
                    f"申请数据格式错误，无法解析为 JSON。\n\n"
                    f"请确保申请数据符合以下格式:\n{APPLICATION_FORMAT}\n\n"
                    f"收到的数据:\n{application_data[:500]}"
                )
        else:
            data = application_data
    except Exception as e:
        return f"申请数据解析失败: {str(e)}"

    # 获取第三方 API 地址（优先 settings（.env），回退到系统环境变量/默认值）
    if api_url is None:
        api_url = settings.SUPPLY_API_URL or os.getenv("SUPPLY_API_URL", "http://localhost:8001/api/v1/supply/application")

    # 校验必填字段
    required_fields = ["applicant", "department", "items"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return f"申请数据缺少必填字段: {', '.join(missing)}"

    if not isinstance(data.get("items"), list) or len(data["items"]) == 0:
        return "申请数据中 items 字段不能为空"

    # 调用第三方 REST API
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                api_url,
                json=data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                return _format_success_result(result, data)
            elif response.status_code == 201:
                result = response.json()
                return _format_success_result(result, data)
            else:
                error_detail = response.text[:300]
                return (
                    f"提交失败: HTTP {response.status_code}\n"
                    f"错误详情: {error_detail}\n\n"
                    f"申请数据已保留，请稍后重试或联系管理员。"
                )

    except httpx.ConnectError:
        return (
            f"无法连接到第三方系统 ({api_url})，申请数据已生成但未提交。\n\n"
            f"申请摘要:\n{_format_application_summary(data)}\n\n"
            f"请确认第三方系统是否在运行，或手动提交。"
        )
    except httpx.TimeoutException:
        return (
            f"第三方系统响应超时，申请可能已提交但未收到确认。\n"
            f"请联系管理员确认申请状态。"
        )
    except Exception as e:
        return (
            f"提交过程中发生异常: {str(e)}\n\n"
            f"申请数据已生成，请重试。\n"
            f"申请摘要:\n{_format_application_summary(data)}"
        )


def _extract_json(text: str) -> Optional[dict]:
    """从文本中提取 JSON 对象"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json ``` 代码块中提取
    import re
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 到最后一个 }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def _format_success_result(result: dict, data: dict) -> str:
    """格式化成功结果"""
    application_id = result.get("application_id", result.get("id", "未知"))
    status = result.get("status", "已受理")
    message = result.get("message", "")

    lines = [
        f"领用申请提交成功！",
        f"- 申请单号: {application_id}",
        f"- 状态: {status}",
    ]
    if message:
        lines.append(f"- 系统消息: {message}")

    lines.append("")
    lines.append(_format_application_summary(data))

    if result.get("estimated_delivery"):
        lines.append(f"- 预计送达: {result['estimated_delivery']}")

    if result.get("approver"):
        lines.append(f"- 审批人: {result['approver']}")

    return "\n".join(lines)


def _format_application_summary(data: dict) -> str:
    """格式化申请摘要"""
    lines = [
        f"申请人: {data.get('applicant', '未知')}",
        f"部门: {data.get('department', '未知')}",
        f"申请物品:",
    ]

    for item in data.get("items", []):
        name = item.get("name", "未知")
        qty = item.get("quantity", 0)
        unit = item.get("unit", "个")
        lines.append(f"  - {name}: {qty}{unit}")

    if data.get("reason"):
        lines.append(f"申请原因: {data['reason']}")

    return "\n".join(lines)
