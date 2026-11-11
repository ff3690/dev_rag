"""
办公用品消耗预测工具 - Consumption Prediction Tool

Agent 可通过此工具分析历史消耗数据，使用 LLM 进行趋势预测，
生成当月领用建议。

典型场景:
    Agent 查到上个月消耗清单 → 调本工具分析 → 预测本月需求 → 生成领用申请
"""

import json
from typing import Optional
from backend.rag.llm_service import llm_service


TOOL_DESCRIPTION = """
工具名称: predict_consumption
功能: 基于历史消耗数据，使用大模型进行趋势分析和本月需求量预测
参数:
  - consumption_data (str, 必填): 上个月（或历史）消耗统计数据的文本描述，
    应包含物品名称、用量、用途等信息
  - department (str, 可选): 部门名称，用于细化预测
返回: 预测分析报告和本月建议领用清单
使用场景:
  - 用户问"帮我预测这个月办公用品需求"时
  - 需要根据历史消耗生成领用申请时
"""


PREDICTION_PROMPT = """你是一个办公用品消耗分析专家。请根据以下历史消耗数据，分析并预测本月的办公用品需求。

## 历史消耗数据
{consumption_data}

## 部门
{department_info}

## 分析要求
1. **趋势分析**: 分析各类物品的消耗趋势（增长/持平/下降）
2. **季节性因素**: 考虑当前是 {current_month} 月，是否有季节性需求变化
3. **安全库存**: 建议每类物品预留 10-20% 的安全余量
4. **预测清单**: 以 JSON 格式输出本月建议领用清单

## 输出格式
请先给出分析报告，然后以 JSON 格式输出领用清单：

```json
{{
  "analysis": "趋势分析文本...",
  "prediction": {{
    "month": "{current_month}",
    "items": [
      {{
        "name": "物品名称",
        "category": "分类（文具/耗材/清洁/电子/其他）",
        "last_month_usage": 100,
        "predicted_need": 110,
        "unit": "个/支/包/箱",
        "reason": "预测理由"
      }}
    ],
    "total_estimated_cost": "预估总费用说明",
    "notes": "补充说明"
  }}
}}
```
"""


def predict_consumption_tool(
    consumption_data: str,
    department: str = "技术部"
) -> str:
    """
    分析历史消耗数据并预测本月需求

    参数:
        consumption_data: 历史消耗统计信息（文本格式）
        department: 部门名称

    返回:
        str: 预测分析报告（含 JSON 格式的领用清单）
    """
    from datetime import datetime

    current_month = datetime.now().strftime("%Y年%m月")
    department_info = f"部门: {department}" if department else "未指定部门"

    prompt = PREDICTION_PROMPT.format(
        consumption_data=consumption_data,
        department_info=department_info,
        current_month=current_month
    )

    try:
        # 构建消息
        messages = [
            {"role": "user", "content": prompt}
        ]

        # 调用 LLM 进行分析预测
        # 使用 llm_service._call_api 直接调用，因为这是独立于 RAG 的 LLM 调用
        print(">>>>>> [Agent] predict_consumption_tool 调用 LLM 进行消耗预测 <<<<<<")
        response_data = llm_service._call_api(messages)
        answer = response_data["choices"][0]["message"]["content"]

        return answer

    except Exception as e:
        # LLM 调用失败时返回基础统计
        return (
            f"大模型预测分析暂时不可用（{str(e)}），以下为基础数据汇总:\n\n"
            f"{consumption_data}\n\n"
            "建议根据上月用量加 10% 余量进行领用。"
        )
