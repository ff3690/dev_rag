# -*- coding: utf-8 -*-
"""
report.py - 解析问题码表与报告辅助
================================================

集中管理所有 ParseIssue 的 code 常量，避免散落在各模块里手写字符串
导致"同一个问题有不同写法"。

码表约定：
    E_ 前缀 = error（错误）：内容可能缺失
    W_ 前缀 = warning（警告）：内容不完整但已兜底
    I_ 前缀 = info（提示）：仅提示信息

用法示例（在 Recognizer 中）：
    from docstruct.report import W_TABLE_TRUNCATED
    report.add(ParseIssue(level="warning", code=W_TABLE_TRUNCATED, ...))
"""

# ---------------------- error（错误） ----------------------

# 表格提取失败（如非线框表 / 解析异常）
E_TABLE_EXTRACT_FAILED = "E_TABLE_EXTRACT_FAILED"

# 结构组装失败：整篇文档找不到任何标题块（无法建树，单层兜底）
E_ASSEMBLE_NO_HEADING = "E_ASSEMBLE_NO_HEADING"

# ---------------------- warning（警告） ----------------------

# 表格超长，触发"保留表头 + 截断行"降级（仍有内容缺失）
W_TABLE_TRUNCATED = "W_TABLE_TRUNCATED"

# 表格超超长，触发"表头+行分组"降级（表格被拆成多块）
W_TABLE_ROW_SPLIT = "W_TABLE_ROW_SPLIT"

# 流程图识别置信度低于阈值，结果可能不完整
W_FLOW_LOW_CONFIDENCE = "W_FLOW_LOW_CONFIDENCE"

# LLM 流程图识别通道失败（回退规则结果或空）
W_FLOW_LLM_FAILED = "W_FLOW_LLM_FAILED"

# 图片 OCR 失败（图片块只有位置没有文本）
W_OCR_FAILED = "W_OCR_FAILED"

# ---------------------- info（提示） ----------------------

# 页面无内容（空白页，正常跳过）
I_EMPTY_PAGE = "I_EMPTY_PAGE"

# 检测到页眉/页脚并已过滤
I_HEADER_FOOTER_FILTERED = "I_HEADER_FOOTER_FILTERED"
