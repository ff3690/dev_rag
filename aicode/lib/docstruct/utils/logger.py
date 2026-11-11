# -*- coding: utf-8 -*-
"""
logger.py - docstruct 日志工具
================================================

统一使用 Python 标准库 logging，日志名固定为 "docstruct"。

用法（在本库任何模块中）：
    from docstruct.utils.logger import logger
    logger.info("开始解析 ...")

级别约定：
    DEBUG   - 元素判定细节（每个元素是什么、为什么这么判定）
    INFO    - 每页元素统计、策略选择、关键里程碑
    WARNING - 降级与质量问题（对应 ParseIssue 的 warning）
    ERROR   - 异常（对应 ParseIssue 的 error）

安全原则：日志只记录元素类型/数量/坐标/耗时，绝不输出文档正文内容。
（与现有 backend/rag/text_splitter.py 的 print 风格保持一致）
"""

import logging

# 日志格式：时间 | 级别 | 模块 | 消息
# %(name)s 显示 logger 名（docstruct.layout.detector 等），便于定位来源
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# getLogger 传入 __name__ 会让日志带模块路径（如 docstruct.layout.detector）
# 这里固定用根名 "docstruct"，子模块通过 logging.getLogger("docstruct.xxx") 继承
logger = logging.getLogger("docstruct")

# 防止重复添加 handler（例如在 Jupyter / 多次 import 时）
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(_handler)

# 默认级别 INFO；可通过外部 logging.basicConfig(level=...) 覆盖
logger.setLevel(logging.INFO)

# 不向上传递到根 logger，避免与 backend 的日志/print 相互干扰
logger.propagate = False
