# -*- coding: utf-8 -*-
"""
config.py - docstruct 库内部配置（默认值）
================================================

作用：
    ParserGate 在创建时如果没有传入 ParserConfig，则使用本文件的默认值。
    这样 docstruct 可以脱离 backend 独立运行；被 backend 集成时，
    backend 会在 DocumentLoader.load() 中把 settings 里的
    STRUCTURED_PARSER_* 配置传入并覆盖这里的默认值。

优先级（低 -> 高）：
    1. 本文件默认值（库独立运行兜底）
    2. ParserConfig 构造参数（调用方显式指定）
"""


class ParserConfig:
    """
    解析器配置。所有字段都带默认值，可按需覆盖。

    参数:
        chunk_size: 常规文本切片的 token 上限（默认 512，与现有 RAG 一致）
        chunk_overlap: 相邻切片的重叠 token 数（默认 64）
        max_table_tokens: 表格切片的 token 上限。
            表格优先整表完整切片；超过该值时触发"保留表头截断"降级。
            说明：bge 模型最大输入 512 token，但表格特殊（整表完整才可回答），
            所以表格允许超长并单独处理，这是本方案的关键决策之一。
        flow_llm_enabled: 是否启用"流程图 LLM 识别通道"（默认 False）。
            本 PDF 的流程图是矢量图，规则分析已足够；LLM 通道用于
            其他"图片型流程图"文档，属扩展能力。
        flow_llm_client: 多模态 LLM 客户端（默认 None 表示未配置）。
            需提供 recognize_image(image_path) -> dict 接口，返回
            {"nodes": [...], "edges": [...], "description": "..."}。
        flow_llm_auto_recheck: 规则通道置信度低（节点<3或边<2）时，
            是否自动调用 flow_llm_client 复核（默认 False）。
            仅当 flow_llm_enabled 且 flow_llm_client 可用时生效。
        image_llm_enabled: 是否启用"普通图片 LLM 解读通道"（默认 False）。
            图片块 OCR 无文本时，调用 vision 客户端 describe_image 生成
            图片语义描述，避免图片块在向量空间完全不可检索。
        image_llm_client: 通用视觉 LLM 客户端（默认 None）。
            需提供 describe_image(image_path) -> str 接口。
        output_dir: 结构化中间产物（md/json/图片）的落盘目录。
        parser_version: 解析器版本号。版本升级后，入库侧可据此
            提示用户重新解析入库（D8 决策：不自动重建，只提示）。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_table_tokens: int = 2000,
        flow_llm_enabled: bool = False,
        flow_llm_client=None,
        flow_llm_auto_recheck: bool = False,
        image_llm_enabled: bool = False,
        image_llm_client=None,
        output_dir: str = "artifacts/structured",
        flow_image_dir: str = "artifacts/structured/flow_images",
        parser_version: str = "1.0.0",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_tokens = max_table_tokens
        self.flow_llm_enabled = flow_llm_enabled
        self.flow_llm_client = flow_llm_client
        self.flow_llm_auto_recheck = flow_llm_auto_recheck
        self.image_llm_enabled = image_llm_enabled
        self.image_llm_client = image_llm_client
        self.output_dir = output_dir
        self.flow_image_dir = flow_image_dir
        self.parser_version = parser_version
