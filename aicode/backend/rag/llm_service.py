"""
LLM 服务模块 - LLM Service Module

基于 DeepSeek API 的大语言模型服务，用于 RAG 系统的回答生成。
封装 API 调用、Prompt 构建、错误处理和降级策略。

模块职责：
1. LLMService: 核心服务，封装 DeepSeek API 调用
2. PromptBuilder: Prompt 构建器，格式化上下文和问题
3. ErrorHandler: 错误处理器，管理 API 异常
4. FallbackStrategy: 降级策略，保证服务可用性

架构设计：
- 采用策略模式支持未来切换其他 LLM 供应商
- 支持同步和异步两种调用方式
- 实现分级降级：重试 → 备用模型 → 模板回答
"""

import json
import time
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import httpx

from backend.config.settings import settings
from backend.rag.retriever import RetrievedDocument


# ==================== Prompt 模板 ====================

RAG_SYSTEM_PROMPT = """你是一个专业的企业知识库助手。请严格遵循以下规则：

1. **仅基于提供的文档内容回答问题**，不要编造信息
2. **如果文档中没有相关内容**，请明确说明"抱歉，在当前知识库中未找到相关信息"
3. **引用来源**：在回答中标注信息来源的文档ID
4. **结构化回答**：使用列表、表格等形式使回答更清晰
5. **语言风格**：专业、简洁、准确"""

RAG_USER_PROMPT_TEMPLATE = """请基于以下参考文档内容回答用户问题。

## 参考文档
{context}

## 用户问题
{question}

## 要求
- 如果参考文档中有相关信息，请基于此回答
- 如果参考文档中没有相关信息，请明确说明
- 回答时可适当扩展相关知识，但需区分"文档内容"和"通用知识"
- 在回答末尾添加来源引用，格式为 [来源: 文档ID/文档名/段落]"""


# ==================== 数据类 ====================

class LLMErrorType(str, Enum):
    """LLM 错误类型枚举"""
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


@dataclass
class LLMResponse:
    """
    LLM 响应数据类

    封装 LLM API 返回的结果信息。

    属性说明：
    - answer: 生成的回答文本
    - usage: Token 使用量统计
    - sources: 引用的来源文档列表
    - is_fallback: 是否为降级回答
    - model: 使用的模型名称
    - latency_ms: 响应延迟（毫秒）
    """
    answer: str
    usage: dict = None
    sources: List[str] = None
    is_fallback: bool = False
    model: str = ""
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.usage is None:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if self.sources is None:
            self.sources = []


@dataclass
class LLMError:
    """
    LLM 错误数据类

    封装 API 错误信息，用于错误处理和日志记录。

    属性说明：
    - error_type: 错误类型枚举
    - message: 错误描述信息
    - status_code: HTTP 状态码（如适用）
    - retry_after: 建议重试时间（秒）
    - is_retryable: 是否可重试
    """
    error_type: LLMErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: int = 0
    is_retryable: bool = False


# ==================== Prompt 构建器 ====================

class PromptBuilder:
    """
    Prompt 构建器类

    负责将检索到的文档和用户问题格式化为 LLM 可处理的 Prompt。

    功能：
    1. 格式化上下文文档
    2. 构建系统提示和用户提示
    3. 控制上下文长度（避免超出 Token 限制）
    4. 添加来源引用标记

    使用示例：
        builder = PromptBuilder()
        messages = builder.build_messages(question, context_docs)
    """

    # Token 估算因子（中文约 1.5 token/字）
    CHARS_PER_TOKEN = 1.5

    def __init__(self, max_context_tokens: int = 4000):
        """
        初始化 Prompt 构建器

        参数:
            max_context_tokens: 上下文最大 Token 数（默认 4000）
                                用于控制发送给 LLM 的文本总量
                                这里是"按字符上限自动裁剪"，不是"固定送 N 篇"——如果每篇文档较短（如各 300 字符），可能 5 篇全部送入；如果文档较长，可能只送 2~3 篇。想精确控制"固定送 N 篇"需要改 _build_context 的逻辑（例如直接 docs[:N]）；想提高上限只需增大 max_context_tokens 或在 LLMService.__init__ 里给 PromptBuilder 传参。
        """
        self.max_context_tokens = max_context_tokens

    def build_messages(
        self,
        question: str,
        context_docs: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> List[dict]:
        """
        构建发送给 LLM 的消息列表

        参数:
            question: 用户问题
            context_docs: 检索到的上下文文档
            system_prompt: 自定义系统提示（可选）

        返回:
            List[dict]: 消息列表，格式为 [{"role": "system", "content": "..."}]
        """
        # 构建上下文
        context = self._build_context(context_docs)

        # 构建用户消息
        user_content = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # 返回消息列表
        return [
            {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """
        将检索文档格式化为上下文文本

        按相关性排序并添加文档标识，控制总长度。

        参数:
            docs: 检索到的文档列表

        返回:
            str: 格式化后的上下文字符串
        """
        # 按分数降序排序
        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

        context_parts = []
        total_chars = 0

        # 换算成字符上限 即 4000 × 1.5 × 0.8 = 4800 字符（CHARS_PER_TOKEN = 1.5，中文约 1.5 字符/token）。
        #   逐篇累加直到超限 — llm_service.py:204-231：
        #   在 _build_context 循环中按分数从高到低累加文档字符数，
        #   达到 【max_chars】 后，后续文档被截断保留（剩余空间 ≥200 字符时）或整体跳过。循环结束的 included_docs 就是实际送入 LLM 的文档数。
        max_chars = int(self.max_context_tokens * self.CHARS_PER_TOKEN * 0.8)
        
        included_docs = 0       # 实际送入 LLM 的文档数
        skipped_docs = 0         # 因长度超限被跳过的文档数
        truncated_docs = 0       # 因长度超限被截断保留的文档数

        for i, doc in enumerate(sorted_docs, 1):
            # 估算当前文档的字符数
            doc_chars = len(doc.content)
            if total_chars + doc_chars > max_chars:
                # 修复：超长文档不再整体跳过，而是截断保留有效内容。
                # 原因：图谱增强模式（graphrag-query）会把"图谱上下文 + 全部文档片段"
                # 拼成单个超长字符串；若这里整体跳过，上下文会变成空（"（无相关文档）"），
                # LLM 依据系统提示词"文档中没有相关内容时需明确说明"而误输出
                # "抱歉，在当前知识库中未找到相关信息" 的 fallback，直接摧毁生成质量。
                remain = max_chars - total_chars
                if remain >= 200:  # 剩余空间足够（>=200 字符）才截断保留，避免只留无意义残片
                    doc_id = doc.metadata.get("document_id", f"未知-{i}") if doc.metadata else f"未知-{i}"
                    context_parts.append(
                        f"### 文档{i} (ID: {doc_id})（内容过长，已截断）\n{doc.content[:remain]}"
                    )
                    total_chars += remain
                    included_docs += 1
                    truncated_docs += 1
                else:
                    skipped_docs += 1
                continue

            # 添加文档标识和内容
            doc_id = doc.metadata.get("document_id", f"未知-{i}") if doc.metadata else f"未知-{i}"
            context_parts.append(f"### 文档{i} (ID: {doc_id})\n{doc.content}")
            total_chars += doc_chars
            included_docs += 1

        context_text = "\n\n".join(context_parts) if context_parts else "（无相关文档）"

        # 日志：送入 LLM 的文档统计与 token 估算
        # 中文约 1.5 字符/token，英文约 4 字符/token，这里取折中值
        est_tokens = int(total_chars / self.CHARS_PER_TOKEN) if self.CHARS_PER_TOKEN > 0 else 0
        print(f"[LLM-Context] === 送入 LLM 的上下文统计 ===")
        print(f"[LLM-Context]   输入文档总数: {len(docs)}")
        print(f"[LLM-Context]   实际送入 LLM: {included_docs} 个文档")
        print(f"[LLM-Context]   因长度超限跳过: {skipped_docs} 个文档")
        print(f"[LLM-Context]   因长度超限截断保留: {truncated_docs} 个文档")
        print(f"[LLM-Context]   上下文总字符数: {total_chars}")
        print(f"[LLM-Context]   上下文预估 token: ~{est_tokens}")
        print(f"[LLM-Context]   max_context_tokens 配置: {self.max_context_tokens}")
        print(f"[LLM-Context]   每个文档分数: {[(i, round(d.score, 3)) for i, d in enumerate(sorted_docs[:included_docs])]}")
        print(f"[LLM-Context] ==================================")

        return context_text


# ==================== 错误处理器 ====================

class ErrorHandler:
    """
    LLM 错误处理器类

    负责捕获和分类 API 错误，提供友好的错误信息和重试建议。

    功能：
    1. 错误分类：将异常映射为标准错误类型
    2. 友好提示：生成用户可理解的错误消息
    3. 重试建议：根据错误类型给出重试策略
    4. 日志记录：记录错误详情用于调试

    使用示例：
        handler = ErrorHandler()
        error = handler.handle_exception(exc)
        if error.is_retryable:
            time.sleep(error.retry_after)
    """

    # HTTP 状态码映射
    STATUS_CODE_MAP = {
        400: "请求参数错误",
        401: "API Key 无效或已过期",
        403: "API Key 权限不足",
        404: "API 端点不存在",
        429: "请求频率超限，请稍后重试",
        500: "服务器内部错误",
        502: "网关错误",
        503: "服务暂时不可用",
        504: "网关超时"
    }

    def handle_exception(self, exception: Exception) -> LLMError:
        """
        处理异常并返回标准错误对象

        参数:
            exception: 捕获的异常对象

        返回:
            LLMError: 标准错误对象
        """
        # 处理 httpx 超时
        if isinstance(exception, httpx.TimeoutException):
            return LLMError(
                error_type=LLMErrorType.TIMEOUT,
                message="API 请求超时，请稍后重试",
                retry_after=30,
                is_retryable=True
            )

        # 处理 httpx 连接错误
        if isinstance(exception, httpx.ConnectError):
            return LLMError(
                error_type=LLMErrorType.CONNECTION,
                message="无法连接到 API 服务，请检查网络连接",
                retry_after=10,
                is_retryable=True
            )

        # 处理 httpx HTTP 错误
        if isinstance(exception, httpx.HTTPStatusError):
            return self._handle_http_error(exception)

        # 处理其他未知错误
        return LLMError(
            error_type=LLMErrorType.UNKNOWN,
            message=f"未知错误: {str(exception)}",
            retry_after=60,
            is_retryable=False
        )

    def _handle_http_error(self, exception: httpx.HTTPStatusError) -> LLMError:
        """
        处理 HTTP 状态码错误

        参数:
            exception: HTTP 状态错误异常

        返回:
            LLMError: 标准错误对象
        """
        status_code = exception.response.status_code

        # 4xx 客户端错误
        if status_code == 401:
            return LLMError(
                error_type=LLMErrorType.AUTHENTICATION,
                message="API Key 无效或已过期，请检查配置",
                status_code=status_code,
                is_retryable=False
            )
        elif status_code == 429:
            # 解析重试时间
            retry_after = self._parse_retry_after(exception.response)
            return LLMError(
                error_type=LLMErrorType.RATE_LIMIT,
                message="请求频率超限，请稍后重试",
                status_code=status_code,
                retry_after=retry_after,
                is_retryable=True
            )
        elif status_code < 500:
            return LLMError(
                error_type=LLMErrorType.API_ERROR,
                message=self.STATUS_CODE_MAP.get(status_code, "请求参数错误"),
                status_code=status_code,
                is_retryable=False
            )

        # 5xx 服务器错误
        return LLMError(
            error_type=LLMErrorType.API_ERROR,
            message=self.STATUS_CODE_MAP.get(status_code, "服务器错误"),
            status_code=status_code,
            retry_after=10,
            is_retryable=True
        )

    def _parse_retry_after(self, response: httpx.Response) -> int:
        """
        解析响应头中的 Retry-After 字段

        参数:
            response: HTTP 响应对象

        返回:
            int: 重试等待时间（秒）
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                pass
        return 60  # 默认 60 秒


# ==================== LLM 服务核心类 ====================

class LLMService:
    """
    基于 DeepSeek API 的 LLM 服务

    核心服务类，封装与 DeepSeek API 的所有交互逻辑。
    支持同步调用、异步调用和流式输出。

    功能：
    1. 同步生成：generate() 方法
    2. 异步生成：agenerate() 方法
    3. 流式生成：agenerate_stream() 方法
    4. 降级处理：API 失败时返回模板化回答

    使用示例：
        service = LLMService()
        
        # 同步调用
        response = service.generate("如何配置系统？", docs)
        print(response.answer)
        
        # 流式调用
        async for chunk in service.agenerate_stream("问题", docs):
            print(chunk, end="")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """
        初始化 LLM 服务

        从 settings 读取默认配置，并允许通过实例参数覆盖，从而实现不同链路
        （如知识图谱构建 vs 检索生成）分别使用不同的大模型。

        参数：
            api_key: 自定义 API Key，为空则使用 settings.DEEPSEEK_API_KEY
            base_url: 自定义 API 地址，为空则使用 settings.DEEPSEEK_API_BASE
            model: 自定义模型名，为空则使用 settings.DEEPSEEK_MODEL
            temperature: 自定义温度，为空则使用 settings.DEEPSEEK_TEMPERATURE
            max_tokens: 自定义最大 token 数，为空则使用 settings.DEEPSEEK_MAX_TOKENS
            timeout: 自定义请求超时（秒），为空则使用 settings.DEEPSEEK_TIMEOUT
        """
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.base_url = base_url or settings.DEEPSEEK_API_BASE
        self.model = model or settings.DEEPSEEK_MODEL
        self.temperature = temperature if temperature is not None else settings.DEEPSEEK_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else settings.DEEPSEEK_MAX_TOKENS
        self.timeout = timeout if timeout else settings.DEEPSEEK_TIMEOUT
        # 是否在流式请求中附带 stream_options 以获取 token 用量（默认开启）
        self.stream_include_usage = getattr(settings, "DEEPSEEK_STREAM_INCLUDE_USAGE", True)

        # 初始化辅助组件
        self.prompt_builder = PromptBuilder()
        self.error_handler = ErrorHandler()

        # HTTP 客户端（延迟初始化）
        self._client = None
        self._async_client = None
        # 记录最近一次流式调用的 token 用量（流式 API 在最后一个 chunk 返回）
        self._last_usage = None

    @property
    def is_available(self) -> bool:
        """
        检查服务是否可用

        返回:
            bool: 是否已配置 API Key
        """
        return bool(self.api_key)

    def _get_client(self) -> httpx.Client:
        """
        获取同步 HTTP 客户端

        采用惰性初始化，首次调用时创建。

        返回:
            httpx.Client: 同步 HTTP 客户端
        """
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """
        获取异步 HTTP 客户端

        采用惰性初始化，首次调用时创建。

        返回:
            httpx.AsyncClient: 异步 HTTP 客户端
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._async_client

    def generate(
        self,
        question: str,
        context_docs: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        同步生成回答

        基于用户问题和检索到的文档上下文，调用 LLM 生成回答。

        参数:
            question: 用户问题
            context_docs: 检索到的上下文文档
            system_prompt: 自定义系统提示（可选）

        返回:
            LLMResponse: 包含回答、Token 使用量、来源引用

        降级策略：
        - API 不可用时返回模板化回答
        """
        # 检查 API Key 配置；未配置则不调用大模型，直接返回向量检索内容
        if not self.is_available:
            self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return self._fallback_response(question, context_docs, "API Key 未配置")

        # 构建 Prompt
        messages = self.prompt_builder.build_messages(
            question, context_docs, system_prompt
        )
        print("=============================")
        print("提示词： ", messages)
        print("=============================")
        # 调用 API（带重试）
        start_time = time.time()
        last_error = None

        for attempt in range(2):  # 最多重试 2 次
            try:
                response_data = self._call_api(messages)
                latency = (time.time() - start_time) * 1000
                return self._parse_response(response_data, context_docs, latency)
            except Exception as e:
                last_error = e
                if attempt == 0:
                    error = self.error_handler.handle_exception(e)
                    if error.is_retryable:
                        time.sleep(1)  # 重试等待

        # 所有重试失败，返回降级回答（直接返回向量检索内容）
        latency = (time.time() - start_time) * 1000
        error_msg = self.error_handler.handle_exception(last_error).message if last_error else "未知错误"
        fallback = self._fallback_response(question, context_docs, error_msg, latency)
        self._last_usage = fallback.usage
        return fallback

    async def agenerate(
        self,
        question: str,
        context_docs: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        异步生成回答

        参数:
            question: 用户问题
            context_docs: 检索到的上下文文档
            system_prompt: 自定义系统提示（可选）

        返回:
            LLMResponse: 包含回答、Token 使用量、来源引用
        """
        if not self.is_available:
            self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return self._fallback_response(question, context_docs, "API Key 未配置")

        # 构建 Prompt
        messages = self.prompt_builder.build_messages(
            question, context_docs, system_prompt
        )
        print("=============================")
        print("提示词： ", messages)
        print("=============================")
        # 调用 API（带重试）
        start_time = time.time()
        try:
            response_data = await self._acall_api(messages)
            latency = (time.time() - start_time) * 1000
            return self._parse_response(response_data, context_docs, latency)
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            error_msg = self.error_handler.handle_exception(e).message
            fallback = self._fallback_response(question, context_docs, error_msg, latency)
            self._last_usage = fallback.usage
            return fallback

    async def agenerate_stream(
        self,
        question: str,
        context_docs: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答

        逐步返回生成的文本片段，用于 SSE 流式响应。

        参数:
            question: 用户问题
            context_docs: 检索到的上下文文档
            system_prompt: 自定义系统提示（可选）

        生成:
            str: 回答文本片段

        使用示例：
            async for chunk in llm_service.agenerate_stream("问题", docs):
                yield f"data: {chunk}\n\n"
        """
        if not self.is_available:
            # API Key 未配置：不调用大模型，直接返回向量检索内容
            self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            fallback = self._fallback_response(question, context_docs, "API Key 未配置")
            yield fallback.answer
            return

        # 构建 Prompt
        messages = self.prompt_builder.build_messages(
            question, context_docs, system_prompt
        )
        print("=============================")
        print("提示词： ", messages)
        print("=============================")
        # 重置本次调用的 token 用量记录
        self._last_usage = None
        try:
            async for chunk in self._acall_api_stream(messages):
                yield chunk
        except Exception as e:
            # API 调用失败（API Key 无效 / 无网络 / 超时等）：降级直接返回向量检索内容
            print(f"[LLM] === 流式生成异常，降级到检索内容 ===")
            print(f"[LLM]   异常类型: {type(e).__name__}")
            print(f"[LLM]   异常信息: {e}")
            import traceback
            traceback.print_exc()
            error_msg = self.error_handler.handle_exception(e).message
            print(f"[LLM]   降级原因: {error_msg}")
            fallback = self._fallback_response(question, context_docs, error_msg)
            self._last_usage = fallback.usage
            yield fallback.answer

    def _call_api(self, messages: List[dict]) -> dict:
        """
        同步调用 API

        参数:
            messages: 消息列表

        返回:
            dict: API 响应数据
        """
        # [LLM 调用日志] 调用大模型之前
        print("========== [LLM] 调用大模型前 ==========")
        print(f"[LLM] model: {self.model}")
        print(f"[LLM] 消息条数: {len(messages)}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"[LLM] 消息[{i}] role={role}: {str(content)[:500]}")
        print("========================================")

        client = self._get_client()
        response = client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        response.raise_for_status()
        result = response.json()

        # [LLM 调用日志] 大模型输出结果之后
        print("========== [LLM] 大模型输出结果 ==========")
        try:
            output_content = result["choices"][0]["message"]["content"]
            print(f"[LLM] 输出内容: {str(output_content)[:1000]}")
        except (KeyError, IndexError):
            print(f"[LLM] 输出结构异常，原始响应: {str(result)[:500]}")
        print(f"[LLM] token 用量: {result.get('usage')}")
        print("=========================================")
        return result

    async def _acall_api(self, messages: List[dict]) -> dict:
        """
        异步调用 API

        参数:
            messages: 消息列表

        返回:
            dict: API 响应数据
        """
        client = self._get_async_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        response.raise_for_status()
        return response.json()

    async def _acall_api_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        异步流式调用 API

        参数:
            messages: 消息列表

        生成:
            str: 回答文本片段
        """
        client = self._get_async_client()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        # 配置开关：部分模型/网关不支持 stream_options 参数（可能返回 400），
        # 可通过 settings.DEEPSEEK_STREAM_INCLUDE_USAGE 关闭，避免报错。
        if self.stream_include_usage:
            payload["stream_options"] = {"include_usage": True}

        print("========== [诊断] 流式请求参数 ==========")
        print("base_url:", self.base_url)
        print("model:", self.model)
        print("stream_include_usage:", self.stream_include_usage)
        print("stream_options 是否加入 payload:", "stream_options" in payload)
        print("请求URL:", f"{self.base_url}/chat/completions")
        print("messages 条数:", len(messages))
        print("==========================================")

        _conn_t0 = time.time()
        try:
            print("[诊断] 开始建立 HTTP 连接...")
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                _conn_ms = (time.time() - _conn_t0) * 1000
                print(f"[诊断] HTTP 连接建立成功 耗时: {_conn_ms:.0f}ms")
                print("========== [诊断] HTTP 状态码 ==========")
                print("status_code:", response.status_code)
                print("==========================================")
                # 关键：HTTP 错误（如 403 API Key 无效、429 限流、5xx）必须在此抛出异常，
                # 否则流式响应会静默吞掉错误，导致 agenerate_stream 不会进入降级分支。
                # 只有正常状态码（如 200）才继续解析流式内容。
                response.raise_for_status()
                raw_line_count = 0
                async for line in response.aiter_lines():
                    raw_line_count += 1
                    #print(f"[诊断] 原始行 {raw_line_count}: {repr(line)}")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            print("[诊断] 收到 [DONE] 结束标记")
                            break
                        try:
                            data = json.loads(data_str)
                            # DeepSeek/OpenAI 在最后一个非 [DONE] 的 chunk 中返回 usage，需保存。
                            # 该 chunk 的 choices 通常为空数组，即使 usage 字段格式不同，
                            # 也只影响 token 统计，不阻塞回答输出。
                            usage = data.get("usage")
                            #print(f"[诊断] 本 chunk 是否含 usage: {usage is not None}")
                            if usage:
                                self._last_usage = usage
                                print(f"[诊断] 已保存 _last_usage: {usage}")
                            delta = data["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            print(f"[诊断] 解析 chunk 异常: {type(e).__name__}: {e} | 内容: {data_str[:200]}")
                            continue
                print(f"[诊断] 流结束，共收到 {raw_line_count} 行，_last_usage = {self._last_usage}")
        except httpx.ConnectError as e:
            print(f"[诊断] === 连接失败（ConnectError）===")
            print(f"[诊断]   异常: {e}")
            raise
        except httpx.TimeoutException as e:
            print(f"[诊断] === 连接超时（TimeoutException）===")
            print(f"[诊断]   超时类型: {type(e).__name__}")
            print(f"[诊断]   配置超时: {self.timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            print(f"[诊断] === HTTP 状态码异常 ===")
            print(f"[诊断]   状态码: {e.response.status_code}")
            print(f"[诊断]   响应体: {e.response.text[:500]}")
            raise
        except Exception as e:
            print(f"[诊断] === 其他异常 ===")
            print(f"[诊断]   类型: {type(e).__name__}")
            print(f"[诊断]   信息: {e}")
            raise

    def _parse_response(
        self,
        response_data: dict,
        context_docs: List[RetrievedDocument],
        latency_ms: float
    ) -> LLMResponse:
        """
        解析 API 响应

        参数:
            response_data: API 原始响应
            context_docs: 检索到的上下文文档
            latency_ms: 响应延迟

        返回:
            LLMResponse: 解析后的响应对象
        """
        # 提取回答文本
        answer = response_data["choices"][0]["message"]["content"]

        # 提取 Token 使用量
        usage = response_data.get("usage", {})

        # 提取来源信息
        sources = [doc.metadata.get("document_id", "") for doc in context_docs if doc.metadata]

        return LLMResponse(
            answer=answer,
            usage=usage,
            sources=sources,
            is_fallback=False,
            model=response_data.get("model", self.model),
            latency_ms=latency_ms
        )

    def _fallback_response(
        self,
        question: str,
        context_docs: List[RetrievedDocument],
        reason: str = "",
        latency_ms: float = 0.0
    ) -> LLMResponse:
        """
        生成降级响应

        当 LLM API 不可用时（API Key 未配置 / API Key 无效 / 无网络 / 超时等），
        不再调用大模型，直接基于向量检索到的文档内容作为回答。

        参数:
            question: 用户问题
            context_docs: 检索到的上下文文档
            reason: 降级原因
            latency_ms: 响应延迟

        返回:
            LLMResponse: 降级响应（is_fallback=True）
        """
        # 无论是否降级，都保留完整的 token_usage 结构（全 0），
        # 便于历史记录保存时展示统一的 Token 统计。
        zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if context_docs:
            # 直接返回向量检索到的文档内容（不截断，保留完整段落）
            answer_parts = [
                f"（当前大模型服务不可用，已直接返回知识库检索内容。原因：{reason}）\n",
                f"针对您的问题“{question}”，在知识库中检索到以下相关文档内容：\n"
            ]
            for i, doc in enumerate(context_docs, 1):
                doc_id = doc.metadata.get("document_id", "未知") if doc.metadata else "未知"
                content = doc.content if doc.content else "（该文档无可用内容）"
                answer_parts.append(f"{i}. 【{doc_id}】\n{content}\n")
            answer = "\n".join(answer_parts)
        else:
            answer = f"抱歉，当前大模型服务不可用且未检索到相关文档。（原因：{reason}）"

        return LLMResponse(
            answer=answer,
            usage=zero_usage,
            is_fallback=True,
            model="fallback-template",
            latency_ms=latency_ms
        )


# ==================== 全局单例 ====================

# 应用启动时创建全局实例
llm_service = LLMService()
