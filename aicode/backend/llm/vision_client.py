# -*- coding: utf-8 -*-
"""
vision_client.py - 多模态（视觉）LLM 客户端
=========================================

用途：
    为 docstruct 的"流程图 LLM 识别通道"(LLMFlowRecognizer) 提供
    recognize_image(image_path) -> dict 接口实现。

为什么单独建一个客户端：
    - backend/rag/llm_service.py 的 DeepSeek 客户端是纯文本模型，
      DeepSeek 官方 API 没有多模态能力，无法"看图"。
    - 本客户端采用 OpenAI 兼容的 Chat Completions + image_url(data URI)
      协议，可对接任意支持视觉的 OpenAI 兼容服务，只需改三个配置：
        * 智谱 GLM-4V / GLM-4V-Flash（免费）: https://open.bigmodel.cn/api/paas/v4
        * 阿里云百炼 Qwen-VL:              https://dashscope.aliyuncs.com/compatible-mode/v1
        * OpenAI GPT-4o:                   https://api.openai.com/v1
      ……通过 settings 中的 STRUCTURED_PARSER_FLOW_LLM_* 配置切换。

接口约定（与 docstruct.extract.llm_flow_recognizer 对齐）：
    client.recognize_image(image_path: str) -> dict
    返回 {"nodes": [...], "edges": [{"from":..,"to":..,"label":..}], "description": "..."}
    失败抛异常（调用方 LLMFlowRecognizer 捕获后回退规则通道）
"""

import base64
import io
import json
import os
import re
from typing import List, Optional

import httpx

from backend.config.settings import settings


# ==================== Prompt ====================

FLOW_IMAGE_SYSTEM_PROMPT = (
    "你是一个专业的文档流程图识别引擎。请仔细查看用户提供的图片，"
    "提取其中的流程结构，只输出合法的 JSON，不要输出任何解释文字。"
)

# 缺陷7：普通图片语义描述 prompt（图片块无 OCR 文本时调用）
IMAGE_DESCRIBE_SYSTEM_PROMPT = (
    "你是一个专业的文档图片解读引擎。请仔细查看用户提供的图片，"
    "用简洁的中文描述图片内容，只输出描述文本，不要输出任何 JSON 或解释。"
)

IMAGE_DESCRIBE_USER_PROMPT = (
    "请用 2-3 句话描述这张图片：图片的类型（如组织架构图、示意图、照片、"
    "图表、截图等）、图中出现的核心内容/文字信息、表达的主要含义。"
    "如果图片完全无法识别，直接回答\"（图片无法识别）\"。"
)

FLOW_IMAGE_USER_PROMPT = """请识别这张流程图图片，严格按如下 JSON 结构返回（不要使用 Markdown 代码块包裹，不要输出 JSON 以外的内容）：

{{
  "nodes": ["节点1的文本", "节点2的文本", ...],
  "edges": [{{"from": "起始节点文本", "to": "目标节点文本", "label": "连线上的条件说明（无文字则为空字符串）"}}],
  "description": "用一段话概括整个流程的主线逻辑"
}}

要求：
1. nodes 只包含去重后的节点文本，按流程顺序排列；
2. edges 必须完整覆盖图片中所有箭头连线关系，from/to 需与 nodes 中的文本一致；
3. 连线上有判断条件（如"是/否""符合条件"）时填入 label，否则为空字符串；
4. 若图片无法识别或不是流程图，返回 {{"nodes": [], "edges": [], "description": "无法识别"}}。"""


class VisionLLMClient:
    """
    OpenAI 兼容的多模态（视觉）LLM 客户端。

    参数（均可不传，默认读取 settings 中的 STRUCTURED_PARSER_FLOW_LLM_*）:
        api_key: 多模态服务商 API Key
        base_url: OpenAI 兼容接口地址（不含 /chat/completions）
        model: 视觉模型名
        timeout: 请求超时（秒）

    线程安全：每次 recognize_image 使用独立 httpx 连接，可复用实例。
    """

    def __init__(self,
                 api_key: str = "",
                 base_url: str = "",
                 model: str = "",
                 timeout: int = 0):
        self.api_key = api_key or settings.STRUCTURED_PARSER_FLOW_LLM_API_KEY
        self.base_url = (base_url or settings.STRUCTURED_PARSER_FLOW_LLM_BASE_URL).rstrip("/")
        self.model = model or settings.STRUCTURED_PARSER_FLOW_LLM_MODEL
        self.timeout = timeout or settings.STRUCTURED_PARSER_FLOW_LLM_TIMEOUT

        if not self.api_key:
            raise ValueError(
                "多模态 LLM 客户端未配置 API Key："
                "请在 backend/config/settings.py 或 .env 中设置 "
                "STRUCTURED_PARSER_FLOW_LLM_API_KEY"
                "（推荐智谱 GLM-4V-Flash，免费申请: https://open.bigmodel.cn/）"
            )

    # ------------------------------------------------------------------
    # 对外接口（docstruct LLMFlowRecognizer 约定）
    # ------------------------------------------------------------------
    def recognize_image(self, image_path: str) -> dict:
        """
        识别流程图图片，返回结构化结果。

        参数:
            image_path: 本地图片文件路径

        返回:
            {"nodes": [...], "edges": [{"from","to","label"}], "description": "..."}

        异常:
            ValueError / httpx.HTTPError: 任何失败直接抛出，由调用方降级处理
        """
        if not os.path.exists(image_path):
            raise ValueError(f"流程图图片不存在: {image_path}")

        # ---- 1. 读取并编码图片（超限时压缩） ----
        data_uri = self._encode_image(image_path)

        # ---- 2. 调用 Chat Completions ----
        payload = {
            "model": self.model,
            "temperature": 0.2,  # 抽取任务低温度，保证结构化稳定
            # GLM-4V-Flash 限制 max_tokens ∈ [1,1024]，超出返回 400(1210)
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": FLOW_IMAGE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": FLOW_IMAGE_USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                ]},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if resp.status_code != 200:
                raise ValueError(
                    f"多模态 LLM 接口返回 {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
            result = resp.json()

        # ---- 3. 解析并归一化返回 ----
        content = self._extract_content(result)
        obj = self._parse_json(content)
        return self._normalize(obj)

    # ------------------------------------------------------------------
    # 通用图片解读（缺陷7：普通图片块 LLM 解读通道）
    # ------------------------------------------------------------------
    def describe_image(self, image_path: str) -> str:
        """
        用一段话描述图片内容（普通图片，非流程图）。

        参数:
            image_path: 本地图片文件路径

        返回:
            str: 图片语义描述文本（用于图片块的 content 兜底，使其可被向量检索）

        异常:
            ValueError / httpx.HTTPError: 任何失败直接抛出，由调用方降级处理
        """
        if not os.path.exists(image_path):
            raise ValueError(f"图片不存在: {image_path}")

        data_uri = self._encode_image(image_path)
        return self._chat_describe(data_uri)

    # ------------------------------------------------------------------
    # Markdown 图片图意化：内存字节版描述（md 图片无需先落临时文件）
    # ------------------------------------------------------------------
    def describe_image_bytes(self, data: bytes, ext: str = "png") -> str:
        """
        用一段话描述图片字节内容（Markdown 图片内存通道）。

        参数:
            data: 图片原始字节（PNG/JPEG/WebP 等）
            ext:  图片扩展名（仅用于日志/占位，编码统一为 PNG）

        返回:
            str: 图片语义描述文本（含"（图片无法识别）"兜底）

        异常:
            ValueError / httpx.HTTPError: 任何失败直接抛出，由调用方降级处理
        """
        if not data:
            raise ValueError("图片字节为空")
        data_uri = VisionLLMClient._bytes_to_data_uri(data)
        return self._chat_describe(data_uri)

    # ------------------------------------------------------------------
    # 通用 Chat Completions 调用（描述任务共享）
    # ------------------------------------------------------------------
    def _chat_describe(self, data_uri: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": IMAGE_DESCRIBE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": IMAGE_DESCRIBE_USER_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                ]},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if resp.status_code != 200:
                raise ValueError(
                    f"多模态 LLM 接口返回 {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
            result = resp.json()

        content = self._extract_content(result)
        if not content or not content.strip():
            raise ValueError("图片描述接口返回空内容")
        # 简单清洗：去掉首尾空白与多余换行
        return content.strip()

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        读取图片为 base64 data URI；超过 4MB 或宽/高超过 2048px 时，
        用 Pillow 等比压缩，避免超出多模态模型的服务端限制。
        """
        with open(image_path, "rb") as f:
            raw = f.read()
        return VisionLLMClient._bytes_to_data_uri(raw)

    @staticmethod
    def _bytes_to_data_uri(raw: bytes) -> str:
        """
        图片字节 -> base64 data URI（含超大体积/超大尺寸压缩），
        Markdown 图片内存通道（describe_image_bytes）与文件通道共用。
        """
        img = None
        # 体积超限或超大尺寸时压缩
        if len(raw) > 4 * 1024 * 1024:
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            if img.width > 2048 or img.height > 2048:
                img.thumbnail((2048, 2048), Image.LANCZOS)

        if img is not None:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw = buf.getvalue()

        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _extract_content(result: dict) -> str:
        """从 Chat Completions 响应中取出 message.content 文本。"""
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"多模态 LLM 响应结构异常: {str(result)[:300]}")

    @staticmethod
    def _parse_json(content: str) -> dict:
        """
        把 LLM 返回的文本解析为 dict。
        兼容三种形态：纯 JSON、```json 代码块包裹、前后夹带说明文字。
        """
        if not content or not isinstance(content, str):
            raise ValueError(f"多模态 LLM 返回空内容: {content!r}")

        text = content.strip()
        # 去掉 Markdown 代码块围栏
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        # 截取第一个 { 到最后一个 }
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"多模态 LLM 返回非 JSON 内容: {content[:300]}")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            # GLM-4V 等视觉模型偶尔输出不严格 JSON（漏逗号/多余逗号），容错重试
            obj = VisionLLMClient._repair_json(text[start:end + 1], content)
        if not isinstance(obj, dict):
            raise ValueError(f"多模态 LLM 返回不是 JSON 对象: {content[:300]}")
        return obj

    @staticmethod
    def _repair_json(text: str, original: str) -> dict:
        """容错解析 LLM 返回的 JSON。

        优先使用 json_repair 库；未安装时用手写正则兜底
        （补字符串间缺逗号、去多余尾逗号）。仍失败则抛 ValueError。
        """
        try:
            from json_repair import loads as repair_loads
            obj = repair_loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            fixed = re.sub(r'"(\s+)"', '", "', text)          # "a" "b" -> "a", "b"
            fixed = re.sub(r'([\]\}])\s*(?=["\[\{])', r'\1,\2', fixed)  # } { -> }, {
            fixed = re.sub(r',\s*([\]\}])', r'\1', fixed)      # 去尾逗号 ,] -> ]
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass
        raise ValueError(f"多模态 LLM 返回 JSON 解析失败; 原文: {original[:300]}")

    @staticmethod
    def _normalize(obj: dict) -> dict:
        """校验并归一化为 {nodes, edges, description} 标准结构。"""
        nodes_raw = obj.get("nodes") if isinstance(obj, dict) else None
        edges_raw = obj.get("edges") if isinstance(obj, dict) else None
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            raise ValueError("多模态 LLM 返回缺少 nodes/edges 列表")

        nodes = [str(n).strip() for n in nodes_raw if str(n).strip()]
        edges = []
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            frm, to = e.get("from"), e.get("to")
            if not frm or not to:
                continue
            edges.append({
                "from": str(frm).strip(),
                "to": str(to).strip(),
                "label": str(e.get("label") or "").strip(),
            })

        description = str(obj.get("description") or "").strip()
        if not nodes and not edges:
            raise ValueError(f"多模态 LLM 未能识别出流程图内容: {description}")

        return {"nodes": nodes, "edges": edges, "description": description}
