# -*- coding: utf-8 -*-
"""
flowchart_to_text.py - 流程图混合管线（方式A 规则骨架 + 方式B LLM 语义补全）
====================================================================

管线背景
--------
RAG 场景下，纯规则（方式A）把流程图还原为节点/边/路径文本，存在 4 类硬伤：
  1. 判定条件标签破碎（[重]、[面]、[否]）→ 分支语义丢失
  2. 同节点在一条路径里重复串行
  3. 坐标占位节点（形如“节点(x,y)”）污染文本
  4. 单节点文本仅 3~15 字，直接嵌入向量空间信息量过低

本模块实现“方式A 与 方式B 的结合”：

  方式A（规则）           FlowchartRecognizer
        │                   产出骨架（nodes / edges / path_text / mermaid）
        ▼
  方式B（LLM）            FlowchartSemanticEnhancer
        │                   把骨架作为【事实锚定清单】注入提示词，
        │                   LLM 只做“语义补全”（还原分组、补全条件、
        │                   按主流程/分支组织自然语言段落），而非从零转写
        ▼
  RAG 输入                按主流程/分支切块的文本片段 + 元数据

防幻觉 / 容错机制
----------------
- 骨架锚定：提示词强制“只能使用骨架中的节点与连线，禁止虚构”
- 覆盖率校验：LLM 输出必须命中 >= min_coverage 的骨架节点，否则重试一次
  （加强指令），仍不达标则降级
- 降级兜底：LLM 不可用 / 异常 / 覆盖率不达标时，退化为方式A的 path_text
  切块，管线始终有输出

LLM 调用通过构造参数注入（llm_call: Callable[[List[dict]], str]），
docstruct 保持零外部依赖；backend 侧可传入 DeepSeek / OpenAI 兼容适配器。

使用示例
--------
    from docstruct.extract.flowchart_to_text import FlowchartSemanticEnhancer

    enhancer = FlowchartSemanticEnhancer(llm_call=deepseek_adapter)
    result = enhancer.enhance(flow_block)          # flow_block 为 flowchart Block
    for chunk in result.chunks:
        print(chunk["content"])
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from docstruct.schemas import Block
from docstruct.utils.logger import logger


# ==================== 提示词模板 ====================

SYSTEM_PROMPT = """你是文档解析助手，严格依据图片/骨架里流程图的节点、箭头关系还原流程。
1. 不新增、不编造任何图中没有出现的业务名词；
2. 还原流程先后顺序、闭环逻辑；
3. 输出 markdown 格式自然语言流程描述，用于 RAG 向量检索；
4. 禁止自行增加图中不存在的步骤；
5. 边上的判定标签若为不完整片段（如“重”“面”“否”），仅能依据上下文补全并加【】标注，不得杜撰判定条件；
6. 忽略无文本的占位节点（形如“节点(x,y)”）。"""

# 上述 SYSTEM_PROMPT 的 4 条核心禁令（供测试断言精确比对，禁止随意改动措辞）
PROMPT_PROHIBITIONS = [
    "不新增、不编造任何图中没有出现的业务名词",
    "还原流程先后顺序、闭环逻辑",
    "输出 markdown 格式自然语言流程描述",
    "禁止自行增加图中不存在的步骤",
]

RETRY_PROMPT_SUFFIX = (
    "\n\n注意：上次输出遗漏了骨架中的关键节点。请重新组织，确保每条分支路径的"
    "中间步骤都完整出现，且不新增骨架中不存在的任何节点或关系。"
)


def _edge_to_dict(edge) -> Dict:
    """FlowEdge / dict 统一转成可 JSON 序列化的 dict。"""
    if hasattr(edge, "from_node"):
        return {"from": edge.from_node, "to": edge.to_node, "label": edge.label}
    return {"from": edge.get("from"), "to": edge.get("to"), "label": edge.get("label")}


def build_user_prompt(skeleton: Dict, retry: bool = False) -> str:
    """
    构建用户消息：骨架作为事实锚定清单。

    参数:
        skeleton: 含 nodes / edges / path_text / mermaid 的骨架字典
        retry: 是否重试（追加“不得遗漏关键节点”的强调）

    返回:
        str: 用户提示词文本
    """
    parts = [
        "以下是流程图的规则识别结果（节点与箭头关系是唯一事实来源，禁止虚构任何节点、连线或业务名词）：\n",
        f"## 节点列表\n{json.dumps(skeleton.get('nodes', []), ensure_ascii=False, indent=1)}",
        f"## 边列表（from --[label]--> to）\n"
        f"{json.dumps(skeleton.get('edges', []), ensure_ascii=False, indent=1)}",
        f"## 已生成的路径文本\n{skeleton.get('path_text', '')}",
        f"## mermaid 描述\n{skeleton.get('mermaid', '')}",
    ]
    if skeleton.get("title"):
        parts.insert(1, f"## 流程图标题\n{skeleton['title']}")
    parts.append("\n请基于以上骨架进行语义化补全，输出连贯的中文流程说明。")
    if retry:
        parts.append(RETRY_PROMPT_SUFFIX)
    return "\n".join(parts)


# ==================== 结果数据类 ====================

@dataclass
class FlowchartTextResult:
    """
    混合管线输出。

    属性说明：
    - text: 语义化补全后的完整文本（LLM 输出或降级后的 path_text）
    - chunks: 按主流程/分支切好的 RAG 块列表，每块含 content 与元数据
    - used_llm: 是否真正走了方式B（LLM 语义补全）
    - coverage: 骨架节点在输出文本中的覆盖率（0~1）
    - fallback_reason: 降级原因（used_llm=False 时非空）
    - node_texts: 用于校验的骨架节点文本（已剔除坐标占位）
    """
    text: str
    chunks: List[Dict]
    used_llm: bool
    coverage: float = 0.0
    fallback_reason: str = ""
    node_texts: List[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return not self.used_llm


# ==================== 混合管线核心类 ====================

class FlowchartSemanticEnhancer:
    """
    流程图混合管线：方式A骨架 → 方式B LLM 语义补全 → 按流程/分支切块。
    它不"识别"图片，而是对规则已提取的骨架做语义补全：
    参数:
        llm_call: LLM 调用函数，入参为 messages（OpenAI 兼容消息列表），
                  返回生成的文本。为空时直接降级为方式A切块。
        min_coverage: 骨架节点覆盖率阈值（默认 0.6），LLM 输出低于该值
                      触发重试（一次）或降级
        min_chunk_len: RAG 块最小字符数（默认 20），过短段落被合并/丢弃
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[List[Dict]], str]] = None,
        min_coverage: float = 0.6,
        min_chunk_len: int = 20,
    ):
        self.llm_call = llm_call
        self.min_coverage = min_coverage
        self.min_chunk_len = min_chunk_len

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def enhance(self, block: Block) -> FlowchartTextResult:
        """
        对 flowchart 块执行混合管线，产出 RAG 可用的切块文本。

        参数:
            block: docstruct 解析出的 flowchart 类型 Block

        返回:
            FlowchartTextResult: 补全文本 + 切块 + 覆盖率 + 降级信息
        """
        data = block.data or {}
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        path_text = data.get("path_text") or ""
        mermaid = data.get("mermaid") or ""
        title = data.get("title") or (block.section_path or "流程图")

        # 骨架清洗：剔除坐标占位节点，得到事实锚定清单
        clean_nodes = self._clean_nodes(nodes)
        skeleton = {
            "title": title,
            "nodes": clean_nodes,
            "edges": [_edge_to_dict(e) for e in edges],
            "path_text": path_text,
            "mermaid": mermaid,
        }

        # LLM 不可用 → 直接降级到方式A切块
        if self.llm_call is None:
            return self._fallback(block, skeleton, clean_nodes, "未配置 LLM 调用")

        # 方式B：骨架锚定 + 语义补全
        try:
            text = self.llm_call(self.build_prompt(skeleton))
            text = self._postprocess(text)
            coverage = self.node_coverage(text, clean_nodes)

            # 覆盖率不达标：带强调指令重试一次
            if coverage < self.min_coverage and len(clean_nodes) >= 3:
                logger.warning(
                    f"[FlowchartToText] 覆盖率 {coverage:.2f} < {self.min_coverage}，重试"
                )
                text = self.llm_call(self.build_prompt(skeleton, retry=True))
                text = self._postprocess(text)
                coverage = self.node_coverage(text, clean_nodes)

            if coverage < self.min_coverage and len(clean_nodes) >= 3:
                return self._fallback(
                    block, skeleton, clean_nodes,
                    f"LLM 输出覆盖率不足 ({coverage:.2f} < {self.min_coverage})",
                )

            chunks = self.to_chunks(text, title, block)
            return FlowchartTextResult(
                text=text,
                chunks=chunks,
                used_llm=True,
                coverage=coverage,
                node_texts=clean_nodes,
            )
        except Exception as e:
            logger.warning(f"[FlowchartToText] LLM 调用失败，降级: {e}")
            return self._fallback(block, skeleton, clean_nodes, str(e))

    # ------------------------------------------------------------------
    # 提示词构建
    # ------------------------------------------------------------------
    def build_prompt(self, skeleton: Dict, retry: bool = False) -> List[Dict]:
        """
        构建 OpenAI 兼容的消息列表。

        参数:
            skeleton: 骨架字典（nodes/edges/path_text/mermaid/title）
            retry: 是否追加“不得遗漏”的强调指令

        返回:
            List[Dict]: [{"role": "system", ...}, {"role": "user", ...}]
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(skeleton, retry=retry)},
        ]

    # ------------------------------------------------------------------
    # 切块
    # ------------------------------------------------------------------
    def to_chunks(
        self, text: str, title: str = "流程图", block: Optional[Block] = None
    ) -> List[Dict]:
        """
        把补全文本按“主流程 / 各分支”段落切块，产出 RAG 输入。

        切块策略：
        - 按空行分段（每段 = 一条完整流程路径或一个分支场景）
        - 段落之间是完整的流程语义，不做跨段拼接，避免切断分支上下文
        - 过短段落（< min_chunk_len）且存在其他段落时丢弃
        - 单段超长（> 400 字）按分号/句号粗切，避免整块过大

        参数:
            text: 语义化补全后的文本（或降级的 path_text）
            title: 流程图标题
            block: 源块（可选，用于元数据）

        返回:
            List[Dict]: 每项含 chunk_id / content / metadata
        """
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paras:
            paras = [p.strip() for p in text.splitlines() if p.strip()]

        chunks: List[Dict] = []
        page_no = block.page_no if block is not None else 0
        for i, para in enumerate(paras):
            # 单段超长粗切
            if len(para) > 400:
                sub = re.split(r"(?<=[。；;])", para)
                merged, cur = [], ""
                for s in sub:
                    s = s.strip()
                    if not s:
                        continue
                    cur = cur + s if cur else s
                    if len(cur) >= self.min_chunk_len:
                        merged.append(cur)
                        cur = ""
                if cur:
                    merged.append(cur)
                for j, s in enumerate(merged):
                    chunks.append(self._make_chunk(s, title, page_no, i * 100 + j))
                continue
            if len(para) < self.min_chunk_len and len(paras) > 1:
                continue
            chunks.append(self._make_chunk(para, title, page_no, i))

        if not chunks and text.strip():
            chunks.append(self._make_chunk(text.strip(), title, page_no, 0))
        return chunks

    def _make_chunk(self, content: str, title: str, page_no: int, seq: int) -> Dict:
        """构建单个 RAG 块：加标题前缀 + 元数据。"""
        return {
            "chunk_id": f"flowchart-{page_no}-{seq}",
            "content": f"【流程图】{title}\n{content}",
            "metadata": {
                "type": "flowchart_chunk",
                "title": title,
                "page_no": page_no,
                "source": "hybrid_A_rules+B_llm",
            },
        }

    # ------------------------------------------------------------------
    # 覆盖率校验（防幻觉锚点）
    # ------------------------------------------------------------------
    def node_coverage(self, text: str, node_texts: List[str]) -> float:
        """
        骨架节点在输出文本中的覆盖率。

        规则：节点文本 >= 4 字才参与统计（太短的节点子串匹配噪声大）；
        命中 = 节点文本作为子串出现在输出文本中。

        参数:
            text: LLM 输出文本
            node_texts: 骨架节点文本（已剔除坐标占位）

        返回:
            float: 覆盖率 0~1（无有效节点时返回 1.0）
        """
        valid = [t for t in node_texts if len(t) >= 4]
        if not valid:
            return 1.0
        hit = sum(1 for t in valid if t in text)
        return hit / len(valid)

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    def _clean_nodes(self, nodes: List[Dict]) -> List[str]:
        """剔除坐标占位节点，返回节点文本列表。"""
        out, seen = [], set()
        for n in nodes:
            t = n.get("text", "")
            if not t or t.startswith("节点(") or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _postprocess(text: str) -> str:
        """清理 LLM 输出中的代码块围栏与多余空行。"""
        text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fallback(
        self,
        block: Block,
        skeleton: Dict,
        clean_nodes: List[str],
        reason: str,
    ) -> FlowchartTextResult:
        """降级：LLM 不可用/异常/覆盖率不足时，退回方式A的 path_text 切块。"""
        path_text = skeleton.get("path_text") or ""
        text = path_text if path_text.strip() else "\n".join(clean_nodes)
        chunks = self.to_chunks(text, skeleton.get("title", "流程图"), block)
        coverage = self.node_coverage(text, clean_nodes)
        return FlowchartTextResult(
            text=text,
            chunks=chunks,
            used_llm=False,
            coverage=coverage,
            fallback_reason=reason,
            node_texts=clean_nodes,
        )
