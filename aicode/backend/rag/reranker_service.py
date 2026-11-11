"""
Reranker 精排服务模块 - Reranker Service Module

基于 BAAI/bge-reranker-v2-m3 的 Cross-Encoder 精排服务，
对向量粗排结果进行二次重排序，提升检索精度。

核心职责：
1. 加载和管理 BGE-Reranker 模型
2. 对粗排结果进行精排重排序
3. 提供优雅降级机制（模型不可用时回退到粗排）

技术架构：
- 模型：BAAI/bge-reranker-v2-m3 (Cross-Encoder, ~560M 参数)
- 架构：XLM-RoBERTa-based，支持 100+ 语言
- 输入：Query-Document 对
- 输出：相关性分数（0-1，经 sigmoid 归一化）

CPU 优化策略（基于实测经验）：
1. 限制输入数量（INPUT_TOP_K=5，避免 13 秒级延迟）
2. 限制序列长度（MAX_LENGTH=256，而非默认 512）
3. 关闭 FP16，使用 FP32（CPU 上 FP16 无加速且可能报错）
4. 文档预截断（DOC_CHAR_LIMIT=800 字符）
5. 批处理优化（BATCH_SIZE=4）
6. 设置 PyTorch 线程数

降级策略：
- 模型加载失败 → 返回粗排结果（按原分数降序）
- 推理超时 → 返回粗排结果
- 推理异常 → 返回粗排结果
"""

import os
import time
import threading
from typing import List, Optional

from backend.config.settings import settings


# ============================================================
# FlagEmbedding 兼容性补丁
# ------------------------------------------------------------
# FlagReranker 内部（FlagEmbedding/inference/reranker/encoder_only/base.py）
# 调用 tokenizer.prepare_for_model(...)，该 API 在 transformers>=4.40 已被移除，
# 导致精排批次计算抛出 AttributeError，精排结果被 0 分污染、检索失效。
#
# 这里在方法缺失时将其补上：prepare_for_model 接收的是已编码的 token id 列表，
# 先 decode 还原为文本，再交给 __call__ 复用 padding/truncation/special-token 逻辑，
# 行为与旧版基本一致。仅在缺失时打补丁，不影响任何已存在的实现。

# 【在没有补丁时，批次计算会抛：】
#     AttributeError: 'XLMRobertaTokenizer' object has no attribute 'prepare_for_model'
# 被 except 捕获后导致精排异常，检索结果被降级或污染（表现为"检索不到内容"）。
# 
# 【如何判断当前项目是否命中】
# 重启后端看启动日志：
#     [Reranker] 已应用 FlagEmbedding prepare_for_model 兼容补丁（transformers>=4.40）
# **看到这行** → 当前环境 transformers ≥ 4.40 且 FlagEmbedding 走旧 API，补丁在兜底
# **没看到这行** → 你的 transformers 是旧版（方法本来就存在），或 FlagEmbedding 新版不再调用该方法，补丁处于休眠状态，属正常情况
# 一句话总结：它在"旧版 FlagEmbedding + 新版 transformers"这个版本错配组合下兜底；版本匹配时它只是安静地挂在基类上，不影响任何行为。
# ============================================================
def _apply_flagembedding_compat_patch():
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
            return  # 旧版 transformers 自带该方法，无需补丁

        def _prepare_for_model_compat(self, ids, pair_ids=None, **kwargs):
            text = self.decode(ids, skip_special_tokens=True)
            text_pair = self.decode(pair_ids, skip_special_tokens=True) if pair_ids is not None else None
            return self(text=text, text_pair=text_pair, **kwargs)

        PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model_compat
        print("[Reranker] 已应用 FlagEmbedding prepare_for_model 兼容补丁（transformers>=4.40）")
    except ImportError:
        # transformers 未安装：精排本身不可用，无需补丁
        pass
    except Exception as e:
        print(f"[Reranker] 应用 FlagEmbedding 兼容补丁失败: {e}")


_apply_flagembedding_compat_patch()


class RerankerService:
    """
    BGE-Reranker 精排服务类

    封装 FlagEmbedding 的 FlagReranker，提供统一的精排接口。

    设计要点：
    1. 单例模式：通过模块级 reranker_service 实例复用模型，避免重复加载
    2. 延迟初始化：模型在首次调用 rerank() 时才加载，减少启动时间
    3. 优雅降级：模型不可用时自动回退到粗排结果
    4. CPU 优化：针对 CPU 环境进行多项性能优化

    使用示例：
        service = RerankerService()
        results = service.rerank("如何配置数据库", documents)
        for r in results:
            print(f"[{r.rerank_score:.3f}] {r.content[:50]}")
    """

    def __init__(self):
        """
        初始化 Reranker 服务

        配置来源：
            settings.RERANKER_ENABLED -> 是否启用精排
            settings.RERANKER_MODEL_NAME -> "BAAI/bge-reranker-v2-m3"
            settings.RERANKER_MODEL_PATH -> "models/reranker"
            settings.RERANKER_DEVICE -> "cpu"
            settings.RERANKER_INPUT_TOP_K -> 5
            settings.RERANKER_OUTPUT_TOP_K -> 5
            settings.RERANKER_MAX_LENGTH -> 256
            settings.RERANKER_BATCH_SIZE -> 4
            settings.RERANKER_SCORE_THRESHOLD -> 0.3
            settings.RERANKER_DOC_CHAR_LIMIT -> 800
            settings.RERANKER_TIMEOUT -> 10
        """
        self.enabled = settings.RERANKER_ENABLED
        self.model_name = settings.RERANKER_MODEL_NAME
        self.model_path = settings.RERANKER_MODEL_PATH
        self.device = settings.RERANKER_DEVICE
        self.input_top_k = settings.RERANKER_INPUT_TOP_K
        self.output_top_k = settings.RERANKER_OUTPUT_TOP_K
        self.max_length = settings.RERANKER_MAX_LENGTH
        self.batch_size = settings.RERANKER_BATCH_SIZE
        self.score_threshold = settings.RERANKER_SCORE_THRESHOLD
        self.doc_char_limit = settings.RERANKER_DOC_CHAR_LIMIT
        self.timeout = settings.RERANKER_TIMEOUT

        # 模型实例，初始为 None，延迟加载
        self._model = None
        self._initialized = False
        # 线程锁：保护延迟初始化，防止多请求并发重复加载模型
        self._lock = threading.Lock()

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        将相对路径转换为绝对路径

        参数:
            relative_path: 相对路径，如 "models/reranker"

        返回:
            绝对路径，如 "E:/dev-rag/dev-rag/aicode/models/reranker"
        """
        if os.path.isabs(relative_path):
            return relative_path
        # 相对于项目根目录（aicode/）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, relative_path)

    def _is_valid_model_path(self, path: str) -> bool:
        """
        检查路径是否包含有效的模型文件

        参数:
            path: 模型目录路径

        返回:
            bool: 是否为有效的模型目录
        """
        if not os.path.isdir(path):
            return False

        # 检查必要文件
        required_files = ["config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False

        # 检查模型权重文件（任一格式即可）
        model_weight_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]
        has_weights = any(os.path.exists(os.path.join(path, f)) for f in model_weight_files)

        return has_weights

    def _init_model(self):
        """
        初始化 BGE-Reranker 模型

        加载策略：
        1. 检查是否启用（RERANKER_ENABLED）
        2. 尝试从本地路径加载
        3. 本地无效时从 HuggingFace 下载/加载

        CPU 优化：
        - 设置 PyTorch 线程数
        - 关闭 FP16，使用 FP32
        - 显式指定 device='cpu'

        异常处理：
        - 库未安装：打印提示，标记为不可用
        - 加载失败：打印错误，标记为不可用（降级到粗排）
        """
        if not self.enabled:
            print("[Reranker] 精排功能已关闭（RERANKER_ENABLED=False）")
            return

        print(f"[Reranker] === 模型加载开始 ===")
        print(f"[Reranker]   模型名称: {self.model_name}")
        print(f"[Reranker]   本地路径: {self.model_path}")
        print(f"[Reranker]   设备: {self.device}")
        print(f"[Reranker]   FP16: False (CPU 强制 FP32)")
        print(f"[Reranker]   max_length: {self.max_length}")
        print(f"[Reranker]   batch_size: {self.batch_size}")

        try:
            from FlagEmbedding import FlagReranker
            print(f"[Reranker]   FlagEmbedding 导入成功")

            # CPU 优化：设置 PyTorch 线程数
            try:
                import torch
                # 根据 CPU 核心数设置线程数（最多 8 线程）
                cpu_count = os.cpu_count() or 4
                num_threads = min(4, max(1, cpu_count // 2))
                torch.set_num_threads(num_threads)
                print(f"[Reranker]   PyTorch 线程数: {num_threads} (CPU 核心数: {cpu_count})")
                print(f"[Reranker]   PyTorch 版本: {torch.__version__}")
            except ImportError:
                print("[Reranker]   警告: 未安装 torch，无法设置线程数")

            _load_t0 = time.time()

            # 尝试从本地路径加载
            if self.model_path:
                abs_path = self._get_absolute_path(self.model_path)
                print(f"[Reranker]   检查本地模型路径: {abs_path}")
                if self._is_valid_model_path(abs_path):
                    # 本地模型有效，直接加载
                    print(f"[Reranker]   本地模型路径有效，开始加载...")
                    self._model = FlagReranker(
                        abs_path,
                        use_fp16=False,  # CPU 必须关闭 FP16
                        device=self.device
                    )
                    _load_ms = (time.time() - _load_t0) * 1000
                    print(f"[Reranker] === 模型加载成功（本地） 耗时: {_load_ms:.0f}ms ===")
                    print(f"[Reranker]   路径: {abs_path}")
                    print(f"[Reranker]   device={self.device}, use_fp16=False")
                    self._initialized = True
                    return
                else:
                    print(f"[Reranker]   本地模型路径无效或不完整: {abs_path}")
                    print(f"[Reranker]   将尝试从 HuggingFace 下载/加载...")

            # 从 HuggingFace 加载（会自动下载并缓存到本地）
            print(f"[Reranker]   从 HuggingFace 加载模型: {self.model_name}")
            print(f"[Reranker]   （首次加载约需下载 1.1GB，请耐心等待...）")
            self._model = FlagReranker(
                self.model_name,
                use_fp16=False,  # CPU 必须关闭 FP16
                device=self.device
            )
            _load_ms = (time.time() - _load_t0) * 1000
            print(f"[Reranker] === 模型加载成功（HuggingFace） 耗时: {_load_ms:.0f}ms ===")
            print(f"[Reranker]   模型: {self.model_name}")
            print(f"[Reranker]   device={self.device}, use_fp16=False")
            print(f"[Reranker]   max_length={self.max_length}, batch_size={self.batch_size}, "
                  f"input_top_k={self.input_top_k}, output_top_k={self.output_top_k}")
            self._initialized = True

        except ImportError as e:
            print(f"[Reranker] === 模型加载失败: FlagEmbedding 未安装 ===")
            print(f"[Reranker]   错误: {e}")
            print(f"[Reranker]   请运行: pip install FlagEmbedding torch transformers")
            self._model = None
            self._initialized = True  # 标记已尝试初始化，避免重复尝试
        except Exception as e:
            print(f"[Reranker] === 模型加载失败 ===")
            print(f"[Reranker]   错误类型: {type(e).__name__}")
            print(f"[Reranker]   错误信息: {e}")
            print(f"[Reranker]   排查建议:")
            print(f"[Reranker]     1. 检查网络连接（首次需下载模型约 1.1GB）")
            print(f"[Reranker]     2. 检查本地模型路径是否正确: {self.model_path}")
            print(f"[Reranker]     3. 或手动下载模型到 models/reranker 目录")
            print(f"[Reranker]     4. 系统将降级到粗排模式")
            import traceback
            traceback.print_exc()
            self._model = None
            self._initialized = True

    def _ensure_model(self):
        """
        确保模型已加载（延迟初始化，线程安全）

        使用双重检查锁定（Double-Checked Locking）模式：
        1. 第一次检查 _initialized（无锁，快速路径）
        2. 加锁后第二次检查（防止并发重复加载）
        3. 仅首次调用时执行 _init_model()

        线程安全说明：
        - FastAPI 异步框架中，同步的 _init_model() 可能在多线程下并发执行
        - 不加锁时，两个请求同时进入 _init_model()，模型会被加载两次
        - 加锁后，第二个请求阻塞等待第一个完成加载，直接复用结果
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._init_model()

    def is_available(self) -> bool:
        """
        检查 Reranker 服务是否可用

        返回:
            bool: 模型是否已成功加载且启用
                  - True: 可以正常使用精排功能
                  - False: 需检查模型安装和配置，系统将降级到粗排
        """
        if not self.enabled:
            return False
        self._ensure_model()
        return self._model is not None

    def _preprocess_text(self, text: str, max_chars: Optional[int] = None) -> str:
        """
        预处理文本（截断过长的文档）

        在送入 Reranker 前截断文档长度，减少序列计算量。
        配合 MAX_LENGTH 使用，进一步优化 CPU 推理速度。

        参数:
            text: 原始文本
            max_chars: 最大字符数，默认使用 settings.RERANKER_DOC_CHAR_LIMIT

        返回:
            截断后的文本（超长时添加省略号）
        """
        if max_chars is None:
            max_chars = self.doc_char_limit

        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def _batch_compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        批量计算 Query-Document 对的相关性分数
        BGE-Reranker 打分(sigmoid 归一化 0-1) → 阈值过滤(score_threshold) → 按精排分降序 → 不足 top_k 补充低分文档

        将文档对分批送入模型，避免内存溢出。
        CPU 环境下使用小 batch_size 保证稳定性。

        参数:
            pairs: Query-Document 对列表，如 [["query", "doc1"], ["query", "doc2"]]

        返回:
            原始分数列表（未归一化）
        """
        if not pairs:
            print(f"[Reranker]   分数计算: 无输入对（pairs 为空）")
            return []

        print(f"[Reranker] === 分数计算开始 ===")
        print(f"[Reranker]   输入: {len(pairs)} 个 Query-Document 对")
        print(f"[Reranker]   batch_size={self.batch_size}, max_length={self.max_length}")
        print(f"[Reranker]   预计批次数: {(len(pairs) + self.batch_size - 1) // self.batch_size}")

        all_scores = []
        total_batches = (len(pairs) + self.batch_size - 1) // self.batch_size

        # 分批处理
        for i in range(0, len(pairs), self.batch_size):
            batch_idx = i // self.batch_size
            batch = pairs[i:i + self.batch_size]
            print(f"[Reranker]   批次 {batch_idx + 1}/{total_batches}: {len(batch)} 对, "
                  f"doc长度={[len(p[1]) for p in batch]}")

            try:
                _batch_t0 = time.time()
                # FlagReranker.compute_score 支持批量输入
                # normalize=True 会自动做 sigmoid 归一化
                scores = self._model.compute_score(
                    batch,
                    max_length=self.max_length,
                    normalize=True  # 自动 sigmoid 归一化到 0-1
                )
                _batch_ms = (time.time() - _batch_t0) * 1000

                # 确保返回列表
                if isinstance(scores, (int, float)):
                    scores = [float(scores)]
                else:
                    scores = [float(s) for s in scores]

                print(f"[Reranker]   批次 {batch_idx + 1} 完成 耗时: {_batch_ms:.1f}ms, "
                        f"分数: {[f'{s:.4f}' for s in scores]}")
                all_scores.extend(scores)
            except Exception as e:
                print(f"[Reranker]   批次 {batch_idx + 1} 计算失败: {type(e).__name__}: {e}")
                # 向上抛出，由 rerank() 捕获后整体降级为粗排结果，
                # 避免用 0 分污染精排结果导致"检索不到内容"
                raise

        print(f"[Reranker] === 分数计算结束 共 {len(all_scores)} 个分数 ===")
        print(f"[Reranker]   分数明细: {[(i, f'{s:.4f}') for i, s in enumerate(all_scores)]}")
        print(f"[Reranker]   最高分: {max(all_scores):.4f}, 最低分: {min(all_scores):.4f}, "
              f"平均分: {sum(all_scores)/len(all_scores):.4f}")

        return all_scores

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None
    ) -> List[dict]:
        """
        执行精排重排序

        对粗排返回的文档进行 Cross-Encoder 精排，
        返回按精排分数降序排列的结果。

        参数:
            query: 用户查询文本
            documents: 粗排返回的文档列表，每个文档包含:
                - content: str, 文本内容
                - score: float, 粗排相似度分数
                - metadata: dict, 元数据
                - id: str, 节点 ID
            top_k: 返回的文档数量，默认使用 settings.RERANKER_OUTPUT_TOP_K

        返回:
            List[Dict]: 精排后的文档列表，每个文档包含:
                - content: str, 文本内容
                - score: float, 精排分数（0-1）
                - original_score: float, 粗排分数
                - metadata: dict, 元数据
                - id: str, 节点 ID

        降级行为：
        - 模型不可用: 返回粗排结果（按原分数降序）
        - 推理超时: 返回粗排结果
        - 推理异常: 返回粗排结果
        """
        # 确保模型已加载（首次调用耗时约 5-10s，需单独计时以便排查）
        _model_load_t0 = time.time()
        self._ensure_model()
        _model_load_ms = (time.time() - _model_load_t0) * 1000
        print(f"[Reranker]   模型准备完成 耗时: {_model_load_ms:.1f}ms, 可用: {self.is_available()}, 已初始化: {self._initialized}")

        # 如果 top_k 未指定，使用配置的输出数量
        if top_k is None:
            top_k = self.output_top_k

        # 降级检查：模型不可用
        if not self.is_available():
            print("[Reranker] === 降级: 模型不可用，返回粗排结果 ===")
            return self._fallback_rerank(documents, top_k)

        # 空结果处理
        if not documents:
            print("[Reranker] 输入文档为空，返回空列表")
            return []

        print(f"[Reranker] === 精排开始 ===")
        print(f"[Reranker]   查询: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
        print(f"[Reranker]   输入文档: {len(documents)} 个")
        _scores_preview = [(i, round(d.get('score', 0), 4)) for i, d in enumerate(documents)]
        print(f"[Reranker]   粗排分数: {_scores_preview}")

        # 限制输入数量（CPU 优化：只取前 input_top_k 个候选）
        candidates = documents[:self.input_top_k]
        print(f"[Reranker]   截取前 {self.input_top_k} 个候选 (实际: {len(candidates)} 个)")

        # 预处理文档（截断过长文本）
        pairs = []
        for doc in candidates:
            content = doc.get("content", "")
            truncated_content = self._preprocess_text(content)
            pairs.append([query, truncated_content])

        # === Phase 1: 模型推理（仅此阶段允许降级到粗排） ===
        try:
            start_time = time.time()
            scores = self._batch_compute_scores(pairs)
            elapsed = time.time() - start_time
            print(f"[Reranker] === 精排推理完成: {len(pairs)} 个文档, 耗时: {elapsed:.3f}s ===")

            # 超时检查
            if elapsed > self.timeout:
                print(f"[Reranker]   警告: 精排耗时 {elapsed:.1f}s 超过阈值 {self.timeout}s")
                print(f"[Reranker]   建议减小 RERANKER_INPUT_TOP_K 或 RERANKER_MAX_LENGTH")
                # 仍然返回结果，只是记录警告

        except Exception as e:
            # 仅模型推理异常才降级（网络超时、模型加载失败、批次计算错误等）
            print(f"[Reranker] === 精排推理异常 降级到粗排结果 ===")
            print(f"[Reranker]   错误类型: {type(e).__name__}")
            print(f"[Reranker]   错误信息: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_rerank(documents, top_k)

        # === Phase 2: 结果处理（业务逻辑，异常应向上传播而非降级） ===
        # 说明：阈值过滤、排序、补充逻辑属于纯数据处理，不应失败。
        # 如果此处异常，说明代码有 bug，不应静默降级，应让上层感知。
        print(f"[Reranker] === 阈值过滤开始 ===")
        print(f"[Reranker]   阈值: {self.score_threshold}")
        print(f"[Reranker]   过滤前: {len(candidates)} 个文档")

        reranked_results = []
        for i, doc in enumerate(candidates):
            rerank_score = scores[i] if i < len(scores) else 0.0
            doc_preview = doc.get("content", "")[:50].replace('\n', ' ')

            # 应用置信度阈值过滤
            if rerank_score < self.score_threshold:
                print(f"[Reranker]   [FILTER] 文档 {i} 过滤 | 精排分={rerank_score:.4f} < 阈值={self.score_threshold} "
                      f"| 粗排分={doc.get('score', 0):.4f} | {doc_preview}...")
                continue
            else:
                print(f"[Reranker]   [KEEP]   文档 {i} 保留 | 精排分={rerank_score:.4f} >= 阈值={self.score_threshold} "
                      f"| 粗排分={doc.get('score', 0):.4f} | {doc_preview}...")

            reranked_results.append({
                "content": doc.get("content", ""),  # 保留原始完整内容
                "score": rerank_score,  # 精排分数
                "original_score": doc.get("score", 0.0),  # 保留粗排分数
                "metadata": doc.get("metadata", {}),
                "id": doc.get("id", "")
            })

        print(f"[Reranker]   过滤后: {len(reranked_results)} 个文档达标")

        # 按精排分数降序排序
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        _top_preview = [(r['id'][:8], round(r['score'], 4)) for r in reranked_results[:top_k]]
        print(f"[Reranker]   排序后 Top-{min(len(reranked_results), top_k)}: {_top_preview}")

        # 如果精排后结果不足 top_k，补充被过滤的低分文档
        #  为什么要有“补充低分文档”的逻辑：
        #   核心问题：精排阈值过滤后，达标的文档数量可能少于用户请求的 top_k。
        # 如果不补充，会发生什么：
        #   只返回 1 条文档给 LLM，问题在于：
        #       1、这条文档可能不足以回答问题 — 比如用户问"如何配置 FastAPI 的数据库连接和中间件"，达标的那 1 条文档可能只覆盖了数据库连接，中间件部分在被过滤的文档中
        #       2、精排也不是完美的 — BGE-Reranker 给出的评分表示"不太相关"，但不代表"完全无用"。被过滤的文档可能包含背景知识、定义说明等辅助信息
        #       3、LLM 缺少上下文时容易"幻觉" — 上下文太少时，LLM 更倾向于用自身参数知识编造内容，而不是基于检索内容回答
        # 当前选择"补充低分文档"的策略：宁可多花 token，也要保证 LLM 有足够的上下文来生成完整回答。这是一个"质量优先于成本"的决策。
        if len(reranked_results) < top_k:
            print(f"[Reranker]   精排后仅 {len(reranked_results)} 个达标，需 {top_k} 个，补充低分文档")
            # 获取被过滤的文档，使用 enumerate 预建索引避免 O(n²) 的 index() 调用
            filtered_ids = {r["id"] for r in reranked_results}
            for idx, doc in enumerate(candidates):
                if doc.get("id", "") not in filtered_ids:
                    rerank_score = scores[idx] if idx < len(scores) else 0.0
                    coarse_score = doc.get("score", 0.0)
                    direction = "UP" if rerank_score > coarse_score else ("DOWN" if rerank_score < coarse_score else "SAME")
                    print(f"[Reranker]   补充: doc_id={doc.get('id', '')[:8]}, 精排分={rerank_score:.4f}, 粗排分={coarse_score:.4f} ({direction})")
                    reranked_results.append({
                        "content": doc.get("content", ""),
                        "score": rerank_score,
                        "original_score": coarse_score,
                        "metadata": doc.get("metadata", {}),
                        "id": doc.get("id", "")
                    })
                    if len(reranked_results) >= top_k:
                        break
        else:
            print(f"[Reranker]   精排后 {len(reranked_results)} 个达标，满足 top_k={top_k}，无需补充")

        # 标记为真正的精排结果（供上层区分精排/降级，避免量纲混用）
        for item in reranked_results:
            item["reranked"] = True

        final_results = reranked_results[:top_k]
        print(f"[Reranker] === 精排结束 返回 {len(final_results)} 个结果 ===")
        for i, r in enumerate(final_results):
            print(f"[Reranker]   [{i+1}] 精排分={r['score']:.4f} | 粗排分={r['original_score']:.4f} | "
                  f"id={r['id'][:12]}...")

        return final_results

    def _fallback_rerank(self, documents: List[dict], top_k: int) -> List[dict]:
        """
        降级处理：返回粗排结果

        当 Reranker 不可用或推理失败时，直接返回粗排结果。
        按粗排分数降序排序。

        参数:
            documents: 粗排返回的文档列表
            top_k: 返回的文档数量

        返回:
            List[Dict]: 按粗排分数降序排列的文档列表
        """
        # 按粗排分数降序排序
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)

        # 添加 original_score 字段（与精排结果格式一致）
        results = []
        for doc in sorted_docs[:top_k]:
            results.append({
                "content": doc.get("content", ""),
                "score": doc.get("score", 0.0),  # 使用粗排分数
                "original_score": doc.get("score", 0.0),
                "metadata": doc.get("metadata", {}),
                "id": doc.get("id", ""),
                "reranked": False,  # 降级标记：未执行精排
            })

        return results

    def rerank_retrieved_documents(
        self,
        query: str,
        documents: List,  # List[RetrievedDocument]
        top_k: Optional[int] = None
    ) -> List:
        """
        精排 RetrievedDocument 列表

        适配 retriever.py 中的 RetrievedDocument 数据类。
        将 RetrievedDocument 转换为 dict 进行精排，再转换回 RetrievedDocument。

        参数:
            query: 用户查询文本
            documents: RetrievedDocument 列表
            top_k: 返回的文档数量

        返回:
            List[RetrievedDocument]: 精排后的 RetrievedDocument 列表
        """
        # 延迟导入，避免循环依赖
        from backend.rag.retriever import RetrievedDocument

        _t_all = time.time()
        print(f"[Reranker-Adaptor] === rerank_retrieved_documents 开始 ===")
        print(f"[Reranker-Adaptor]   输入: {len(documents)} 个 RetrievedDocument, top_k={top_k}")

        # 转换为 dict 格式
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "content": doc.content,
                "score": doc.score,
                "metadata": doc.metadata or {},
                "id": doc.chunk_id
            })

        print(f"[Reranker-Adaptor]   转换为 dict 格式完成, 调用 rerank()")
        # 执行精排
        reranked = self.rerank(query, doc_dicts, top_k)
        print(f"[Reranker-Adaptor]   rerank() 返回, 结果 {len(reranked)} 条")

        # 转换回 RetrievedDocument
        # 字段语义：
        #   - score:          主分数。真正精排时为精排分数（下游按此排序），降级时为粗排分数
        #   - rerank_score:   精排分数，仅真正执行精排时非 None（用于区分精排/粗排量纲）
        #   - original_score: 粗排原始分数（用于展示/对比）
        results = []
        reranked_count = 0
        fallback_count = 0
        for r in reranked:
            is_reranked = r.get("reranked", False)
            if is_reranked:
                reranked_count += 1
            else:
                fallback_count += 1
            results.append(RetrievedDocument(
                document_id=r.get("metadata", {}).get("document_id", ""),
                chunk_id=r.get("id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),  # 精排分数（降级时为粗排分数）
                metadata=r.get("metadata"),
                rerank_score=r.get("score") if is_reranked else None,
                original_score=r.get("original_score", r.get("score", 0.0)),
            ))

        _total_ms = (time.time() - _t_all) * 1000
        print(f"[Reranker-Adaptor]   转换回 RetrievedDocument 完成: 精排 {reranked_count} 条, 降级 {fallback_count} 条")
        print(f"[Reranker-Adaptor] === rerank_retrieved_documents 结束 总耗时: {_total_ms:.1f}ms ===")

        return results


# 全局单例 - 应用启动时初始化，整个生命周期复用
# 这样做的好处：
# 1. 避免模型重复加载（加载一次约 5-10 秒）
# 2. 统一管理模型实例
# 3. 减少内存占用
reranker_service = RerankerService()
