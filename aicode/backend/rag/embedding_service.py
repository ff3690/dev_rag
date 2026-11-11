"""
Embedding 服务模块 - Embedding Service Module

基于 LlamaIndex 的 HuggingFaceEmbedding 封装，提供文本向量化能力。
使用 BAAI/bge-small-zh-v1.5 模型（512维），专为中文优化，支持 CPU 推理。

核心职责：
1. 将文本转换为固定维度的向量表示
2. 支持批量向量化，提高处理效率
3. 提供降级方案：模型不可用时返回零向量

技术选型说明：
- 选择 HuggingFaceEmbedding 而非 sentence-transformers：
  LlamaIndex 原生集成，支持与 VectorStoreIndex 无缝协作
- 选择 bge-small-zh-v1.5 而非 bge-base/large：
  无 GPU 环境下的性能最优选择，内存占用仅 ~300MB

模型加载策略：
1. 优先使用本地路径（EMBEDDING_MODEL_PATH）：适合已下载模型或离线环境
2. 本地无效时多源自动下载（ModelScope → HF镜像 → HF官方）：
   - ModelScope：阿里云国内源，速度快，无需代理
   - HuggingFace 镜像站（hf-mirror.com）：国内可访问
   - HuggingFace 官方：需代理，兜底方案
   下载成功后回退到本地路径加载，避免 HuggingFace 单源网络超时
"""

import os
import time
import shutil
from typing import List, Optional

from backend.config.settings import settings


class EmbeddingService:
    """
    Embedding 服务类 - Embedding Service Class

    封装 LlamaIndex 的 HuggingFaceEmbedding，提供统一的文本向量化接口。

    设计要点：
    1. 单例模式：通过模块级 embedding_service 实例复用模型，避免重复加载
    2. 延迟初始化：_init_model 在 __init__ 中调用，但模型文件首次使用时才下载
    3. 优雅降级：模型加载失败时返回零向量，不影响主流程

    使用示例：
        service = EmbeddingService()
        vector = service.get_embedding("你好世界")  # 返回 512 维向量
        vectors = service.get_embeddings(["文本1", "文本2"])  # 批量处理
    """

    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 Embedding 服务

        参数:
            model_name: 模型名称，默认从 settings.EMBEDDING_MODEL_NAME 读取
            model_path: 本地模型路径，默认从 settings.EMBEDDING_MODEL_PATH 读取
            device: 运行设备，默认从 settings.EMBEDDING_DEVICE 读取
                    可选值: "cpu", "cuda", "cuda:0", "mps"

        配置来源：
            settings.EMBEDDING_MODEL_NAME -> "BAAI/bge-small-zh-v1.5"
            settings.EMBEDDING_MODEL_PATH -> "models/embedding" (本地路径)
            settings.EMBEDDING_DEVICE -> "cpu"
            settings.EMBEDDING_DIMENSION -> 512

        加载优先级：
            1. 如果 model_path 有效且包含模型文件，直接从本地加载
            2. 否则使用 model_name 从 HuggingFace Hub 下载/加载
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.device = device or settings.EMBEDDING_DEVICE
        self.dimension = settings.EMBEDDING_DIMENSION

        # 模型实例，初始为 None，延迟加载
        self._embed_model = None
        self._init_model()

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        将相对路径转换为绝对路径

        参数:
            relative_path: 相对路径，如 "models/embedding"

        返回:
            绝对路径，如 "E:/dev-rag/dev-rag/aicode/models/embedding"
        """
        if os.path.isabs(relative_path):
            return relative_path
        # 相对于项目根目录（aicode/）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, relative_path)

    def _is_valid_model_path(self, path: str) -> bool:
        """
        检查路径是否包含有效的模型文件

        必要文件：config.json, tokenizer.json, 模型权重文件

        参数:
            path: 模型目录路径

        返回:
            bool: 是否为有效的模型目录
        """
        if not os.path.isdir(path):
            return False

        # 检查必要文件
        required_files = ["config.json", "tokenizer.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False

        # 检查模型权重文件（任一格式即可）
        model_weight_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]
        has_weights = any(os.path.exists(os.path.join(path, f)) for f in model_weight_files)

        return has_weights

    def _download_from_modelscope(self, model_name: str, output_dir: str) -> bool:
        """
        从 ModelScope 下载模型（国内推荐源）

        ModelScope 是阿里云的机器学习平台，国内访问速度快、无需代理。

        参数:
            model_name: HuggingFace 格式的模型名（如 BAAI/bge-small-zh-v1.5）
            output_dir: 目标输出目录（绝对路径）

        返回:
            bool: 是否下载成功
        """
        print("  [源 1] 尝试从 ModelScope 下载（国内源，推荐）...")
        try:
            from modelscope import snapshot_download as ms_snapshot_download

            print(f"    仓库: {model_name}")
            print(f"    路径: {output_dir}")

            start_time = time.time()
            local_path = ms_snapshot_download(
                model_id=model_name,
                cache_dir=output_dir,
            )
            elapsed = time.time() - start_time
            print(f"  ✓ ModelScope 下载成功！耗时: {elapsed:.1f} 秒")
            print(f"    下载位置: {local_path}")

            # ModelScope 会将文件放在 output_dir/<模型名>/snapshots/<hash>/ 下
            # 需要将模型文件复制/移动到 output_dir 根目录，便于直接加载
            self._flatten_model_dir(local_path, output_dir)
            return True

        except ImportError:
            print("  ✗ 未安装 modelscope 库（可运行: pip install modelscope）")
            return False
        except Exception as e:
            print(f"  ✗ ModelScope 下载失败: {str(e)}")
            return False

    def _download_from_huggingface(self, model_name: str, output_dir: str, mirror: bool = True) -> bool:
        """
        从 HuggingFace 下载模型

        参数:
            model_name: HuggingFace 模型标识符
            output_dir: 目标输出目录（绝对路径）
            mirror: True=使用 hf-mirror.com 镜像；False=使用 HuggingFace 官方

        返回:
            bool: 是否下载成功
        """
        source_name = "HuggingFace 镜像站 (hf-mirror.com)" if mirror else "HuggingFace 官方"
        print(f"  [源 {'2' if mirror else '3'}] 尝试从 {source_name} 下载...")

        # 设置镜像站环境变量（仅本次进程生效）
        if mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"    已设置镜像: https://hf-mirror.com")
        else:
            # 清除镜像设置，使用官方源
            os.environ.pop("HF_ENDPOINT", None)

        try:
            from huggingface_hub import snapshot_download

            print(f"    模型: {model_name}")
            print(f"    路径: {output_dir}")

            start_time = time.time()
            local_path = snapshot_download(
                repo_id=model_name,
                local_dir=output_dir,
            )
            elapsed = time.time() - start_time
            print(f"  ✓ {source_name} 下载成功！耗时: {elapsed:.1f} 秒")
            print(f"    下载位置: {local_path}")
            return True

        except ImportError:
            print("  ✗ 未安装 huggingface_hub 库（可运行: pip install huggingface_hub）")
            return False
        except Exception as e:
            print(f"  ✗ {source_name} 下载失败: {str(e)}")
            return False

    def _flatten_model_dir(self, src_dir: str, dst_dir: str):
        """
        将 ModelScope 下载的嵌套目录结构扁平化到目标目录

        ModelScope 的 snapshot_download 会将文件放在:
            <cache_dir>/<model_name>/snapshots/<commit_hash>/
        而本项目直接使用 models/embedding/ 作为加载路径，
        因此需要把模型文件复制到 dst_dir 根目录下，确保可被直接识别。

        参数:
            src_dir: snapshot_download 返回的实际文件目录
            dst_dir: 目标目录（一般为 models/embedding）
        """
        # 如果源目录就是目标目录，无需处理
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            return

        os.makedirs(dst_dir, exist_ok=True)
        print(f"    整理模型文件到: {dst_dir}")

        for item in os.listdir(src_dir):
            src_path = os.path.join(src_dir, item)
            dst_path = os.path.join(dst_dir, item)

            # 跳过 ModelScope 缓存的元数据目录，避免循环复制
            if item in (".cache", "models"):
                continue

            try:
                if os.path.isfile(src_path):
                    # 文件直接复制
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    # 目录复制（如 1_Pooling/）
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
            except Exception as e:
                print(f"    警告: 复制 {item} 时出错: {e}")

    def _download_model_from_sources(self, model_name: str, output_dir: str) -> bool:
        """
        自动从多个下载源尝试下载模型（按优先级顺序）

        下载源顺序（与 scripts/download_embedding_model.py 保持一致）：
        1. ModelScope（国内源，速度最快，无需代理）
        2. HuggingFace 镜像站（hf-mirror.com，国内可访问）
        3. HuggingFace 官方（需代理，最后兜底）

        参数:
            model_name: 模型名称，如 BAAI/bge-small-zh-v1.5
            output_dir: 目标输出目录（绝对路径）

        返回:
            bool: 是否至少有一个源下载成功
        """
        print("\n" + "=" * 60)
        print("自动选择下载源（ModelScope → HF镜像 → HF官方）")
        print("=" * 60)
        print(f"  模型: {model_name}")
        print(f"  目标: {output_dir}")
        print()

        os.makedirs(output_dir, exist_ok=True)

        # 源 1: ModelScope
        if self._download_from_modelscope(model_name, output_dir):
            return True

        # 源 2: HuggingFace 镜像站
        if self._download_from_huggingface(model_name, output_dir, mirror=True):
            return True

        # 源 3: HuggingFace 官方
        if self._download_from_huggingface(model_name, output_dir, mirror=False):
            return True

        # 所有源都失败
        print("\n✗ 所有下载源均失败！")
        print("建议:")
        print("  1. 检查网络连接")
        print("  2. 手动执行: python scripts/download_embedding_model.py --source auto")
        print("  3. 使用代理: 在 .env 或环境变量中设置 HTTP_PROXY / HTTPS_PROXY")
        return False

    def _init_model(self):
        """
        初始化 Embedding 模型

        加载策略（按优先级尝试）：
        1. 本地路径加载：如果 EMBEDDING_MODEL_PATH 指向有效的模型目录
        2. 多源自动下载：本地无效时依次尝试 ModelScope → HF镜像 → HF官方
           下载成功后回退到本地路径加载，避免 HuggingFace 单源网络超时

        异常处理：
        - ImportError: llama-index 未安装时打印提示
        - 其他异常：模型文件损坏或下载失败时降级为零向量
        """
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # 尝试从本地路径加载
            if self.model_path:
                abs_path = self._get_absolute_path(self.model_path)
                if self._is_valid_model_path(abs_path):
                    # 本地模型有效，直接加载
                    print(f"从本地路径加载模型: {abs_path}")
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=abs_path,      # 使用绝对路径作为 model_name
                        device=self.device,        # 推理设备
                        show_progress_bar=True     # 恢复模型加载进度条显示
                    )
                    print(f"✓ 本地 Embedding 模型加载成功: {abs_path} (device={self.device} , dimension={self.dimension})")
                    return
                else:
                    print(f"提示: 本地模型路径无效或不完整: {abs_path}")
                    print(f"  将尝试自动下载模型（支持多源切换）...")

                    # 本地模型无效，尝试多源下载
                    cache_folder = abs_path  # 直接下载到配置的本地路径
                    if self._download_model_from_sources(self.model_name, cache_folder):
                        # 下载成功后，重新校验本地路径并加载
                        if self._is_valid_model_path(cache_folder):
                            print(f"\n从本地路径加载已下载的模型: {cache_folder}")
                            self._embed_model = HuggingFaceEmbedding(
                                model_name=cache_folder,
                                device=self.device,
                                show_progress_bar=True     # 恢复模型加载进度条显示
                            )
                            print(f"✓ 本地 Embedding 模型加载成功: {cache_folder} (device={self.device} , dimension={self.dimension})")
                            return
                        else:
                            print(f"警告: 下载完成但模型文件仍不完整: {cache_folder}")
                            print("  将尝试直接使用 HuggingFace 加载方式...")
                    else:
                        print("  多源下载均失败，最后尝试直接通过 HuggingFaceEmbedding 加载...")

            # 最终回退：通过 HuggingFaceEmbedding 自动加载（可能再次尝试 HF 下载）
            cache_folder = self._get_absolute_path("models/embedding")
            os.makedirs(cache_folder, exist_ok=True)

            # 如果配置了镜像，确保环境变量已设置（兼容用户手动设置的 HF_ENDPOINT）
            print(f"从 HuggingFace 加载模型: {self.model_name}")
            self._embed_model = HuggingFaceEmbedding(
                model_name=self.model_name,      # HuggingFace 模型标识符
                cache_folder=cache_folder,        # 本地缓存路径
                device=self.device,               # 推理设备
                show_progress_bar=True            # 恢复模型加载进度条显示
            )
            print(f"✓ Embedding 模型加载成功: {self.model_name} (device={self.device})")

        except ImportError:
            print("警告: llama-index 未安装，Embedding 功能不可用")
            print("请运行: pip install llama-index-core llama-index-embeddings-huggingface")
            self._embed_model = None
        except Exception as e:
            print(f"警告: Embedding 模型加载失败: {str(e)}")
            print("提示:")
            print("  1. 检查本地模型路径是否正确")
            print("  2. 首次运行需联网下载模型")
            print("  3. 或手动执行【python scripts/download_embedding_model.py】下载模型到文件 models/embedding/")
            self._embed_model = None

    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的 Embedding 向量

        将输入文本转换为 512 维的稠密向量表示，
        用于后续的向量相似度计算。

        参数:
            text: 输入文本，建议不超过 512 个 token（约 200-300 中文字）

        返回:
            list: 512 维浮点向量
                  - 模型可用时：真实的语义向量
                  - 模型不可用时：全零向量（降级处理）

        性能：CPU 环境下单次约 10-30ms
        """
        if self._embed_model is None:
            # 降级方案：返回零向量，保证系统不崩溃
            # 注意：零向量无法进行有效的相似度检索
            return [0.0] * self.dimension

        # 使用 LlamaIndex 公共 API
        return self._embed_model.get_text_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的 Embedding 向量

        一次性处理多个文本，利用向量化并行性提高效率。
        比逐个调用 get_embedding 快 2-3 倍。

        参数:
            texts: 文本列表，建议批量大小不超过 64

        返回:
            list: 向量列表，每个元素是 512 维向量
                  输入 texts[i] 对应输出 embeddings[i]

        性能：CPU 环境下批量 32 条约 200-500ms
        """
        if self._embed_model is None:
            return [[0.0] * self.dimension for _ in texts]

        # LlamaIndex HuggingFaceEmbedding API 变化处理
        # 新版本使用 _get_text_embeddings (私有方法)
        # 或者逐个调用 get_text_embedding
        try:
            # 尝试使用批量方法
            return self._embed_model.get_text_embeddings(texts)
        except AttributeError:
            # 回退到逐个获取
            return [self._embed_model.get_text_embedding(text) for text in texts]

    def get_embeddings_with_progress(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量获取向量（带进度提示）

        适用于大批量文本处理（如文档索引），
        定期打印进度信息，避免长时间无反馈。

        参数:
            texts: 文本列表
            batch_size: 批处理大小
                        - CPU 环境: 建议 16-32（内存与速度平衡）
                        - GPU 环境: 建议 64-128（充分利用并行能力）

        返回:
            list: 向量列表

        进度输出示例：
            向量化进度: 25.0% (25/100)
            向量化进度: 50.0% (50/100)
            ...
        """
        if self._embed_model is None:
            return [[0.0] * self.dimension for _ in texts]

        embeddings = []
        total = len(texts)

        # 分批次处理，每批次打印进度
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            # 使用 get_embeddings 方法（已处理 API 兼容性）
            batch_embeddings = self.get_embeddings(batch)
            embeddings.extend(batch_embeddings)

            # 计算并打印进度
            progress = min((i + batch_size) / total * 100, 100)
            print(f"向量化进度: {progress:.1f}% ({min(i + batch_size, total)}/{total})")

        return embeddings

    @property
    def embed_model(self):
        """
        获取底层 HuggingFaceEmbedding 模型实例

        用于需要直接访问 LlamaIndex Embedding 模型的场景，
        如 VectorStoreIndex 初始化时传入 embed_model 参数。
        """
        return self._embed_model

    def is_available(self) -> bool:
        """
        检查 Embedding 服务是否可用

        返回:
            bool: 模型是否已成功加载
                  - True: 可以正常使用向量化功能
                  - False: 需要检查模型安装和配置
        """
        return self._embed_model is not None


# 全局单例 - 应用启动时初始化，整个生命周期复用
# 这样做的好处：
# 1. 避免模型重复加载（加载一次约 2-5 秒）
# 2. 统一管理模型实例
# 3. 减少内存占用
embedding_service = EmbeddingService()