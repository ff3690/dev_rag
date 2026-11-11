"""
Embedding 模型下载脚本 - Embedding Model Download Script

手动执行此脚本可将 BAAI/bge-small-zh-v1.5 模型下载到本地。
支持多种下载源，自动重试，适合国内网络环境。

使用方法:
    # 默认使用 ModelScope 国内源（推荐，速度最快）
    python scripts/download_embedding_model.py

    # 使用 HuggingFace 镜像站
    python scripts/download_embedding_model.py --source hf-mirror

    # 使用 HuggingFace 官方源（需要代理）
    python scripts/download_embedding_model.py --source hf

可选参数:
    --model: 指定模型名称
    --output: 指定输出目录
    --source: 下载源 (modelscope/hf-mirror/hf)
    --proxy: 使用代理 (http://127.0.0.1:7890)
"""

import os
import sys
import argparse
import time
import shutil


# 下载源配置
DOWNLOAD_SOURCES = {
    "modelscope": {
        "name": "ModelScope (国内源，推荐)",
        "url": "https://www.modelscope.cn",
        "description": "阿里云机器学习平台，国内访问速度快"
    },
    "hf-mirror": {
        "name": "HuggingFace 镜像站",
        "url": "https://hf-mirror.com",
        "description": "HuggingFace 官方镜像，国内可访问"
    },
    "hf": {
        "name": "HuggingFace 官方",
        "url": "https://huggingface.co",
        "description": "官方源，需要代理才能访问"
    }
}


def download_from_modelscope(model_name: str, output_dir: str) -> bool:
    """
    从 ModelScope 下载模型（国内推荐）

    ModelScope 是阿里云的机器学习平台，
    国内访问速度快，无需代理。
    """
    print("\n" + "=" * 60)
    print("尝试从 ModelScope 下载...")
    print("=" * 60)

    try:
        from modelscope import snapshot_download as ms_snapshot_download

        # BAAI/bge-small-zh-v1.5 在 ModelScope 上的仓库名
        # 可能需要使用不同的组织名
        model_id = model_name.replace("BAAI/", "BAAI/")

        print(f"  仓库名: {model_id}")
        print(f"  下载路径: {output_dir}")
        print(f"  预计耗时: 1-3 分钟\n")

        start_time = time.time()

        # ModelScope 下载
        local_path = ms_snapshot_download(
            model_id=model_id,
            cache_dir=output_dir,
        )

        elapsed_time = time.time() - start_time
        print(f"\n✓ ModelScope 下载成功！耗时: {elapsed_time:.1f} 秒")
        print(f"  路径: {local_path}")

        # 如果下载路径与 output_dir 不同，复制文件
        if local_path != output_dir:
            print(f"\n  复制模型文件到目标目录...")
            copy_model_files(local_path, output_dir)

        return True

    except ImportError:
        print("  ✗ 未安装 modelscope 库")
        print("    运行: pip install modelscope")
        return False

    except Exception as e:
        print(f"  ✗ ModelScope 下载失败: {str(e)}")
        return False


def download_from_huggingface(model_name: str, output_dir: str, mirror: bool = True, proxy: str = None) -> bool:
    """
    从 HuggingFace 下载模型

    参数:
        model_name: 模型名称
        output_dir: 输出目录
        mirror: 是否使用镜像站
        proxy: 代理地址（如 http://127.0.0.1:7890）
    """
    source_name = "HuggingFace 镜像站" if mirror else "HuggingFace 官方"
    print(f"\n{'=' * 60}")
    print(f"尝试从 {source_name} 下载...")
    print(f"{'=' * 60}")

    # 设置代理（如果指定）
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"  已设置代理: {proxy}")

    # 设置镜像站
    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"  已设置镜像: https://hf-mirror.com")
    else:
        os.environ.pop("HF_ENDPOINT", None)

    try:
        from huggingface_hub import snapshot_download

        print(f"  模型: {model_name}")
        print(f"  路径: {output_dir}")
        print(f"  预计耗时: 2-5 分钟\n")

        start_time = time.time()

        # HuggingFace 下载
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
        )

        elapsed_time = time.time() - start_time
        print(f"\n✓ {source_name} 下载成功！耗时: {elapsed_time:.1f} 秒")
        print(f"  路径: {local_path}")

        return True

    except ImportError:
        print("  ✗ 未安装 huggingface_hub 库")
        print("    运行: pip install huggingface_hub")
        return False

    except Exception as e:
        print(f"  ✗ {source_name} 下载失败: {str(e)}")
        return False
    finally:
        # 清除代理设置
        if proxy:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)


def copy_model_files(src_dir: str, dst_dir: str):
    """
    复制模型文件到目标目录

    当下载路径与目标路径不同时使用。
    """
    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"    复制: {item}")
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"    复制目录: {item}/")


def verify_download(output_dir: str):
    """
    验证模型下载是否完整
    """
    print(f"\n{'=' * 60}")
    print("验证模型文件...")
    print(f"{'=' * 60}")

    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]

    missing_required = []

    # 检查必要文件
    for file in required_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            missing_required.append(file)
            print(f"  ✗ {file} (缺失)")

    # 检查模型权重文件（格式可能不同）
    has_model = False
    for model_file in model_files:
        file_path = os.path.join(output_dir, model_file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {model_file} ({size / 1024 / 1024:.1f} MB)")
            has_model = True
            break

    if not has_model:
        missing_required.append("model file (pytorch_model.bin / model.safetensors)")
        print(f"  ✗ 模型权重文件 (缺失)")

    # 检查其他必要文件
    other_required = ["special_tokens_map.json", "vocab.txt"]
    for file in other_required:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            # 有些模型可能没有这个文件，不算严重缺失
            if file == "vocab.txt":
                print(f"  • {file} (可选，部分模型使用 tokenizer.json)")
            else:
                print(f"  ✗ {file} (缺失)")
                if file not in ["vocab.txt"]:
                    missing_required.append(file)

    # 计算总大小
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_dir):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, filename))

    print(f"\n总大小: {total_size / 1024 / 1024:.1f} MB")

    if missing_required:
        print(f"\n⚠ 警告: 缺少必要文件: {', '.join(missing_required)}")
        print("  模型可能无法正常工作。")
        return False
    else:
        print(f"\n✓ 模型文件验证通过！")
        return True


def download_embedding_model(model_name: str, output_dir: str, source: str, proxy: str = None) -> bool:
    """
    下载 Embedding 模型（自动选择下载源）

    参数:
        model_name: 模型名称
        output_dir: 输出目录
        source: 下载源 (modelscope/hf-mirror/hf/auto)
        proxy: 代理地址

    返回:
        bool: 是否下载成功
    """
    print("=" * 60)
    print("Embedding 模型下载工具")
    print("=" * 60)
    print(f"  模型名称: {model_name}")
    print(f"  存储路径: {output_dir}")
    print(f"  下载源: {source}")
    if proxy:
        print(f"  代理: {proxy}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据 source 参数选择下载方式
    if source == "modelscope":
        return download_from_modelscope(model_name, output_dir)

    elif source == "hf-mirror":
        return download_from_huggingface(model_name, output_dir, mirror=True, proxy=proxy)

    elif source == "hf":
        return download_from_huggingface(model_name, output_dir, mirror=False, proxy=proxy)

    elif source == "auto":
        # 自动尝试多个源
        print("\n将按顺序尝试以下下载源：")
        print("  1. ModelScope (国内源)")
        print("  2. HuggingFace 镜像站")
        print("  3. HuggingFace 官方")

        # 尝试 ModelScope
        if download_from_modelscope(model_name, output_dir):
            return True

        # 尝试 HuggingFace 镜像站
        if download_from_huggingface(model_name, output_dir, mirror=True, proxy=proxy):
            return True

        # 尝试 HuggingFace 官方
        if download_from_huggingface(model_name, output_dir, mirror=False, proxy=proxy):
            return True

        print("\n✗ 所有下载源都失败了！")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 使用代理: python scripts/download_embedding_model.py --source hf --proxy http://127.0.0.1:7890")
        print("  3. 手动下载模型文件")
        return False

    else:
        print(f"\n错误: 未知的下载源 '{source}'")
        print(f"可选值: {', '.join(DOWNLOAD_SOURCES.keys())}, auto")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载 BAAI/bge-small-zh-v1.5 Embedding 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动选择最优源（推荐）
  python scripts/download_embedding_model.py --source auto

  # 使用 ModelScope 国内源（速度最快）
  python scripts/download_embedding_model.py --source modelscope

  # 使用 HuggingFace 镜像站
  python scripts/download_embedding_model.py --source hf-mirror

  # 使用代理访问 HuggingFace 官方
  python scripts/download_embedding_model.py --source hf --proxy http://127.0.0.1:7890

  # 自定义存储路径
  python scripts/download_embedding_model.py --output D:/models

  # 下载其他模型
  python scripts/download_embedding_model.py --model BAAI/bge-base-zh-v1.5
        """
    )

    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="模型名称 (默认: BAAI/bge-small-zh-v1.5)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="输出目录 (默认: models/embedding/)"
    )

    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "modelscope", "hf-mirror", "hf"],
        help="下载源 (默认: auto，自动选择)"
    )

    parser.add_argument(
        "--proxy",
        default=None,
        help="代理地址，如 http://127.0.0.1:7890"
    )

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "models", "embedding")
    else:
        output_dir = os.path.abspath(args.output)

    # 检查是否已下载
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"\n⚠ 检测到输出目录已存在: {output_dir}")
        print("  如果模型文件完整，可以跳过下载。")
        verify_download(output_dir)

        overwrite = input("\n是否重新下载？(y/N): ").strip().lower()
        if overwrite != 'y':
            print("已取消下载。")
            return

    # 执行下载
    success = download_embedding_model(
        model_name=args.model,
        output_dir=output_dir,
        source=args.source,
        proxy=args.proxy
    )

    if success:
        # 验证下载
        is_valid = verify_download(output_dir)

        if is_valid:
            print(f"\n{'=' * 60}")
            print("后续步骤:")
            print(f"{'=' * 60}")
            print(f"  1. 确认 settings.py 中的配置:")
            print(f"     EMBEDDING_MODEL_NAME = '{args.model}'")
            print(f"     EMBEDDING_DEVICE = 'cpu'")
            print(f"     EMBEDDING_DIMENSION = 512")
            print()
            print(f"  2. 启动应用时会自动加载本地模型")
            print(f"     模型路径: {output_dir}")
            print(f"{'=' * 60}")
        else:
            print("\n⚠ 模型文件不完整，请重试或手动下载。")
    else:
        print("\n" + "=" * 60)
        print("下载失败！")
        print("=" * 60)
        print("\n可能的解决方案:")
        print("  1. 安装 modelscope: pip install modelscope")
        print("  2. 使用代理: --proxy http://127.0.0.1:7890")
        print("  3. 手动下载:")
        print("     - 访问 https://www.modelscope.cn/models/BAAI/bge-small-zh-v1.5")
        print("     - 点击下载全部文件")
        print("     - 将文件放到: " + output_dir)
        print("  4. 或使用浏览器访问:")
        print("     - https://hf-mirror.com/BAAI/bge-small-zh-v1.5")
        print("     - 下载后放到: " + output_dir)


if __name__ == "__main__":
    main()