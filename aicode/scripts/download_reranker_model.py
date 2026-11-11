"""
精排 (Reranker) 模型下载脚本 - Reranker Model Download Script

手动执行此脚本，交互式选择要下载的 BGE-Reranker 精排模型，
并自动检测当前硬件（CPU / 内存 / GPU / 磁盘），给出推荐。

支持三个版本：
    1. BAAI/bge-reranker-base   （轻量，CPU 首选）
    2. BAAI/bge-reranker-large  （大模型，英文精度高，适合 GPU）
    3. BAAI/bge-reranker-v2-m3  （官方最新，多语言/中文效果最好）

下载源（自动多源尝试）：
    1. ModelScope（国内推荐，速度快，无需代理）
    2. HuggingFace 镜像站 hf-mirror.com（国内可访问）
    3. HuggingFace 官方（需要代理）

使用方法:
    # 交互式选择模型（推荐）
    python scripts/download_reranker_model.py

    # 指定模型 + 指定下载源（非交互）
    python scripts/download_reranker_model.py --model base --source modelscope

    # 指定输出目录
    python scripts/download_reranker_model.py --output D:/models/reranker

    # 使用代理
    python scripts/download_reranker_model.py --proxy http://127.0.0.1:7890

可选参数:
    --model:  模型版本 (base/large/v2-m3)
    --output: 输出目录 (默认: models/reranker)
    --source: 下载源 (modelscope/hf-mirror/hf/auto)
    --proxy:  代理地址 (如 http://127.0.0.1:7890)
    --list:   仅展示模型对比和硬件推荐，不下载
"""

import os
import sys
import shutil
import time
import argparse
import platform
import subprocess

# =====================================================================
# 模型信息配置
# =====================================================================
# 各模型的元信息（大小、参数量为官方/社区估算值，实际以下载为准）
MODELS = {
    "base": {
        "hf_id": "BAAI/bge-reranker-base",
        "ms_id": "BAAI/bge-reranker-base",
        "name": "bge-reranker-base",
        "params": "约 278M",
        "size": "约 1.1 GB",
        "speed": "快（CPU 上约 2-4 秒/批）",
        "memory": "内存占用低（约 1.2-2 GB）",
        "pros": [
            "轻量高效，CPU 推理最快，首个请求延迟低",
            "内存/磁盘占用小，低配机器也能流畅运行",
            "与 large/v2-m3 同架构（XLM-RoBERTa base），部署最稳",
            "适合对响应速度敏感、机器资源紧张的内部系统",
        ],
        "cons": [
            "精度为三者最低（英文和中文均弱于 large/v2-m3）",
            "多语言/中文场景能力一般",
        ],
        "scenario": "低配 CPU（≤4 核）或内存 ≤8GB 的机器；对精度要求不高的日常检索",
    },
    "large": {
        "hf_id": "BAAI/bge-reranker-large",
        "ms_id": "BAAI/bge-reranker-large",
        "name": "bge-reranker-large",
        "params": "约 560M",
        "size": "约 2.2 GB",
        "speed": "慢（CPU 上约 10-20 秒/批，建议配合 GPU）",
        "memory": "内存占用高（约 3-5 GB）",
        "pros": [
            "英文检索精度高，强于 base",
            "适合 GPU 环境，精度与速度兼得",
            "成熟稳定，官方长期维护",
        ],
        "cons": [
            "CPU 推理很慢，首请求可能 15 秒+（容易超时降级）",
            "多语言能力不如 v2-m3（偏英文场景）",
            "内存/磁盘占用大",
        ],
        "scenario": "有 GPU（显存 ≥8GB）或英文为主的场景；对精度要求高且不在乎延迟",
    },
    "v2-m3": {
        "hf_id": "BAAI/bge-reranker-v2-m3",
        "ms_id": "BAAI/bge-reranker-v2-m3",
        "name": "bge-reranker-v2-m3",
        "params": "约 568M",
        "size": "约 2.2 GB",
        "speed": "慢（CPU 上约 10-20 秒/批，建议配合 GPU）",
        "memory": "内存占用高（约 3-5 GB）",
        "pros": [
            "官方最新一代精排模型，综合精度最高",
            "多语言能力极强（支持 100+ 语言），中文效果最好",
            "对中文私有化知识库检索提升最明显（本项目默认推荐）",
        ],
        "cons": [
            "CPU 推理慢，首请求延迟高（项目已通过 TOP_K=5 + 256 截断优化）",
            "内存/磁盘占用大",
        ],
        "scenario": "内存充足（≥16GB）的 CPU 机器或 GPU 机器；中文/多语言知识库检索（本项目默认）",
    },
}

# 推荐的优先级顺序（默认推荐排序，后续按硬件调整）
DEFAULT_RECOMMEND_ORDER = ["v2-m3", "large", "base"]


# =====================================================================
# 系统硬件检测
# =====================================================================
def get_system_info(output_dir: str) -> dict:
    """
    检测当前硬件环境：
    - CPU 核心数
    - 总内存（GB）
    - 是否 GPU（nvidia-smi / torch）
    - 磁盘剩余空间（GB）
    - 操作系统
    """
    info = {
        "os": platform.system(),
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb": _get_total_ram_gb(),
        "gpu": _detect_gpu(),
        "disk_free_gb": _get_disk_free_gb(output_dir),
    }
    return info


def _get_total_ram_gb() -> float:
    """获取总内存（GB），跨平台实现，不依赖第三方库"""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)

        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 / 1024  # kB -> GB

        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            ).stdout.strip()
            if out:
                return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 0.0  # 未知


def _detect_gpu() -> dict:
    """
    检测 GPU：
    优先通过 nvidia-smi 获取显存；其次尝试 torch.cuda。
    返回 {"available": bool, "name": str, "vram_gb": float}
    """
    gpu = {"available": False, "name": "未检测到（纯 CPU 环境）", "vram_gb": 0.0}

    # 方式一：nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                name, mem = lines[0].split(",")
                # mem 形如 "81920 MiB"
                try:
                    vram_mib = float(mem.replace("MiB", "").strip())
                except ValueError:
                    vram_mib = 0.0
                gpu = {
                    "available": True,
                    "name": name.strip(),
                    "vram_gb": round(vram_mib / 1024, 1),
                }
                return gpu
    except Exception:
        pass

    # 方式二：torch.cuda（不强制依赖 torch，安装了才检测）
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1),
            }
    except ImportError:
        pass
    except Exception:
        pass

    return gpu


def _get_disk_free_gb(output_dir: str) -> float:
    """获取目标磁盘剩余空间（GB）"""
    try:
        # 目标目录可能不存在，取其已存在的最近祖先目录
        check_dir = output_dir
        while check_dir and not os.path.exists(check_dir):
            parent = os.path.dirname(check_dir)
            if parent == check_dir:
                break
            check_dir = parent
        usage = shutil.disk_usage(check_dir)
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


# =====================================================================
# 推荐逻辑
# =====================================================================
def recommend_model(info: dict) -> str:
    """
    根据硬件环境推荐模型：
    - 有 GPU（显存 ≥ 8GB）→ v2-m3（中文/多语言）或 large（英文）
    - CPU + 内存 ≤ 8GB 或核数 ≤ 4 → base
    - CPU + 内存充足（≥16GB）→ v2-m3（默认，中文效果最好）
    """
    gpu = info["gpu"]["available"] and info["gpu"]["vram_gb"] >= 8
    ram = info["ram_gb"]
    cores = info["cpu_cores"]

    if gpu:
        return "v2-m3", "检测到 NVIDIA GPU（显存 {:.1f} GB），可以流畅运行大模型。当前场景为中文知识库检索，优先推荐 v2-m3（中文效果最佳）；若偏英文场景可选 large。".format(info["gpu"]["vram_gb"])

    if ram <= 0:
        return "base", "无法检测内存大小，保守起见推荐轻量的 base，避免内存不足。"

    if ram <= 8 or cores <= 4:
        return "base", "当前为 CPU 环境且资源较紧张（内存 {:.0f} GB / {} 核），推荐 base：CPU 推理快、内存占用低，避免首请求超时降级。".format(ram, cores)

    if ram < 16:
        return "base", "当前为 CPU 环境，内存 {:.0f} GB 中等偏紧，推荐 base（更稳更快）；若内存足够（≥16GB）再考虑 v2-m3。".format(ram)

    return "v2-m3", "当前 CPU 内存充足（{:.0f} GB / {} 核），可以运行 v2-m3。本项目面向中文知识库检索，v2-m3 多语言/中文效果最佳，是官方和本项目默认推荐。".format(ram, cores)


# =====================================================================
# 展示：硬件信息 + 模型对比
# =====================================================================
def print_system_info(info: dict):
    """打印硬件信息"""
    print("=" * 66)
    print("当前硬件环境")
    print("=" * 66)
    print(f"  操作系统      : {info['os']}")
    print(f"  CPU 核心数    : {info['cpu_cores']} 核")
    if info["ram_gb"] > 0:
        print(f"  物理内存      : {info['ram_gb']:.0f} GB")
    else:
        print(f"  物理内存      : 无法检测")
    gpu = info["gpu"]
    if gpu["available"]:
        print(f"  GPU           : {gpu['name']}（显存 {gpu['vram_gb']:.1f} GB）")
    else:
        print(f"  GPU           : {gpu['name']}（仅 CPU 推理）")
    if info["disk_free_gb"] > 0:
        print(f"  磁盘剩余      : {info['disk_free_gb']:.0f} GB")
    else:
        print(f"  磁盘剩余      : 无法检测")
    print("=" * 66)


def print_model_comparison():
    """打印三个模型的优缺点对比"""
    print()
    print("=" * 66)
    print("可选精排模型对比")
    print("=" * 66)
    for key in ["base", "large", "v2-m3"]:
        m = MODELS[key]
        print(f"\n【{m['name']}】({m['hf_id']})")
        print(f"  参数量        : {m['params']}")
        print(f"  磁盘占用      : {m['size']}")
        print(f"  CPU 推理速度  : {m['speed']}")
        print(f"  内存占用      : {m['memory']}")
        print("  优点:")
        for p in m["pros"]:
            print(f"    ✓ {p}")
        print("  缺点:")
        for c in m["cons"]:
            print(f"    ✗ {c}")
        print(f"  适用场景      : {m['scenario']}")
    print("=" * 66)


# =====================================================================
# 下载实现
# =====================================================================
def download_from_modelscope(model_key: str, output_dir: str) -> bool:
    """从 ModelScope 下载（国内推荐）"""
    model_id = MODELS[model_key]["ms_id"]
    print("\n" + "-" * 66)
    print("下载源 1: ModelScope（国内源，推荐）")
    print(f"  仓库名: {model_id}")
    print(f"  预计耗时: 2-5 分钟（视网速而定）")
    print("-" * 66)
    try:
        from modelscope import snapshot_download as ms_snapshot_download
        start = time.time()
        local_path = ms_snapshot_download(model_id=model_id, cache_dir=output_dir)
        elapsed = time.time() - start
        print(f"\n✓ ModelScope 下载完成，耗时 {elapsed:.1f} 秒")
        # ModelScope 可能把文件放在子目录，整理到 output_dir 根目录
        normalize_model_files(local_path, output_dir)
        return True
    except ImportError:
        print("  ✗ 未安装 modelscope 库，请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"  ✗ ModelScope 下载失败: {str(e)}")
        return False


def download_from_huggingface(model_key: str, output_dir: str, mirror: bool = True, proxy: str = None) -> bool:
    """从 HuggingFace（镜像站/官方）下载"""
    model_id = MODELS[model_key]["hf_id"]
    source_name = "HuggingFace 镜像站 hf-mirror.com" if mirror else "HuggingFace 官方"
    print("\n" + "-" * 66)
    print(f"下载源 2: {source_name}")
    print(f"  仓库名: {model_id}")
    print(f"  预计耗时: 3-10 分钟（视网速而定）")
    print("-" * 66)

    old_endpoint = os.environ.get("HF_ENDPOINT")
    old_proxy = (os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY"))
    try:
        if mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        else:
            os.environ.pop("HF_ENDPOINT", None)
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy

        from huggingface_hub import snapshot_download
        start = time.time()
        local_path = snapshot_download(repo_id=model_id, local_dir=output_dir)
        elapsed = time.time() - start
        print(f"\n✓ {source_name} 下载完成，耗时 {elapsed:.1f} 秒")
        return True
    except ImportError:
        print("  ✗ 未安装 huggingface_hub 库，请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  ✗ {source_name} 下载失败: {str(e)}")
        return False
    finally:
        # 恢复环境变量
        if old_endpoint is not None:
            os.environ["HF_ENDPOINT"] = old_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
        old_http, old_https = old_proxy
        if old_http:
            os.environ["HTTP_PROXY"] = old_http
        else:
            os.environ.pop("HTTP_PROXY", None)
        if old_https:
            os.environ["HTTPS_PROXY"] = old_https
        else:
            os.environ.pop("HTTPS_PROXY", None)


def normalize_model_files(src_dir: str, output_dir: str):
    """
    将下载的文件整理到 output_dir 根目录。

    ModelScope 的 snapshot_download 会把文件放在
    {cache_dir}/models--{org}--{model}/snapshots/{hash}/ 子目录，
    而 reranker_service.py 要求模型文件直接在目录根（config.json + 权重）。
    """
    if not src_dir or src_dir == output_dir:
        return
    src_dir = os.path.abspath(src_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(src_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    # 若 src 就是 output_dir 的子目录，将其内容上移到 output_dir 根
    try:
        is_sub = os.path.commonpath([src_dir, output_dir]) == output_dir and src_dir != output_dir
    except ValueError:
        is_sub = False

    if is_sub:
        # 找出真正的模型文件目录（可能是 src 或其下一层）
        src_root = _find_model_root(src_dir, output_dir)
        _copy_tree_contents(src_root, output_dir)
    else:
        _copy_tree_contents(src_dir, output_dir)


def _find_model_root(start_dir: str, stop_dir: str) -> str:
    """
    从 start_dir 向下找包含 config.json 的目录；
    找不到则返回 start_dir 本身。
    """
    current = start_dir
    depth = 0
    while depth < 5 and os.path.isdir(current):
        if os.path.exists(os.path.join(current, "config.json")):
            return current
        # 若当前目录只有一个子目录，继续下钻
        subdirs = [d for d in os.listdir(current)
                   if os.path.isdir(os.path.join(current, d)) and d != "cache"]
        if len(subdirs) == 1:
            current = os.path.join(current, subdirs[0])
            depth += 1
        else:
            break
    return start_dir


def _copy_tree_contents(src: str, dst: str):
    """复制 src 下的所有内容到 dst，跳过 dst 自身避免递归"""
    if not os.path.isdir(src):
        return
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    for item in os.listdir(src):
        src_item = os.path.join(src_abs, item)
        dst_item = os.path.join(dst_abs, item)
        # 跳过指向 dst 本身或已存在于 dst 的同名目录（防止嵌套拷贝）
        if os.path.abspath(src_item) == dst_abs:
            continue
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)
        elif os.path.isdir(src_item):
            if os.path.exists(dst_item):
                # 若 dst 中已存在同名目录且内容不同，先删除再复制
                if os.path.abspath(dst_item) != os.path.abspath(src_item):
                    shutil.rmtree(dst_item, ignore_errors=True)
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)


def verify_download(output_dir: str, model_key: str) -> bool:
    """验证模型文件是否完整"""
    m = MODELS[model_key]
    print("\n" + "=" * 66)
    print("验证模型文件完整性...")
    print("=" * 66)

    if not os.path.isdir(output_dir):
        print("  ✗ 输出目录不存在")
        return False

    required_files = ["config.json"]
    weight_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]
    missing = []

    for f in required_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
        else:
            missing.append(f)
            print(f"  ✗ {f} (缺失)")

    has_weights = False
    for f in weight_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
            has_weights = True
            break
    if not has_weights:
        missing.append("模型权重文件")
        print("  ✗ 模型权重文件 (缺失)")

    # 计算总大小
    total = 0
    for dirpath, _, filenames in os.walk(output_dir):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    print(f"\n  总大小: {total / 1024 / 1024:.1f} MB")

    if missing:
        print(f"\n⚠ 缺少必要文件: {', '.join(missing)}")
        print("  模型可能无法正常工作，请检查网络后重新下载。")
        return False

    print("\n✓ 模型文件验证通过！")
    print(f"  模型: {m['name']}")
    return True


def print_followup(model_key: str, output_dir: str):
    """下载完成后打印配置说明"""
    m = MODELS[model_key]
    print("\n" + "=" * 66)
    print("后续步骤")
    print("=" * 66)
    print(f"  已下载模型    : {m['name']}")
    print(f"  本地路径      : {output_dir}")
    print()
    print("  1. 确认 settings.py 中的精排配置:")
    print("     RERANKER_ENABLED    = True")
    print(f"     RERANKER_MODEL_NAME = '{m['hf_id']}'")
    print("     RERANKER_MODEL_PATH = 'models/reranker'")
    print("     RERANKER_DEVICE     = 'cpu'  (有 GPU 可改为 'cuda')")
    print()
    print("  2. 修改 RERANKER_MODEL_NAME 后重启后端服务")
    print("     程序将优先从本地 models/reranker 加载，不再联网下载")
    print("=" * 66)


# =====================================================================
# 交互选择
# =====================================================================
def _clean_input(text: str) -> str:
    """
    清洗用户输入：去除首尾空白、BOM 与 NUL 字符。

    兼容从文件/管道重定向输入时可能带上的 \ufeff、\x00 等不可见字符。
    """
    return text.strip().replace("\x00", "").replace("\ufeff", "").lower()


def select_model_interactively(info: dict) -> str:
    """交互式选择模型，返回模型 key（base/large/v2-m3）"""
    print_model_comparison()

    rec_key, rec_reason = recommend_model(info)
    rec = MODELS[rec_key]
    print("\n" + "=" * 66)
    print("推荐建议")
    print("=" * 66)
    print(f"  ★ 根据当前硬件，推荐: {rec['name']}")
    print(f"    {rec_reason}")
    print("=" * 66)

    print("\n请输入要下载的模型编号:")
    keys = ["base", "large", "v2-m3"]
    labels = {
        "base": f"1. {MODELS['base']['name']}（轻量 · CPU 首选）",
        "large": f"2. {MODELS['large']['name']}（大模型 · 英文精度高）",
        "v2-m3": f"3. {MODELS['v2-m3']['name']}（多语言 · 中文最佳 · 项目默认）",
    }
    for k in keys:
        print(f"     {labels[k]}")
    print("     q. 退出")

    while True:
        choice = _clean_input(input("\n请输入编号 [1/2/3，推荐 " + str(keys.index(rec_key) + 1) + "]: "))
        if choice == "q":
            print("已取消下载。")
            sys.exit(0)
        if choice in ("1", "2", "3"):
            return keys[int(choice) - 1]
        print("  输入无效，请输入 1、2、3 或 q。")


def select_source_interactively(model_key: str, output_dir: str) -> str:
    """交互式选择下载源，返回 'auto'/'modelscope'/'hf-mirror'/'hf'"""
    m = MODELS[model_key]
    print("\n" + "-" * 66)
    print(f"即将下载: {m['name']} -> {output_dir}")
    print("请选择下载源:")
    print("   1. ModelScope（国内源，推荐，速度快）")
    print("   2. HuggingFace 镜像站（hf-mirror.com，国内可访问）")
    print("   3. HuggingFace 官方（需要代理）")
    print("   4. 自动尝试（ModelScope → 镜像站 → 官方）")
    print("   q. 退出")
    while True:
        choice = _clean_input(input("\n请输入编号 [1/2/3/4]: "))
        if choice == "q":
            print("已取消下载。")
            sys.exit(0)
        if choice in ("1", "2", "3", "4"):
            return {"1": "modelscope", "2": "hf-mirror", "3": "hf", "4": "auto"}[choice]
        print("  输入无效，请输入 1、2、3、4 或 q。")


# =====================================================================
# 主流程
# =====================================================================
def run_download(model_key: str, output_dir: str, source: str, proxy: str = None) -> bool:
    """执行下载主流程"""
    os.makedirs(output_dir, exist_ok=True)

    if source == "modelscope":
        return download_from_modelscope(model_key, output_dir)
    if source == "hf-mirror":
        return download_from_huggingface(model_key, output_dir, mirror=True, proxy=proxy)
    if source == "hf":
        return download_from_huggingface(model_key, output_dir, mirror=False, proxy=proxy)
    if source == "auto":
        print("\n将按顺序尝试以下下载源：")
        print("  1. ModelScope（国内源）")
        print("  2. HuggingFace 镜像站")
        print("  3. HuggingFace 官方")
        if download_from_modelscope(model_key, output_dir):
            return True
        if download_from_huggingface(model_key, output_dir, mirror=True, proxy=proxy):
            return True
        if download_from_huggingface(model_key, output_dir, mirror=False, proxy=proxy):
            return True
        print("\n✗ 所有下载源都失败了！")
        print("  建议:")
        print("  1. 检查网络连接")
        print(f"  2. 使用代理: python scripts/download_reranker_model.py --model {model_key} --source hf --proxy http://127.0.0.1:7890")
        print("  3. 手动下载模型文件放入: " + output_dir)
        return False

    print(f"错误: 未知的下载源 '{source}'")
    return False


def _setup_stdout():
    """
    Windows 下重配置 stdout/stderr 为 UTF-8 并容错。

    默认 GBK 编码无法输出 ✓ ✗ ★ 等字符，会抛出 UnicodeEncodeError，
    此处统一降级为 UTF-8（控制台不支持时用 ? 替代，避免崩溃）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _setup_stdout()
    parser = argparse.ArgumentParser(
        description="下载 BGE-Reranker 精排模型（交互式选择版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式选择模型（推荐，含硬件检测与推荐）
  python scripts/download_reranker_model.py

  # 指定模型下载
  python scripts/download_reranker_model.py --model v2-m3

  # 使用 ModelScope 国内源
  python scripts/download_reranker_model.py --model base --source modelscope

  # 使用代理
  python scripts/download_reranker_model.py --proxy http://127.0.0.1:7890

  # 仅查看模型对比和推荐，不下载
  python scripts/download_reranker_model.py --list
        """
    )
    parser.add_argument("--model", choices=["base", "large", "v2-m3"], default=None,
                        help="模型版本 (base/large/v2-m3)，不指定则交互式选择")
    parser.add_argument("--output", default=None,
                        help="输出目录 (默认: models/reranker)")
    parser.add_argument("--source", choices=["auto", "modelscope", "hf-mirror", "hf"], default="auto",
                        help="下载源 (默认: auto，自动多源尝试)")
    parser.add_argument("--proxy", default=None,
                        help="代理地址，如 http://127.0.0.1:7890")
    parser.add_argument("--list", action="store_true",
                        help="仅展示模型对比和硬件推荐，不下载")
    args = parser.parse_args()

    # 默认输出目录：aicode/models/reranker
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "models", "reranker")
    else:
        output_dir = os.path.abspath(args.output)

    print()
    print("=" * 66)
    print("BGE-Reranker 精排模型下载工具")
    print("=" * 66)

    # 检测硬件并展示
    info = get_system_info(output_dir)
    print_system_info(info)

    # --list 模式：只展示对比和推荐
    if args.list:
        print_model_comparison()
        rec_key, rec_reason = recommend_model(info)
        print("\n" + "=" * 66)
        print("推荐建议")
        print("=" * 66)
        print(f"  ★ 根据当前硬件，推荐: {MODELS[rec_key]['name']}")
        print(f"    {rec_reason}")
        print("=" * 66)
        return

    # 选择模型（交互式或参数指定）
    if args.model:
        model_key = args.model
    else:
        model_key = select_model_interactively(info)

    # 确认目标目录是否已有模型
    has_existing = os.path.isdir(output_dir) and any(
        f in os.listdir(output_dir) for f in ("config.json", "pytorch_model.bin", "model.safetensors")
    )
    if has_existing:
        print(f"\n⚠ 检测到 {output_dir} 下已有模型文件。")
        overwrite = input("是否删除并重新下载？(y/N): ").strip().lower()
        if overwrite != "y":
            print("已取消下载。")
            return
        shutil.rmtree(output_dir, ignore_errors=True)

    # 选择下载源（交互式或参数指定）
    if args.source == "auto" and not args.model and args.proxy is None:
        source = select_source_interactively(model_key, output_dir)
    else:
        source = args.source

    # 执行下载
    success = run_download(model_key, output_dir, source, args.proxy)

    if success:
        if verify_download(output_dir, model_key):
            print_followup(model_key, output_dir)
        else:
            print("\n⚠ 模型文件不完整，请重试或手动下载。")
    else:
        print("\n" + "=" * 66)
        print("下载失败！")
        print("=" * 66)
        print("\n可能的解决方案:")
        print("  1. 安装依赖: pip install modelscope huggingface_hub")
        print("  2. 使用代理: python scripts/download_reranker_model.py --proxy http://127.0.0.1:7890")
        print("  3. 手动下载:")
        print("     - 访问 https://www.modelscope.cn/models/" + MODELS[model_key]["ms_id"])
        print("     - 点击下载全部文件，放入: " + output_dir)
        print("     - 或访问 https://hf-mirror.com/" + MODELS[model_key]["hf_id"])


if __name__ == "__main__":
    main()
