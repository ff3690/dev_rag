# -*- coding: utf-8 -*-
"""
markdown_image_service.py - Markdown 图片语义化 + 落盘 + 去重缓存
================================================================

对 MdStruct 中的 MdImageRef 执行图意化，模式（settings.MD_IMAGE_ENHANCE_MODE）：

    off  - 不 OCR/不调用视觉大模型，图片仅保留 alt（保底兼容值）
    ocr  - 仅内存 OCR（extract_text_from_image_bytes，不落临时文件）
    auto - 先 OCR；文本为空或过短(< MD_IMAGE_AUTO_VL_MIN_TEXT)再调视觉大模型
           （API Key 缺失/调用失败自动降级为 OCR/原文，不阻断入库）
    all  - OCR + 视觉大模型双写

落盘策略（内容寻址 + 文档目录双写）：
    <MD_IMAGE_STATIC_DIR>/_cache/<md5>.<ext>        全局去重缓存（复用 OCR/VL 结果）
    <MD_IMAGE_STATIC_DIR>/<doc_id>/img_<k>_<md5>.<ext>  文档级副本（删除文档时整目录清理）
    静态访问前缀固定为 /md_images/...（由 main.py 挂载同一目录）

健壮性约定：任一步失败只记 warning / 置 enhanced_by="none"，绝不抛出阻断入库。
"""

import base64
import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
from typing import Optional, Tuple

from .markdown_struct import MdImageRef, MdStruct

# 静态访问前缀（与 main.py 挂载点保持一致）
MD_IMAGE_URL_PREFIX = "/md_images"

_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+)?(;base64)?,(.*)$", re.S)
_HTTP_RE = re.compile(r"^https?://", re.I)

_MIME_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/gif": "gif", "image/webp": "webp", "image/bmp": "bmp",
    "image/tiff": "tiff",
}


# 图片图意化链路日志开关（MD_IMAGE_VERBOSE 环境变量，默认开；设 0/false/no 关闭）
MD_IMAGE_VERBOSE = os.getenv("MD_IMAGE_VERBOSE", "1").strip().lower() not in (
    "0", "false", "no",
)


def _iprint(*args, **kwargs) -> None:
    """图片图意化链路跟踪输出，由 MD_IMAGE_VERBOSE 控制。"""
    if MD_IMAGE_VERBOSE:
        print(*args, **kwargs)


def _settings():
    from backend.config.settings import settings
    return settings


def _abs_path(cfg_path: str) -> str:
    """配置里的目录路径 -> 绝对路径（相对路径视为相对 CWD）。"""
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(os.getcwd(), cfg_path)


def md_image_root() -> str:
    """静态托管根目录（物理路径）。"""
    return _abs_path(_settings().MD_IMAGE_STATIC_DIR)


def doc_image_dir(doc_id: str) -> str:
    """某文档图片目录（物理路径）。"""
    return os.path.join(md_image_root(), str(doc_id))


def cache_dir() -> str:
    """全局内容寻址缓存目录（物理路径）。"""
    return os.path.join(md_image_root(), "_cache")


def _cache_index_path() -> str:
    return os.path.join(cache_dir(), "index.json")


def _load_cache_index() -> dict:
    try:
        with open(_cache_index_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache_index(index: dict):
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        with open(_cache_index_path(), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
    except Exception:
        pass  # 缓存索引写失败不影响入库


# ---------------------------------------------------------------------------
# 图片来源解析
# ---------------------------------------------------------------------------

def _clean_src(src: str) -> str:
    """去 < > 包裹、标题引号与查询片段等噪音。"""
    s = src.strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    # 取第一个空白（去掉 markdown title 语法）
    s = s.split()[0] if s.split() else s
    # 去掉常见外链参数中的空引用
    s = s.split('"')[0].strip()
    return s


def _decode_data_uri(src: str) -> Tuple[Optional[bytes], str]:
    """
    解析 data:image/...;base64,xxx 数据 URI。
    返回 (bytes, ext)；非 base64/解析失败返回 (None, "")。
    """
    m = _DATA_URI_RE.match(src.strip())
    if not m:
        return None, ""
    mime, flag, payload = m.group(1), m.group(2), m.group(3)
    if flag != ";base64":
        return None, ""
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception:
        return None, ""
    ext = _MIME_EXT.get((mime or "").lower(), "png")
    return data, ext


def _guess_ext_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext if ext in {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"} else "png"


def _resolve_image_bytes(img: MdImageRef, md_dir: str) -> Tuple[Optional[bytes], str]:
    """
    把 MdImageRef.src 解析为 (图片字节, 扩展名)。

    来源优先级: base64 data URI > 相对路径（相对 md 所在目录）> 本地绝对路径(限工作区内) > 外链(受开关)
    失败返回 (None, "")，绝不抛出。
    """
    src = _clean_src(img.src)
    if not src:
        img.src_type = "missing"
        return None, ""

    # 1) base64 内嵌
    if src.startswith("data:"):
        data, ext = _decode_data_uri(src)
        img.src_type = "base64" if data is not None else "missing"
        return (data, ext) if data is not None else (None, "")

    # 2) 外链 URL
    if _HTTP_RE.match(src):
        img.src_type = "url"
        s = _settings()
        if not s.MD_IMAGE_URL_DOWNLOAD:
            return None, ""
        try:
            import httpx
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                r = client.get(src)
                if r.status_code != 200:
                    return None, ""
                ext = _guess_ext_from_path(urllib.parse.urlparse(src).path)
                return r.content, ext
        except Exception:
            return None, ""

    # 3) 相对/绝对本地路径
    rel = src.replace("\\", "/")
    rel = urllib.parse.unquote(rel)
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        # 绝对路径：仅允许指向工作区内文件，防越权读取
        target = os.path.abspath(rel)
        cwd = os.path.abspath(os.getcwd())
        try:
            in_ws = os.path.commonpath([target, cwd]) == cwd
        except Exception:
            in_ws = False
        if not in_ws:
            img.src_type = "missing"
            return None, ""
    else:
        target = os.path.abspath(os.path.join(md_dir or "", rel.lstrip("./")))

    if not os.path.isfile(target):
        img.src_type = "missing"
        return None, ""
    img.src_type = "path"
    try:
        with open(target, "rb") as f:
            data = f.read()
        return data, _guess_ext_from_path(target)
    except Exception:
        img.src_type = "missing"
        return None, ""


# ---------------------------------------------------------------------------
# OCR / 视觉大模型
# ---------------------------------------------------------------------------

_OCR_STATE_PRINTED = False


def _ocr_bytes(data: bytes) -> str:
    global _OCR_STATE_PRINTED
    try:
        from backend.rag import document_loader as _dl
        if not _OCR_STATE_PRINTED:
            _OCR_STATE_PRINTED = True
            _iprint(f"[MD图片] OCR 引擎状态: available={_dl.OCR_AVAILABLE} "
                    f"lang={getattr(_dl._ocr_processor, 'lang', None)}")
        return (_dl.extract_text_from_image_bytes(data) or "").strip()
    except Exception as exc:
        _iprint(f"[MD图片] ✗ OCR 调用异常: {type(exc).__name__}: {exc}")
        return ""


def _make_vision_client():
    from backend.llm.vision_client import VisionLLMClient
    return VisionLLMClient()  # 未配置 API Key 会抛 ValueError（上层降级）


def _vl_describe_bytes(data: bytes, ext: str) -> str:
    """调用视觉大模型描述图片字节；失败抛异常由上层降级。"""
    client = _make_vision_client()
    if hasattr(client, "describe_image_bytes"):
        return (client.describe_image_bytes(data, ext) or "").strip()
    return ""


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------

def _persist_image(data: bytes, ext: str, doc_id: str, img_index: int,
                   digest: str) -> Tuple[str, str]:
    """
    写缓存 + 文档副本。返回 (物理路径, 静态URL)。
    失败返回 ("", "")，不抛出。
    """
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        cache_file = os.path.join(cache_dir(), f"{digest}.{ext}")
        if not os.path.isfile(cache_file):
            with open(cache_file, "wb") as f:
                f.write(data)

        ddir = doc_image_dir(doc_id)
        os.makedirs(ddir, exist_ok=True)
        doc_file = os.path.join(ddir, f"img_{img_index}_{digest[:8]}.{ext}")
        with open(doc_file, "wb") as f:
            f.write(data)
        url = f"{MD_IMAGE_URL_PREFIX}/{doc_id}/img_{img_index}_{digest[:8]}.{ext}"
        return doc_file, url
    except Exception as exc:
        _iprint(f"[MD图片] ✗ 图片落盘失败（doc_id={doc_id!r} 序号={img_index}）: "
                f"{type(exc).__name__}: {exc}")
        return "", ""


def _exceeds_max_kb(data: bytes) -> bool:
    max_kb = int(getattr(_settings(), "MD_IMAGE_MAX_KB", 5120) or 5120)
    return max_kb > 0 and len(data) > max_kb * 1024


# ---------------------------------------------------------------------------
# 核心入口
# ---------------------------------------------------------------------------

def process_md_images(struct: MdStruct, doc_id: str = "",
                      md_dir: str = "") -> dict:
    """
    遍历 MdStruct 内全部图片引用，执行图意化 + 落盘（就地回填 MdImageRef）。

    参数:
        struct: MdStruct（解析产物）
        doc_id: 文档 ID（落盘目录与静态 URL 依据）
        md_dir: md 源文件所在目录（相对图片解析基准）

    返回:
        stats 字典（幂等：重复调用且 struct.images_processed=True 时直接返回已处理统计）
    """
    stats = {
        "processed": True, "images_found": 0, "enhanced_by_ocr": 0,
        "enhanced_by_vl": 0, "enhanced_by_explicit": 0, "none": 0,
        "ocr_chars_total": 0, "vl_calls": 0, "cache_hits": 0,
        "warnings": [], "cost_ms": 0,
    }
    if struct is None:
        return stats
    if getattr(struct, "images_processed", False):
        return dict(getattr(struct, "image_stats", stats))
    struct.images_processed = True

    t0 = time.time()
    settings = _settings()
    mode = (getattr(settings, "MD_IMAGE_ENHANCE_MODE", "auto") or "auto").strip().lower()
    auto_min_text = int(getattr(settings, "MD_IMAGE_AUTO_VL_MIN_TEXT", 20) or 20)
    cache_index = _load_cache_index()

    image_total = sum(len(block.images) for block in (struct.blocks or []) if block.images)
    stats["images_found"] = image_total

    _iprint(f"[MD图片] 开始图意化: doc_id={doc_id or '-'} 模式={mode} "
            f"auto_VL阈值={auto_min_text}字 图片数={image_total} "
            f"md目录={md_dir or '(空)'} "
            f"大小上限={getattr(settings, 'MD_IMAGE_MAX_KB', 5120)}KB")

    if mode == "off":
        _iprint("[MD图片] 模式=off：跳过 OCR/视觉大模型，图片仅保留 alt 文本")
        stats["none"] = image_total
        struct.image_stats = stats
        return stats

    idx = 0
    vision_failed = False
    for block in (struct.blocks or []):
        for img in (block.images or []):
            idx += 1
            # ---- 已带 doc_to_md 显式增强：跳过 OCR/VL（省成本），保留原文 ----
            if getattr(block, "has_explicit_enhance", False) or getattr(img, "explicit_enhance", False):
                img.enhanced_by = "explicit_text"
                stats["enhanced_by_explicit"] += 1
                # 显式增强场景仍尝试落盘图片（供 UI 展示原图），OCR/VL 不再调用
                data, ext = _resolve_image_bytes(img, md_dir)
                if data and not _exceeds_max_kb(data):
                    digest = hashlib.md5(data).hexdigest()
                    img.file_path, img.url = _persist_image(data, ext, doc_id, idx, digest)
                    _iprint(f"[MD图片][{idx}/{image_total}] 显式增强(原文已含描述) -> "
                            f"跳过 OCR/VL，仅落盘 src='{img.src}' url={img.url}")
                else:
                    _iprint(f"[MD图片][{idx}/{image_total}] 显式增强(原文已含描述) -> "
                            f"跳过 OCR/VL，图片不可用或超限 src='{img.src}' "
                            f"来源={img.src_type}")
                continue

            data, ext = _resolve_image_bytes(img, md_dir)
            if not data:
                reason = {
                    "missing": "路径/内容不存在或不可读",
                    "url": "外链未下载（MD_IMAGE_URL_DOWNLOAD=False 或下载失败）",
                }.get(img.src_type, "解析失败")
                stats["none"] += 1
                _iprint(f"[MD图片][{idx}/{image_total}] ✗ 跳过（{reason}）: "
                        f"src='{img.src}' 来源类型={img.src_type}")
                continue
            if _exceeds_max_kb(data):
                img.enhanced_by = "none"
                stats["none"] += 1
                stats["warnings"].append(f"img#{idx} 超过大小上限，已跳过: {len(data)//1024}KB")
                _iprint(f"[MD图片][{idx}/{image_total}] ⚠ 超过大小上限 "
                        f"{len(data)//1024}KB，跳过 OCR/VL: src='{img.src}'")
                continue

            digest = hashlib.md5(data).hexdigest()
            cached = cache_index.get(digest)
            if cached and isinstance(cached, dict):
                # 缓存命中：复用 OCR/VL 结果，仍写文档副本保证 URL 可用
                img.ocr_text = (cached.get("ocr_text") or "").strip()
                img.vl_description = (cached.get("vl_description") or "").strip()
                img.enhanced_by = cached.get("enhanced_by") or "ocr"
                stats["cache_hits"] += 1
                img.file_path, img.url = _persist_image(data, ext, doc_id, idx, digest)
                _iprint(f"[MD图片][{idx}/{image_total}] 缓存命中 md5={digest[:8]} "
                        f"enhanced_by={img.enhanced_by} OCR={len(img.ocr_text)}字 "
                        f"VL={len(img.vl_description)}字 url={img.url}")
                if img.ocr_text:
                    stats["ocr_chars_total"] += len(img.ocr_text)
                if img.enhanced_by == "vl":
                    stats["enhanced_by_vl"] += 1
                elif img.enhanced_by == "ocr":
                    stats["enhanced_by_ocr"] += 1
                else:
                    stats["none"] += 1
                continue

            # ---- 未命中缓存：执行 OCR（auto/all/ocr）----
            _iprint(f"[MD图片][{idx}/{image_total}] 新图片: src='{img.src}' "
                    f"来源={img.src_type} 大小={len(data)/1024:.1f}KB md5={digest[:8]}")
            ocr_text = _ocr_bytes(data) if mode in ("auto", "all", "ocr") else ""
            img.ocr_text = ocr_text
            if ocr_text:
                _iprint(f"[MD图片][{idx}/{image_total}] OCR ✓ {len(ocr_text)} 字: "
                        f"{ocr_text[:60]!r}")
            else:
                _iprint(f"[MD图片][{idx}/{image_total}] OCR 未识别出文本（0 字）")
            entry = {"ext": ext, "ocr_text": ocr_text, "vl_description": "",
                     "enhanced_by": "ocr" if ocr_text else "none"}

            # ---- VL 判定（auto: OCR 空/过短才调用；all: 必调）----
            need_vl = mode == "all" or (mode == "auto" and len(ocr_text) < auto_min_text)
            vl_reason = ("mode=all 强制双写" if mode == "all"
                         else f"OCR 文本过短({len(ocr_text)}字 < {auto_min_text}字)")
            if need_vl and not vision_failed:
                _iprint(f"[MD图片][{idx}/{image_total}] 调用视觉大模型（{vl_reason}）: "
                        f"model={getattr(settings, 'STRUCTURED_PARSER_FLOW_LLM_MODEL', '-')}")
                _vl_t0 = time.time()
                try:
                    desc = _vl_describe_bytes(data, ext)
                    img.vl_description = desc
                    _vl_ms = int((time.time() - _vl_t0) * 1000)
                    if desc:
                        entry["vl_description"] = desc
                        entry["enhanced_by"] = "vl"
                        stats["vl_calls"] += 1
                        _iprint(f"[MD图片][{idx}/{image_total}] 视觉大模型 ✓ {len(desc)} 字 "
                                f"耗时={_vl_ms}ms: {desc[:60]!r}")
                    else:
                        _iprint(f"[MD图片][{idx}/{image_total}] 视觉大模型返回空描述 "
                                f"（耗时={_vl_ms}ms）")
                except Exception as exc:
                    vision_failed = True  # Key 缺失/网络故障：降级 OCR，不再重试
                    stats["warnings"].append(f"视觉大模型调用失败（降级 OCR）: {exc}")
                    _iprint(f"[MD图片][{idx}/{image_total}] ✗ 视觉大模型失败，"
                            f"后续图片仅用 OCR: {type(exc).__name__}: {exc}")
            elif need_vl and vision_failed:
                _iprint(f"[MD图片][{idx}/{image_total}] 视觉模型已标记不可用，跳过 VL（仅 OCR）")
            else:
                _iprint(f"[MD图片][{idx}/{image_total}] OCR 文本充足，无需 VL "
                        f"-> enhanced_by=ocr")

            if ocr_text:
                stats["ocr_chars_total"] += len(ocr_text)
            if entry["enhanced_by"] == "vl":
                stats["enhanced_by_vl"] += 1
            elif entry["enhanced_by"] == "ocr":
                stats["enhanced_by_ocr"] += 1
            else:
                stats["none"] += 1
            img.enhanced_by = entry["enhanced_by"]

            # ---- 缓存索引（内容寻址，跨文档复用）----
            cache_index[digest] = entry
            img.file_path, img.url = _persist_image(data, ext, doc_id, idx, digest)

    stats["cost_ms"] = int((time.time() - t0) * 1000)
    struct.image_stats = stats
    _save_cache_index(cache_index)
    _iprint(f"[MD图片] 汇总: 发现={stats['images_found']} OCR={stats['enhanced_by_ocr']} "
            f"VL={stats['enhanced_by_vl']} 显式={stats['enhanced_by_explicit']} "
            f"未处理={stats['none']} 缓存命中={stats['cache_hits']} "
            f"OCR字符={stats['ocr_chars_total']} VL调用={stats['vl_calls']} "
            f"耗时={stats['cost_ms']}ms")
    for _w in stats["warnings"]:
        _iprint(f"[MD图片] ⚠ {_w}")
    return stats


def enhance_document_images(document) -> dict:
    """
    对带 MdStruct 的 Document 执行图片图意化（供切片前统一调用）。

    要求 document.structure 为 MdStruct 且 document.id 已生成
    （DocumentLoader 解析期仅建结构，图意化发生在 id 分配后的切片阶段）。
    返回 stats；任何异常返回空 stats（不阻断切片）。
    """
    structure = getattr(document, "structure", None)
    if not isinstance(structure, MdStruct):
        _iprint(f"[MD图片] 跳过图意化：document.structure 不是 MdStruct"
                f"（type={type(structure).__name__}，常见原因是 "
                f"MD_STRUCTURED_PARSING_ENABLED=False）: "
                f"file={getattr(document, 'file_path', '')}")
        return {}
    try:
        md_dir = os.path.dirname(document.file_path) if getattr(document, "file_path", "") else ""
        doc_id = getattr(document, "id", "") or structure.doc_id
        _iprint(f"[MD图片] 触发图意化: doc_id={doc_id} "
                f"file={getattr(document, 'file_path', '')}")
        return process_md_images(structure, doc_id=doc_id, md_dir=md_dir)
    except Exception as exc:
        _iprint(f"[MD图片] ✗ 图意化异常（不影响切片）: {type(exc).__name__}: {exc}")
        structure.warnings = list(getattr(structure, "warnings", [])) + [
            f"图片图意化失败（不影响切片）: {exc}"]
        return {"processed": False, "images_found": 0, "warnings": structure.warnings}


def render_document_content(document) -> str:
    """把 MdStruct 渲染为纯文本（含图意文本），作为 document.content。"""
    from .markdown_text_templates import render_struct_to_text
    structure = getattr(document, "structure", None)
    if not isinstance(structure, MdStruct):
        return getattr(document, "content", "") or ""
    return render_struct_to_text(structure)


def cleanup_doc_images(doc_id: str) -> bool:
    """删除某文档的图片目录（删除文档/清除时调用）；不存在返回 False。"""
    if not doc_id:
        return False
    ddir = doc_image_dir(doc_id)
    try:
        if os.path.isdir(ddir):
            shutil.rmtree(ddir, ignore_errors=True)
            return True
    except Exception:
        pass
    return False


def cleanup_md_image_root() -> None:
    """清空 md 图片根目录（含缓存）；失败不抛出。"""
    try:
        root = md_image_root()
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
