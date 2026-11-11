"""
PDF 解析统计日志模块。

为每个上传/更新的 PDF 单独创建一个日志文件夹，集中记录该 PDF 的解析情况：

    <PARSE_LOG_ROOT>/<PDF名>_<时间戳>/
        info.json         PDF 基本信息（文件名/总页数/文件大小/结构类型数量/解析耗时）
                          + 解析质量报告（errors/warnings 清单）
        table_001.md      表格解析片段（Markdown）
        flow_001.png      流程图图片（复用识别阶段裁剪图；无图时降级整页渲染并标注）
        flow_001_context.txt  流程图内容解析文本（规则识别，与图片同编号）
        flow_001_context_llm.txt  流程图内容解析文本（LLM 识别/LLM 复核）
        img_001.txt       图片 OCR 结果（OCR 失败时写入错误信息）
        metadata.json     向量存储元数据（逐切片一条记录，含全部元数据字段）

设计要点：
    1. 所有写入均 try/except 容错：日志失败只打印警告，绝不影响解析/入库主链路；
    2. 时间戳命名保证同名 PDF 重复上传时历史可追溯、互不覆盖；
    3. 结构片段（表格/流程图/图片）仅在结构化解析成功（doc.structure 非 None）时写入；
    4. 元数据日志在切片完成后由 text_splitter 调用导出，与向量库实际入库元数据一致。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


def get_log_root() -> str:
    """获取日志根目录（settings.PARSE_LOG_ROOT，相对路径基于进程工作区根）。"""
    try:
        from backend.config.settings import settings
        root = getattr(settings, "PARSE_LOG_ROOT", "") or "logs/解析日志"
    except Exception:
        root = "logs/解析日志"
    if not os.path.isabs(root):
        root = os.path.join(os.getcwd(), root)
    return root


def create_log_dir(pdf_path: str) -> str:
    """
    为单个 PDF 创建日志文件夹：<PARSE_LOG_ROOT>/<PDF名>_<YYYYMMDD_HHMMSS>/。

    每次上传/更新都会新建（时间戳后缀），同名 PDF 的历史日志互不覆盖。
    """
    stem, _ = os.path.splitext(os.path.basename(pdf_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(get_log_root(), f"{stem}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _write_json(path: str, payload: dict) -> None:
    """UTF-8 写 JSON（ensure_ascii=False 保留中文）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_jsonable(value):
    """把不可序列化对象降级为字符串，保证 json.dump 不抛异常。"""
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_pdf_parse_log(pdf_path: str, doc, parse_ms: float = 0.0) -> Optional[str]:
    """
    记录 PDF 解析日志（上传/更新时调用）。

    1. 无论结构化解析是否开启/成功，都写 info.json 基本信息；
    2. 结构化解析成功（doc.structure 非 None）时，追加结构类型数量、质量报告，
       并导出表格/流程图/图片等结构片段；
    3. 返回日志目录路径（供切片阶段导出向量元数据），失败返回 None。

    参数:
        pdf_path: PDF 文件路径
        doc: 已加载的 Document 对象（带 pages/metadata/id）
        parse_ms: 结构化解析耗时（毫秒，0 表示未执行）
    """
    log_dir = None
    try:
        log_dir = create_log_dir(pdf_path)
    except Exception as exc:
        print(f"[PARSELOG] 日志目录创建失败: {exc}")
        return None

    info = {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "doc_id": getattr(doc, "id", "") or "",
        "file_size_bytes": _get_file_size(pdf_path),
        "total_pages": int(getattr(doc, "pages", 0)
                           or (getattr(doc, "metadata", {}) or {}).get("pages", 0) or 0),
        "parsed_at": datetime.now().isoformat(),
        "structured_parsed": getattr(doc, "structure", None) is not None,
        "parse_duration_ms": round(parse_ms, 1),
        "structure_stats": {},
        "total_blocks": 0,
        "issues": {"errors": [], "warnings": [], "total": 0},
    }

    sd = getattr(doc, "structure", None)
    if sd is not None:
        try:
            # 结构类型数量：按块类型统计（含文本/标题等全部结构）
            stats: Dict[str, int] = {}
            for block in getattr(sd, "blocks", []) or []:
                btype = getattr(block, "type", "") or "unknown"
                stats[btype] = stats.get(btype, 0) + 1
            info["structure_stats"] = stats
            info["total_blocks"] = len(getattr(sd, "blocks", []) or [])
            info["toc_count"] = len(getattr(sd, "toc", []) or [])
            info["parser_version"] = getattr(sd, "parser_version", "")
            # 解析质量报告（errors/warnings 清单）
            report = getattr(sd, "report", None)
            if report is not None:
                for issue in getattr(report, "issues", []) or []:
                    item = {
                        "level": getattr(issue, "level", ""),
                        "code": getattr(issue, "code", ""),
                        "message": getattr(issue, "message", ""),
                        "page_no": getattr(issue, "page_no", None),
                        "block_id": getattr(issue, "block_id", None),
                        "detail": _safe_jsonable(getattr(issue, "detail", {})),
                    }
                    if item["level"] == "error":
                        info["issues"]["errors"].append(item)
                    elif item["level"] == "warning":
                        info["issues"]["warnings"].append(item)
                    else:
                        info["issues"].setdefault("info", []).append(item)
                info["issues"]["total"] = len(getattr(report, "issues", []) or [])
            # 结构片段：表格 md / 流程图 png / 图片 ocr txt
            write_structure_artifacts(log_dir, sd)
        except Exception as exc:
            print(f"[PARSELOG] 结构明细写入失败（不影响主流程）: {exc}")

    try:
        _write_json(os.path.join(log_dir, "info.json"), info)
    except Exception as exc:
        print(f"[PARSELOG] info.json 写入失败: {exc}")
    return log_dir


def write_structure_artifacts(log_dir: str, sd) -> None:
    """
    导出结构化解析的片段到日志目录（每块一个文件，统一编号）：
        table_00X.md        表格 Markdown 内容
        flow_00X.png        流程图裁剪图（无图时降级渲染整页并标注 _fallback）
        flow_00X_context.txt     流程图内容解析文本（规则识别）
        flow_00X_context_llm.txt 流程图内容解析文本（LLM 识别/LLM 复核）
        img_00X.txt         图片 OCR 结果（失败时写错误信息）
    """
    table_idx = flow_idx = img_idx = 0
    issues_by_block: Dict[str, List] = {}
    report = getattr(sd, "report", None)
    if report is not None:
        for issue in getattr(report, "issues", []) or []:
            bid = getattr(issue, "block_id", None)
            if bid:
                issues_by_block.setdefault(bid, []).append(issue)

    for block in getattr(sd, "blocks", []) or []:
        btype = getattr(block, "type", "")
        try:
            if btype == "table":
                table_idx += 1
                _write_utf8(
                    os.path.join(log_dir, f"table_{table_idx:03d}.md"),
                    getattr(block, "content", "") or "（表格内容为空）",
                )
            elif btype == "flowchart":
                flow_idx += 1
                prefix = f"flow_{flow_idx:03d}"
                _save_flow_image(log_dir, f"{prefix}.png", block, sd)
                _save_flow_context(log_dir, prefix, block)
            elif btype == "image":
                img_idx += 1
                _save_image_ocr(log_dir, f"img_{img_idx:03d}.txt", block, issues_by_block)
        except Exception as exc:
            print(f"[PARSELOG] 结构片段 {btype} 导出失败: {exc}")


def _write_utf8(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _get_file_size(pdf_path: str) -> int:
    try:
        return os.path.getsize(pdf_path)
    except OSError:
        return 0


def _save_flow_context(log_dir: str, prefix: str, block) -> None:
    """保存流程图内容解析文本（节点/箭头路径）。

    与 flow_00X.png 图片同编号：
        - 规则识别：flow_00X_context.txt
        - LLM 识别（source=llm）或 LLM 复核替换：flow_00X_context_llm.txt
    content 为空时写出占位说明，避免文件缺失。
    """
    data = getattr(block, "data", None) or {}
    is_llm = (getattr(block, "source", "") == "llm"
              or data.get("source_llm") is True
              or data.get("rechecked_by_llm") is True)
    name = f"{prefix}_context_llm.txt" if is_llm else f"{prefix}_context.txt"
    content = getattr(block, "content", "") or ""
    if not content.strip():
        title = str(data.get("title", "") or "").strip()
        content = f"（流程图内容为空）\n标题: {title}" if title else "（流程图内容为空）"
    _write_utf8(os.path.join(log_dir, name), content)


def _save_flow_image(log_dir: str, name: str, block, sd) -> None:
    """保存流程图图片：优先复制识别阶段裁剪图；无图时渲染整页并标注 _fallback。"""
    image_path = getattr(block, "image_path", "") or ""
    if image_path and os.path.isfile(image_path):
        dest = os.path.join(log_dir, name)
        import shutil
        shutil.copy2(image_path, dest)
        return
    # 降级：渲染流程图所在页整页图（文件名标注 _fallback + 页码）
    page_no = getattr(block, "page_no", None)
    if page_no is not None:
        pdf_path = getattr(sd, "file_path", "") or ""
        if pdf_path and os.path.isfile(pdf_path):
            rendered = _render_page_png(pdf_path, int(page_no))
            if rendered:
                dest = os.path.join(log_dir, f"{os.path.splitext(name)[0]}_page{int(page_no):02d}_fallback.png")
                with open(dest, "wb") as f:
                    f.write(rendered)
                return
    _write_utf8(
        os.path.join(log_dir, f"{os.path.splitext(name)[0]}_note.txt"),
        "流程图裁剪图与整页渲染均不可用，无法导出图片。",
    )


def _render_page_png(pdf_path: str, page_no: int) -> Optional[bytes]:
    """用 pymupdf 渲染指定页为 PNG（1 页起算），失败返回 None。"""
    try:
        import fitz  # pymupdf
        with fitz.open(pdf_path) as doc:
            if 0 < page_no <= len(doc):
                pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                return pix.tobytes("png")
    except Exception:
        pass
    return None


def _save_image_ocr(log_dir: str, name: str, block, issues_by_block: Dict[str, List]) -> None:
    """保存图片 OCR 结果到 txt；OCR 无文本/失败时写入错误信息便于跟踪。"""
    data = getattr(block, "data", {}) or {}
    ocr_text = str(data.get("ocr_text", "") or "").strip()
    lines = [
        f"图片块: {getattr(block, 'block_id', '')}",
        f"页码: {getattr(block, 'page_no', None)}",
        f"图片文件: {data.get('image_path', '') or ''}",
        "",
    ]
    if ocr_text:
        lines.append("OCR 识别结果:")
        lines.append("")
        lines.append(ocr_text)
    else:
        # OCR 失败/无文本：优先取该块在质量报告中的错误信息
        issues = issues_by_block.get(getattr(block, "block_id", ""), []) or []
        errors = [i for i in issues if getattr(i, "level", "") == "error"]
        details = errors or issues
        llm_desc = str(data.get("llm_description", "") or "").strip()
        if details:
            lines.append("OCR 未识别到文本，解析问题:")
            for issue in details:
                lines.append(f"  [{getattr(issue, 'level', '')}] {getattr(issue, 'code', '')}: "
                             f"{getattr(issue, 'message', '')}")
        elif llm_desc:
            lines.append("OCR 未识别到文本（图片语义描述）:")
            lines.append("")
            lines.append(llm_desc)
        else:
            lines.append("OCR 未识别到文本，且无解析错误详情。")
    _write_utf8(os.path.join(log_dir, name), "\n".join(lines) + "\n")


def write_chunk_metadata(log_dir: str, chunks) -> None:
    """
    导出向量存储元数据到 metadata.json（切片完成后调用）。

    每个切片一条记录：chunk_index / total_chunks / content_preview
    + 全部 metadata 字段（与向量库实际入库元数据一致，如 block_id、chunk_type、
    page_no、section_path、image_path、row_group、low_confidence 等）。
    """
    records = []
    for chunk in chunks:
        content = getattr(chunk, "content", "") or ""
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        record = {
            "chunk_index": getattr(chunk, "chunk_index", 0),
            "total_chunks": getattr(chunk, "total_chunks", len(chunks)),
            "content_length": len(content),
            "content_preview": content[:100],
        }
        record.update(metadata)
        records.append(_safe_jsonable(record))

    payload = {
        "exported_at": datetime.now().isoformat(),
        "chunk_count": len(records),
        "records": records,
    }
    _write_json(os.path.join(log_dir, "metadata.json"), payload)


# ===========================================================================
# Markdown 结构化解析日志（新增，md 与 pdf 共用 <PARSE_LOG_ROOT> 根目录）
# ===========================================================================

def write_md_parse_log(md_path: str, doc, parse_ms: float = 0.0) -> Optional[str]:
    """
    记录 Markdown 解析日志（上传/更新时调用）。

    目录结构（与 PDF 日志并列，仅当 MD_STRUCTURED_PARSING_ENABLED 开启才有结构信息）：
        <PARSE_LOG_ROOT>/<md名>_<时间戳>/
            info.json            基本信息 + 块级统计 + 图片清单
            images_inventory.txt 图片"抽检"清单（alt/来源/图意化方式/静态URL）
            metadata.json        向量存储元数据（切片完成后由 text_splitter 导出）

    返回日志目录路径，失败返回 None（绝不影响主链路）。
    """
    if not (getattr(doc, "file_type", "") in ("md", "markdown")):
        return None
    try:
        log_dir = create_log_dir(md_path)
    except Exception as exc:
        print(f"[PARSELOG][md] 日志目录创建失败: {exc}")
        return None

    info = _build_md_info(md_path, doc, parse_ms=parse_ms)
    try:
        _write_json(os.path.join(log_dir, "info.json"), info)
    except Exception as exc:
        print(f"[PARSELOG][md] info.json 写入失败: {exc}")
    try:
        _write_md_images_inventory(log_dir, doc)
    except Exception as exc:
        print(f"[PARSELOG][md] 图片清单写入失败: {exc}")
    return log_dir


def _build_md_info(md_path: str, doc, parse_ms: float = 0.0) -> dict:
    """构建 md info.json 内容（含结构/图片统计；未图意化时以解析期占位为准）。"""
    md_meta = getattr(doc, "metadata", {}) or {}
    info = {
        "file_name": os.path.basename(md_path),
        "file_path": md_path,
        "doc_id": getattr(doc, "id", "") or "",
        "file_type": "md",
        "file_size_bytes": _get_file_size(md_path),
        "chars": len(getattr(doc, "content", "") or ""),
        "parsed_at": datetime.now().isoformat(),
        "structured_parsed": False,
        "parse_duration_ms": round(parse_ms, 1),
        "structure_stats": {},
        "image_stats": {},
        "warnings": [],
    }
    sd = getattr(doc, "structure", None)
    if sd is not None and getattr(sd, "splitter_key", "") == "md":
        info["structured_parsed"] = True
        info["structure_stats"] = dict(getattr(sd, "stats", {}) or {})
        info["image_stats"] = dict(getattr(sd, "image_stats", {}) or {})
        info["warnings"] = list(getattr(sd, "warnings", []) or [])
    else:
        # 未开启/解析失败：保留 MarkdownLoader 的轻量统计
        for key in ("headings", "tables", "code_blocks", "images_found", "line_count"):
            if key in md_meta:
                info["structure_stats"][key] = md_meta[key]
    return info


def _write_md_images_inventory(log_dir: str, doc) -> None:
    """把 md 内图片清单写成 txt（抽检用途：来源/图意化方式/OCR/VL/静态URL）。"""
    sd = getattr(doc, "structure", None)
    if sd is None or getattr(sd, "splitter_key", "") != "md":
        return
    lines = ["# Markdown 图片抽检清单", ""]
    img_no = 0
    for block in getattr(sd, "blocks", []) or []:
        for img in getattr(block, "images", []) or []:
            img_no += 1
            lines.append(f"[图片 {img_no:02d}]")
            lines.append(f"  章节: {block.section_path or '（未标注）'}")
            lines.append(f"  来源: {getattr(img, 'src', '')}")
            lines.append(f"  类型: {getattr(img, 'src_type', '')}")
            lines.append(f"  图意化: {getattr(img, 'enhanced_by', 'none')}")
            if getattr(img, "ocr_text", ""):
                lines.append(f"  OCR: {img.ocr_text[:80]}")
            if getattr(img, "vl_description", ""):
                lines.append(f"  AI解读: {img.vl_description[:80]}")
            if getattr(img, "url", ""):
                lines.append(f"  静态URL: {img.url}")
            lines.append("")
    if not img_no:
        lines.append("（未发现图片）")
    _write_utf8(os.path.join(log_dir, "images_inventory.txt"), "\n".join(lines))


def finalize_md_parse_log(log_dir: str, doc) -> None:
    """
    切片/图意化完成后回写 info.json（图片统计、warnings 以最终态为准）。
    供 text_splitter md 分支切片成功后调用；失败仅打印警告。
    """
    if not log_dir:
        return
    try:
        info_path = os.path.join(log_dir, "info.json")
        if not os.path.isfile(info_path):
            return
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        sd = getattr(doc, "structure", None)
        if sd is not None and getattr(sd, "splitter_key", "") == "md":
            info["image_stats"] = dict(getattr(sd, "image_stats", {}) or {})
            info["warnings"] = list(getattr(sd, "warnings", []) or [])
        _write_json(info_path, info)
    except Exception as exc:
        print(f"[PARSELOG][md] info.json 回写失败（不影响主流程）: {exc}")
