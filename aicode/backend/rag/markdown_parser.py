# -*- coding: utf-8 -*-
"""
markdown_parser.py - Markdown 块级结构化解析器（纯 Python 状态机）
==================================================================

设计决策：
    1. 零第三方依赖（不引入 markdown-it-py），仅做"块级"切分：
       heading / 围栏代码 / GFM 表格 / 独立图片 / 段落（含列表与引用）
       等，不做 inline 深度解析；
    2. 解析错误绝不抛出（任何异常由外层捕获降级旧链路）；
    3. 图片增强检测：图片块后紧邻 doc_to_md 增强产物（含"图片语义增强"
       等关键词的段落/引用、或 mermaid 围栏）时置 has_explicit_enhance=True，
       图意化阶段据此跳过 OCR/VL（避免重复付费）。
"""

import os
import re
from typing import List, Tuple

from .markdown_struct import MdBlock, MdImageRef, MdStruct

# 结构跟踪日志开关（MD_PARSER_VERBOSE 环境变量，默认开；设 0/false/no 关闭）
MD_PARSER_VERBOSE = os.getenv("MD_PARSER_VERBOSE", "1").strip().lower() not in (
    "0", "false", "no",
)


def _pprint(*args, **kwargs) -> None:
    """md 解析结构跟踪输出统一入口，由 MD_PARSER_VERBOSE 控制。"""
    if MD_PARSER_VERBOSE:
        print(*args, **kwargs)

# ---------------------------------------------------------------------------
# 正则（编译一次提升性能）
# ---------------------------------------------------------------------------
_RE_FENCE_START = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+-]*)\s*$")
_RE_HEADING = re.compile(r"^(#{1,6})\s*(.*?)\s*#*\s*$")
_RE_IMAGE_ONLY = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]*)\)\s*$")
_RE_INLINE_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")
_RE_CHECKBOX = re.compile(r"^\s*[-*+]\s*\[([ xX])\]\s*(.*)$")
_RE_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_RE_HTML_COMMENT_FULL = re.compile(r"<!--(.*?)-->", re.S)
_RE_TABLE_DELIM = re.compile(r"^[\s|:|-]+$")

_ENHANCE_MARKERS = (
    "图片语义增强", "语义增强", "GLM-4V", "图表类型", "AI解读", "AI 解读",
)


def _is_table_delimiter(line: str) -> bool:
    """GFM 表格分隔行：仅含 | : - 与空白，且同时含 | 和 -。"""
    return "|" in line and "-" in line and bool(_RE_TABLE_DELIM.match(line.strip()))


def _split_table_row(line: str) -> List[str]:
    """按未转义竖线切分表格行，去首尾典型空列。"""
    parts = []
    buf = ""
    for ch in line:
        if ch == "|":
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [c.strip() for c in parts]


class _Parser:
    """内部状态机：把 md 文本解析成 MdStruct。"""

    def __init__(self, text: str, file_path: str = ""):
        self.file_path = file_path
        self._text = text
        self._lines = text.split("\n")
        self.blocks: List[MdBlock] = []
        self.warnings: List[str] = []
        self._img_counter = 0
        # 章节栈 [(level, title)]
        self._section_stack: List[Tuple[int, str]] = []
        # 行起始偏移表（溯源用）
        self._line_offsets = [0]
        acc = 0
        for ln in self._lines:
            acc += len(ln) + 1
            self._line_offsets.append(acc)
        # 段落缓冲
        self._para_lines: List[str] = []
        self._para_pos = 0

    # ---------------- 章节 ----------------
    def _current_section(self) -> str:
        return "/".join(t for _lv, t in self._section_stack)

    def _push_heading(self, level: int, title: str):
        while self._section_stack and self._section_stack[-1][0] >= level:
            self._section_stack.pop()
        self._section_stack.append((level, title))

    # ---------------- 辅助 ----------------
    def _line_start(self, line_no: int) -> int:
        if 0 <= line_no < len(self._line_offsets):
            return self._line_offsets[line_no]
        return 0

    def _line_end(self, line_no: int) -> int:
        if 0 <= line_no < len(self._lines):
            return self._line_start(line_no) + len(self._lines[line_no]) + 1
        return self._line_start(line_no)

    # ---------------- 结构跟踪 ----------------
    def _append_block(self, block: MdBlock) -> None:
        """统一登记块并打印结构跟踪日志（MD_PARSER_VERBOSE 控制）。"""
        self.blocks.append(block)
        _pprint(f"[MD解析] 块#{len(self.blocks)} type={block.type} "
                f"行={block.pos + 1} 章节='{block.section_path or '-'}' "
                f"{self._describe_block(block)}")

    @staticmethod
    def _describe_block(block: MdBlock) -> str:
        """块的单行摘要（用于结构跟踪日志）。"""
        if block.type == "heading":
            return f"L{block.level} 标题='{block.content}'"
        if block.type == "table":
            return (f"列={len(block.header)} 数据行={len(block.rows)} "
                    f"表头={block.header} 标题='{block.caption or '-'}'")
        if block.type == "code":
            return f"lang={block.language or '-'} 字符={len(block.content)}"
        if block.type == "image":
            img = block.images[0] if block.images else None
            return (f"src='{img.src if img else ''}' alt='{block.content}' "
                    f"显式增强={bool(block.has_explicit_enhance)}")
        return (f"字符={len(block.content)} 图片={len(block.images)} "
                f"摘要='{(block.content or '')[:40]}'")

    def _is_comment_line(self, raw: str) -> bool:
        return raw.strip().startswith("<!--")

    def _extract_comment_block(self, i: int) -> Tuple[str, int]:
        """读取 (可能跨行的) HTML 注释，返回 (注释内文, 结束行索引+1)。"""
        buf: List[str] = []
        j = i
        while j < len(self._lines):
            buf.append(self._lines[j])
            if "-->" in self._lines[j]:
                j += 1
                break
            j += 1
        m = _RE_HTML_COMMENT_FULL.search("\n".join(buf))
        return (m.group(1).strip() if m else ""), j

    # ---------------- 段落 ----------------
    def _flush_paragraph(self, end_line: int = 0):
        """把缓冲的连续正文行收成一个 paragraph/list/blockquote 块。"""
        if not self._para_lines:
            return
        start_line = self._para_pos
        images: List[MdImageRef] = []
        cleaned: List[str] = []
        has_quote = False
        has_checkbox = False
        has_list = False
        for ln in self._para_lines:
            mq = _RE_BLOCKQUOTE.match(ln)
            if mq:
                has_quote = True
                ln = mq.group(1)
            mc = _RE_CHECKBOX.match(ln)
            if mc:
                has_checkbox = True
                text = f"[{mc.group(1).lower()}] {mc.group(2).strip()}"
                text, imgs = self._extract_inline_images(text)
                images.extend(imgs)
                cleaned.append(text.strip())
                continue
            ml = _RE_LIST_ITEM.match(ln)
            if ml:
                has_list = True
                text, imgs = self._extract_inline_images(ml.group(1).strip())
                images.extend(imgs)
                cleaned.append(text.strip())
                continue
            line, imgs = self._extract_inline_images(ln)
            images.extend(imgs)
            cleaned.append(line.strip())

        if has_checkbox:
            btype = "list"
        elif has_quote:
            btype = "blockquote"
        elif has_list:
            btype = "list"
        else:
            btype = "paragraph"

        content = "\n".join(c for c in cleaned if c).strip()
        if not content and not images:
            self._para_lines = []
            return

        block = MdBlock(
            type=btype,
            content=content,
            section_path=self._current_section(),
            images=images,
            pos=start_line,
            start_offset=self._line_start(start_line),
            end_offset=self._line_end(max(end_line, start_line)),
        )
        self._append_block(block)
        self._para_lines = []
        self._para_pos = 0

    def _extract_inline_images(self, line: str) -> Tuple[str, List[MdImageRef]]:
        """剥离行内图片语法注册 MdImageRef；alt 文本留原位供上下文阅读。"""
        refs: List[MdImageRef] = []
        counter = self._img_counter

        def _repl(m):
            nonlocal counter
            refs.append(MdImageRef(index=counter, alt=m.group(1).strip(),
                                   src=m.group(2).strip()))
            counter += 1
            return m.group(1).strip()  # alt 占位保留

        line = _RE_INLINE_IMAGE.sub(_repl, line)
        self._img_counter = counter
        return line, refs

    # ---------------- 图片独立块 ----------------
    def _make_image_block(self, alt: str, src: str, line_no: int) -> MdBlock:
        img = MdImageRef(index=self._img_counter, alt=alt.strip(), src=src.strip())
        self._img_counter += 1
        return MdBlock(
            type="image", content=alt.strip(), section_path=self._current_section(),
            images=[img], pos=line_no,
            start_offset=self._line_start(line_no), end_offset=self._line_end(line_no),
        )

    # ---------------- 主流程 ----------------
    def parse(self) -> MdStruct:
        n = len(self._lines)
        i = 0
        fence_marker = ""
        fence_lang = ""
        fence_lines: List[str] = []
        fence_start = 0
        fence_offset = 0

        while i < n:
            raw = self._lines[i]
            stripped = raw.strip()

            # 1) 围栏开始
            mf = _RE_FENCE_START.match(raw)
            if mf and not fence_marker:
                self._flush_paragraph(end_line=i - 1 if i > 0 else 0)
                fence_marker = mf.group(1)[0]
                fence_lang = (mf.group(2) or "").strip()
                fence_lines = []
                fence_start = i
                fence_offset = self._line_start(i)
                i += 1
                continue

            # 2) 围栏结束
            if fence_marker and stripped.startswith(fence_marker * 3):
                code = "\n".join(fence_lines)
                self._append_block(MdBlock(
                    type="code", content=code, language=fence_lang,
                    section_path=self._current_section(), pos=fence_start,
                    start_offset=fence_offset, end_offset=self._line_end(i),
                ))
                fence_marker = ""
                fence_lang = ""
                fence_lines = []
                i += 1
                continue

            # 3) 围栏内容
            if fence_marker:
                fence_lines.append(raw)
                i += 1
                continue

            # 4) 空行 -> 段落结束
            if not stripped:
                self._flush_paragraph(end_line=i - 1 if i > 0 else 0)
                i += 1
                continue

            # 5) HTML 注释：识别增强标记后跳过（脚手架不进入向量文本）
            if self._is_comment_line(raw):
                content, i = self._extract_comment_block(i)
                if any(mk in content for mk in _ENHANCE_MARKERS):
                    if self.blocks and self.blocks[-1].type == "image":
                        self.blocks[-1].has_explicit_enhance = True
                continue

            # 6) ATX 标题
            mh = _RE_HEADING.match(raw)
            if mh:
                self._flush_paragraph(end_line=i - 1 if i > 0 else 0)
                title = mh.group(2).strip()
                level = len(mh.group(1))
                self._push_heading(level, title)
                self._append_block(MdBlock(
                    type="heading", level=level, content=title,
                    section_path=self._current_section(), pos=i,
                    start_offset=self._line_start(i), end_offset=self._line_end(i),
                ))
                i += 1
                continue

            # 7) 独立图片行
            mi = _RE_IMAGE_ONLY.match(raw)
            if mi:
                self._flush_paragraph(end_line=i - 1 if i > 0 else 0)
                self._append_block(self._make_image_block(mi.group(1), mi.group(2), i))
                i += 1
                continue

            # 8) GFM 表格：本行含 | 且下一行为分隔行
            if "|" in raw and i + 1 < n and _is_table_delimiter(self._lines[i + 1]):
                self._flush_paragraph(end_line=i - 1 if i > 0 else 0)
                header = _split_table_row(raw)
                row_i = i + 2
                rows: List[List[str]] = []
                while row_i < n:
                    lr = self._lines[row_i]
                    if not lr.strip() or "|" not in lr or _RE_FENCE_START.match(lr):
                        break
                    rows.append(_split_table_row(lr))
                    row_i += 1
                # 无数据行时，把表头当唯一数据行（分隔行后即文末的极端情况）
                end_line = max(row_i - 1, i)
                self._append_block(MdBlock(
                    type="table", header=header, rows=rows,
                    section_path=self._current_section(), pos=i,
                    start_offset=self._line_start(i), end_offset=self._line_end(end_line),
                ))
                i = row_i
                continue

            # 9) 其他行 -> 段落缓冲
            if not self._para_lines:
                self._para_pos = i
            self._para_lines.append(raw)
            i += 1

        # 收尾：未闭合围栏 / 段落
        if fence_marker:
            self._append_block(MdBlock(
                type="code", content="\n".join(fence_lines), language=fence_lang,
                section_path=self._current_section(), pos=fence_start,
                start_offset=fence_offset,
                end_offset=fence_offset + len("\n".join(fence_lines)) + 1,
            ))
        self._flush_paragraph(end_line=n - 1)

        _before = len(self.blocks)
        self._post_process_tables()
        self._post_detect_enhancement()
        _enhanced = sum(1 for b in self.blocks
                        if b.type == "image" and b.has_explicit_enhance)
        _pprint(f"[MD解析] 后处理: 表格说明合并 {_before - len(self.blocks)} 处；"
                f"doc_to_md 显式增强图片 {_enhanced} 个（将跳过 OCR/VL）")
        return MdStruct(
            file_path=self.file_path, blocks=self.blocks,
            stats=self._compute_stats(), warnings=self.warnings,
        )

    # ---------------- 后处理 ----------------
    def _post_process_tables(self):
        """表格前紧邻的短段落 -> 表格说明（leading）/ 标题（caption），避免散落成小块。"""
        out: List[MdBlock] = []
        for blk in self.blocks:
            if blk.type == "table" and out:
                prev = out[-1]
                if prev.type in ("paragraph", "blockquote") and not prev.images:
                    txt = prev.content.strip()
                    if txt and len(txt) <= 200 and (blk.pos - prev.pos) <= 4:
                        blk.leading_text = txt
                        low = txt.lower()
                        if txt[:1] in ("表", "图", "附") or low.startswith("table"):
                            blk.caption = txt[:60]
                        out.pop()
            out.append(blk)
        self.blocks = out

    def _post_detect_enhancement(self):
        """
        图片块紧邻增强产物（前/后 ≤1 个相邻块含关键词段/引用，或 mermaid 围栏）
        → 置 has_explicit_enhance=True，图意化阶段跳过 OCR/VL（避免重复付费）。
        """
        n = len(self.blocks)
        for idx, blk in enumerate(self.blocks):
            if blk.type != "image" or blk.has_explicit_enhance:
                continue
            neighbors = []
            if idx + 1 < n:
                neighbors.append(self.blocks[idx + 1])
            if idx - 1 >= 0:
                neighbors.append(self.blocks[idx - 1])
            for nb in neighbors:
                if nb.type == "code" and (nb.language or "").lower() == "mermaid":
                    blk.has_explicit_enhance = True
                    break
                if nb.type in ("paragraph", "blockquote", "list"):
                    if any(mk in nb.content for mk in _ENHANCE_MARKERS):
                        blk.has_explicit_enhance = True
                        break

    def _compute_stats(self) -> dict:
        stats = {
            "headings": 0, "paragraphs": 0, "tables": 0, "code_blocks": 0,
            "images_found": 0, "images_explicit_enhance": 0, "lines": len(self._lines),
        }
        for blk in self.blocks:
            if blk.type == "heading":
                stats["headings"] += 1
            elif blk.type == "paragraph":
                stats["paragraphs"] += 1
            elif blk.type == "table":
                stats["tables"] += 1
            elif blk.type == "code":
                stats["code_blocks"] += 1
            if blk.images:
                stats["images_found"] += len(blk.images)
            if blk.type == "image" and blk.has_explicit_enhance:
                stats["images_explicit_enhance"] += 1
        return stats


def parse_markdown(text: str, file_path: str = "") -> MdStruct:
    """
    Markdown 块级解析统一入口。

    参数:
        text: md 原始文本
        file_path: 源文件路径（作为相对图片的解析基准目录）

    返回:
        MdStruct（异常时返回空结构，调用方据此降级旧链路）。
        本函数保证不抛出异常。
    """
    if not text:
        _pprint(f"[MD解析] 空文本，跳过块级解析: file={file_path}")
        return MdStruct(file_path=file_path, stats={"lines": 0})
    try:
        line_count = text.count("\n") + 1
        _pprint(f"[MD解析] 开始: file={os.path.basename(file_path) or '(text)'} 行={line_count}")
        struct = _Parser(text, file_path).parse()
        _pprint(f"[MD解析] 完成: 块={len(struct.blocks)} 统计={struct.stats} "
                f"warnings={struct.warnings}")
        return struct
    except Exception as exc:  # noqa: BLE001 - 解析绝不阻断入库
        _pprint(f"[MD解析] ✗ 解析器异常（已降级为空结构）: {exc}")
        return MdStruct(
            file_path=file_path,
            stats={"lines": text.count("\n") + 1, "parse_error": str(exc)[:200]},
            warnings=[f"解析器异常（已降级为空结构）: {exc}"],
        )
