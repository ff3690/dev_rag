"""
文本分割器模块 - Text Splitter Module

该模块提供多种文本分割策略，用于将长文档切分为适合向量化的小块。

设计决策：
1. 采用策略模式，定义BaseSplitter抽象基类，便于扩展新的分割策略
2. 默认使用自研语义分割策略，适用于大多数文本类型
3. 支持语义分割（按段落）、递归字符分割、代码感知分割（按函数/类）
4. 集成 LangChain 分割器（递归字符 / Token / Markdown / 语义），通过策略切换
5. 每个Chunk记录位置信息，便于后续溯源和定位

LangChain 集成说明：
- 采用惰性导入，LangChain 为可选依赖，未安装时自动降级到自研分割器
- 通过适配器模式将 LangChain Document 转换为项目 TextChunk，保持接口兼容
- 通过桥接类解决 LlamaIndex 与 LangChain Embedding 接口差异

日志设计说明：
- 使用标准 logging 模块，模块级 logger（name 自动带上模块/包路径）
- 日志格式包含：时间戳、日志级别、所在模块与函数名
- 分级输出：DEBUG（关键流程与变量状态）、INFO（策略选择与概要）、
  WARNING（降级与缺失依赖）、ERROR（异常及 traceback）
- 灵活开关：通过 setup_logging() 或 enable_debug_logging() 控制调试日志
- 安全原则：日志仅记录文本长度/块数/策略名等元信息，绝不输出完整文档内容
  （文档正文可能包含敏感业务数据）
"""

import logging
import os
import sys
from typing import List,Tuple,Optional,Dict,Any,Callable
from dataclasses import dataclass
import re
import warnings

# 模块级 logger：name 自动包含 backend.rag.text_splitter，便于过滤定位
logger = logging.getLogger(__name__)

# 缺陷9：历史遗留的 91 处 print 统一收口为 _vprint。
# TEXT_SPLITTER_VERBOSE 环境变量开关（默认开，保持现网调试输出；设 0/false 关闭）
TEXT_SPLITTER_VERBOSE = os.getenv("TEXT_SPLITTER_VERBOSE", "1").strip().lower() not in (
    "0", "false", "no",
)


def _vprint(*args, **kwargs) -> None:
    """调试输出统一入口，由 TEXT_SPLITTER_VERBOSE 控制开关。"""
    if TEXT_SPLITTER_VERBOSE:
        print(*args, **kwargs)


# 模块级日志格式缓存，setup_logging 幂等使用
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)s | %(message)s"
_LOG_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    配置本模块的日志输出（幂等，可重复调用）

    参数:
        level: 日志级别，可选 logging.DEBUG / INFO / WARNING / ERROR，
               默认 INFO（只输出关键流程，不含 DEBUG 细节）
        fmt:   自定义日志格式；默认含时间戳、级别、模块.函数名
        force: 若 True，强制覆盖已有 handler 配置（用于切换调试级别）

    说明:
        - 仅配置名为 "backend.rag.text_splitter" 的 logger，不干扰应用其他日志
        - 调用方可在应用启动时调用以启用调试日志
    """
    global _LOG_CONFIGURED
    lgr = logging.getLogger("backend.rag.text_splitter")
    if _LOG_CONFIGURED and not force:
        return

    formatter = logging.Formatter(fmt or _LOG_FORMAT)

    # 日志输出到控制台（stderr，避免污染 stdout 数据流）
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream.setLevel(level)

    # 输出到文件（可选：与 stderr 并存）
    # file_handler = logging.FileHandler("text_splitter.log", encoding="utf-8")
    # file_handler.setFormatter(formatter)
    # file_handler.setLevel(level)

    # 若强制重建，先清除旧 handler 避免重复输出
    if force:
        lgr.handlers.clear()

    lgr.setLevel(level)
    lgr.addHandler(stream)
    #lgr.addHandler(file_handler)
    lgr.propagate = False  # 避免与应用 root logger 重复输出
    _LOG_CONFIGURED = True
    lgr.log(level, "文本分割器日志已配置: level=%s", logging.getLevelName(level))


def enable_debug_logging() -> None:
    """
    便捷开关：启用 DEBUG 级调试日志

    用于排查分割流程时，可在运行时临时开启详细日志。
    等价于 setup_logging(logging.DEBUG, force=True)。
    """
    setup_logging(logging.DEBUG, force=True)


@dataclass
class TextChunk:
    """
    文本块数据类 - Text Chunk Data Class
    
    封装文本分割后的单个块信息，包含内容、位置索引和元数据。
    
    属性:
        content: 块内容（纯文本）
        chunk_index: 块索引（从0开始）
        total_chunks: 总块数
        start_pos: 在原始文本中的起始位置
        end_pos: 在原始文本中的结束位置
        metadata: 元数据字典（包含分割类型、分隔符等信息）
    """
    content: str
    chunk_index: int
    total_chunks: int
    start_pos: int
    end_pos: int
    metadata: Optional[dict] = None


class BaseSplitter:
    """
    文本分割器抽象基类 - Text Splitter Abstract Base Class
    
    定义所有分割器必须实现的接口方法，提供统一的配置参数。
    
    设计决策：使用ABC抽象基类强制子类实现split_text方法，确保接口一致性。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separator: str = "\n\n"
    ):
        """
        初始化分割器配置
        
        参数:
            chunk_size: 每个块的最大字符数，默认512
            chunk_overlap: 块之间的重叠字符数，默认64（保持上下文连续性）
            separator: 默认分隔符，用于文本分割
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[TextChunk]:
        """
        分割文本（抽象方法，子类必须实现）
        
        参数:
            text: 待分割的文本内容
            
        返回:
            List[TextChunk]: 分割后的文本块列表
        """
        raise NotImplementedError


class RecursiveCharacterSplitter(BaseSplitter):
    """
    递归字符分割器 - Recursive Character Splitter
    
    按优先级依次尝试不同分隔符进行文本分割，确保语义完整性。
    优先使用段落分隔符（\n\n），其次是换行符（\n）、句点等。
    
    设计决策：这是最通用的分割策略，适用于大多数文本类型，
    能在保持语义完整的前提下将文本切分为合适大小的块。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None
    ):
        """
        初始化递归字符分割器
        
        参数:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符优先级列表，默认按[段落、换行、句点(英文)、句点(中文)、空格、无]排序
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "。", " ", ""]

    def split_text(self, text: str) -> List[TextChunk]:
        """
        递归按分隔符优先级拆分文本
        
        算法流程：
        1. 从文本起始位置开始，尝试截取chunk_size大小的片段
        2. 在片段内从后向前查找最高优先级的分隔符
        3. 找到后在分隔符处分割，确保语义完整性
        4. 下一次分割从当前位置减去overlap开始，保持上下文连续性
        
        参数:
            text: 待分割的文本内容
            
        返回:
            List[TextChunk]: 分割后的文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            last_separator = -1
            actual_sep = ""
            
            # 按优先级查找分隔符，优先选择靠后的分隔符位置
            for sep in self.separators:
                if sep:
                    pos = text.rfind(sep, start, end)
                    if pos > last_separator:
                        last_separator = pos
                        actual_sep = sep

            # 如果找到分隔符，调整结束位置到分隔符之后
            if last_separator >= start:
                end = last_separator + len(actual_sep)
            
            # 提取块内容并去除首尾空白
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(TextChunk(
                    content=chunk,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_pos=start,
                    end_pos=end,
                    metadata={"separator": actual_sep}
                ))
                chunk_index += 1

            # 更新起始位置，保留overlap以保持上下文连续性
            if end == start:
                end += 1
            start = max(end - self.chunk_overlap, start + 1)

        # 更新所有块的总块数
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

class SemanticSplitter(BaseSplitter):
    """
    修复版语义分割器（无三方依赖）
    修复了4大缺陷：超长段落硬切、完整语义重叠、精准位置溯源、新块起始溢出校验
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
        # 预编译正则提升性能
        self.para_splitter = re.compile(r'\n\s*\n')          # 段落分隔（备用）
        self.sent_splitter = re.compile(r'(?<=[。！？；！\?;])')   # 句子边界（中英文标点）
        self.clause_splitter = re.compile(r'(?<=[，、,\s])')     # 子句边界（逗号/空格）
        
        _vprint(f"\n[INIT] 语义分割器初始化: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        _vprint(f"[INIT] 修复特性: 超长硬切 | 完整原子重叠 | 精准坐标 | 溢出回退\n{'-'*60}")

    # ======================================================================
    # 核心方法1：将原始文本拆解为“原子单元”（带绝对坐标）
    # 修复：【缺陷一】单段落超长导致长度限制失效
    # ======================================================================
    def _atomic_split(self, text: str) -> List[Tuple[str, int, int]]:
        """
        将文本递归拆解为最小语义单元（原子）。
        降级策略：段落 → 句子 → 子句 → 强制硬切。
        每个原子返回：(文本内容, 绝对起始位, 绝对结束位)
        """
        atoms = []
        # 【V2优化】使用 finditer 直接获取段落及其绝对坐标，避免 split 丢失空行位置
        para_matches = list(re.finditer(r'[^\n]+(?:\n[^\n]+)*', text))
        
        for match in para_matches:
            para_text = match.group()          # 段落内容
            para_start = match.start()         # 绝对起始坐标
            para_end = match.end()             # 绝对结束坐标
            orig_len = len(para_text)

            # ---------- 情况1：段落本身合规，直接作为原子 ----------
            if len(para_text) <= self.chunk_size:
                atoms.append((para_text, para_start, para_end))
                continue

            # ---------- 情况2：段落超长 -> 触发【缺陷一修复】 ----------
            _vprint(f"[FIX-1] 检测到超长段落 (长度:{orig_len} > {self.chunk_size})，开始递归降级拆分。")
            
            # 尝试按句子拆分（。！？；）
            sent_matches = list(re.finditer(r'[^。！？；！\?;]+[。！？；！\?;]?', para_text))
            
            # 若无法拆出句子（如纯英文无标点），直接暴力硬切
            if not sent_matches:
                for i in range(0, len(para_text), self.chunk_size):
                    sub = para_text[i:i+self.chunk_size]
                    atoms.append((sub, para_start + i, para_start + min(i+self.chunk_size, len(para_text))))
                    _vprint(f"  ↳ [硬切] 片段 [{i}:{i+self.chunk_size}] 长度:{len(sub)}")
                continue

            # 遍历句子
            for s_match in sent_matches:
                sent_text = s_match.group()
                sent_abs_start = para_start + s_match.start()
                sent_abs_end = para_start + s_match.end()

                # 句子合规 -> 原子
                if len(sent_text) <= self.chunk_size:
                    atoms.append((sent_text, sent_abs_start, sent_abs_end))
                else:
                    # 句子超长 -> 降级为子句拆分（按逗号、空格）
                    clause_matches = list(re.finditer(r'[^，、,\s]+[，、,\s]?', sent_text))
                    if not clause_matches:
                        # 子句拆分失败，暴力硬切
                        for i in range(0, len(sent_text), self.chunk_size):
                            sub = sent_text[i:i+self.chunk_size]
                            atoms.append((sub, sent_abs_start + i, sent_abs_start + min(i+self.chunk_size, len(sent_text))))
                            _vprint(f"    ↳ [子句硬切] 长度:{len(sub)}")
                    else:
                        for c_match in clause_matches:
                            clause_text = c_match.group()
                            if clause_text.strip():  # 忽略纯空格
                                atoms.append((clause_text, sent_abs_start + c_match.start(), sent_abs_start + c_match.end()))

        # ---------- 终极安全兜底：若有原子仍超长（如Base64），强制按字符硬切 ----------
        final_atoms = []
        for atom_text, a_start, a_end in atoms:
            if len(atom_text) > self.chunk_size:
                _vprint(f"[FIX-1] 终极兜底硬切超长原子 (长度:{len(atom_text)})")
                for i in range(0, len(atom_text), self.chunk_size):
                    sub = atom_text[i:i+self.chunk_size]
                    final_atoms.append((sub, a_start + i, a_start + min(i+self.chunk_size, len(atom_text))))
            else:
                final_atoms.append((atom_text, a_start, a_end))
        
        _vprint(f"[ATOMIC] 原子拆分完成，共生成 {len(final_atoms)} 个语义单元。")
        return final_atoms

    # ======================================================================
    # 核心方法2：从原子列表中提取“完整语义重叠”
    # 修复：【缺陷二】字符级重叠割裂语义（改为完整原子提取）
    # ======================================================================
    def _get_overlap_atoms(self, atoms: List[Tuple[str, int, int]], max_chars: int) -> List[Tuple[str, int, int]]:
        """
        从原子列表末尾反向提取尽可能多的【完整原子】，总长度不超过 max_chars。
        绝不截断字符串，确保重叠部分语义完整。
        """
        overlap = []
        total_len = 0
        for atom_text, a_start, a_end in reversed(atoms):
            atom_len = len(atom_text)
            if total_len + atom_len <= max_chars:
                overlap.insert(0, (atom_text, a_start, a_end))  # 保持原顺序
                total_len += atom_len
            else:
                break  # 放不下就停止，宁缺毋滥
        
        if overlap:
            preview = overlap[0][0][:30] + "..." if len(overlap[0][0]) > 30 else overlap[0][0]
            _vprint(f"[FIX-2] 完整语义重叠: 提取 {len(overlap)} 个完整原子 (共{total_len}/{max_chars}字符)")
            _vprint(f"  ↳ 重叠预览: '{preview}' (对比原代码的暴力字符截断)")
        return overlap

    # ======================================================================
    # 核心方法3：主分割逻辑
    # 修复：【缺陷三】位置追踪完全失效（取首尾原子绝对坐标）
    # 修复：【缺陷四】新块起始即溢出（头部原子弹出校验）
    # ======================================================================
    def split_text(self, text: str) -> List[TextChunk]:
        _vprint(f"\n[SPLIT] 开始分割，原始文本总字符数: {len(text)}")
        
        # 1. 获取带坐标的原子列表
        atoms = self._atomic_split(text)
        if not atoms:
            return []

        chunks = []
        current_atoms = []      # 当前块包含的原子列表
        current_length = 0      # 当前块字符数
        chunk_index = 0

        for atom_text, a_start, a_end in atoms:
            atom_len = len(atom_text)

            # ---------- 触发切分条件 ----------
            if current_length + atom_len > self.chunk_size and current_atoms:
                # --- 生成当前块（修复【缺陷三】） ---
                chunk_text = "".join([a[0] for a in current_atoms])
                real_start = current_atoms[0][1]   # 块内首个原子的绝对起始
                real_end = current_atoms[-1][2]    # 块内末尾原子的绝对结束
                
                _vprint(f"\n[FIX-3] 生成块 #{chunk_index}: 内容长度 {len(chunk_text)}")
                _vprint(f"  ↳ 精准溯源: start_pos={real_start}, end_pos={real_end} (原代码恒为0)")

                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_pos=real_start,
                    end_pos=real_end,
                    metadata={"type": "semantic_fixed"}
                ))
                chunk_index += 1

                # --- 重叠处理（修复【缺陷二】） ---
                if self.chunk_overlap > 0:
                    overlap_atoms = self._get_overlap_atoms(current_atoms, self.chunk_overlap)
                    current_atoms = overlap_atoms + [(atom_text, a_start, a_end)]
                else:
                    current_atoms = [(atom_text, a_start, a_end)]

                # --- 强校验新块是否溢出（修复【缺陷四】） ---
                current_length = sum(len(a[0]) for a in current_atoms)
                if current_length > self.chunk_size:
                    _vprint(f"[FIX-4] 新块拼接后超限 (长度:{current_length} > {self.chunk_size})，触发头部弹出校验。")

                # 循环丢弃头部原子，直到长度合规（保证新块从创建起就合法）
                while current_length > self.chunk_size and len(current_atoms) > 1:
                    removed = current_atoms.pop(0)
                    removed_len = len(removed[0])
                    current_length -= removed_len
                    _vprint(f"  ↳ 弹出头部原子 (丢弃{removed_len}字符), 当前长度: {current_length}")

                # 极端情况：只剩一个超长原子，强制截断（避免死循环）
                if current_length > self.chunk_size and len(current_atoms) == 1:
                    old_text, old_start, old_end = current_atoms[0]
                    _vprint(f"[FIX-4] 极限兜底: 截断唯一超长原子 (原长:{len(old_text)})")
                    current_atoms = [(old_text[:self.chunk_size], old_start, old_start + self.chunk_size)]
                    current_length = self.chunk_size
                    _vprint(f"  ↳ 截断后长度为 {current_length}")

            else:
                # --- 未超限，正常加入 ---
                current_atoms.append((atom_text, a_start, a_end))
                current_length += atom_len

        # ---------- 保存最后一个块 ----------
        if current_atoms:
            chunk_text = "".join([a[0] for a in current_atoms])
            real_start = current_atoms[0][1]
            real_end = current_atoms[-1][2]
            
            _vprint(f"\n[FIX-3] 生成最后块 #{chunk_index}: 内容长度 {len(chunk_text)}")
            _vprint(f"  ↳ 精准溯源: start_pos={real_start}, end_pos={real_end}")

            chunks.append(TextChunk(
                content=chunk_text,
                chunk_index=chunk_index,
                total_chunks=0,
                start_pos=real_start,
                end_pos=real_end,
                metadata={"type": "semantic_fixed"}
            ))

        # 回填总块数
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        _vprint(f"\n[RESULT] 分割完成，共生成 {len(chunks)} 个块。\n{'='*60}\n")
        return chunks

class CodeSplitter(BaseSplitter):
    """
    代码感知分割器 - Code-aware Splitter
    
    针对代码文件进行分割，按函数/类定义边界进行切分，确保代码块的完整性。
    
    设计决策：代码文件有特殊的结构（函数、类），按语法边界分割能更好地保留代码语义，
    避免将函数定义分割到多个块中。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):
        """
        初始化代码分割器
        
        参数:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        super().__init__(chunk_size, chunk_overlap)

    def split_text(self, text: str) -> List[TextChunk]:
        """
        按函数/类边界分割代码文本
        
        算法流程：
        1. 使用正则表达式按def/class关键字分割代码
        2. 保持函数/类定义的完整性
        3. 超过chunk_size时保存当前块，保留部分上下文作为overlap
        
        参数:
            text: 待分割的代码文本
            
        返回:
            List[TextChunk]: 分割后的代码块列表
        """
        # 按函数定义（def）或类定义（class）进行分割
        functions = re.split(r'(def\s+\w+\s*\(|class\s+\w+)', text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        # 正则分割后，奇数索引为分隔符（def/class），偶数索引为内容
        for i in range(0, len(functions), 2):
            if i + 1 < len(functions):
                func_def = functions[i] + functions[i + 1]  # 合并分隔符和内容
            else:
                func_def = functions[i]

            func_length = len(func_def)
            
            # 如果加入当前函数会超过限制，且当前块不为空，则保存当前块
            if current_length + func_length > self.chunk_size and current_chunk:
                chunk_content = "".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_pos=0,  # 代码分割暂不记录精确位置
                    end_pos=len(chunk_content),
                    metadata={"type": "code"}
                ))
                chunk_index += 1
                
                # 处理重叠：保留最后一部分内容作为上下文
                if self.chunk_overlap > 0 and current_chunk:
                    last_part = current_chunk[-1]
                    overlap = last_part[-self.chunk_overlap:] if len(last_part) > self.chunk_overlap else last_part
                    current_chunk = [overlap + func_def]
                else:
                    current_chunk = [func_def]
                current_length = len(current_chunk[0])
            else:
                current_chunk.append(func_def)
                current_length += func_length

        # 保存最后一个块
        if current_chunk:
            chunk_content = "".join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_content,
                chunk_index=chunk_index,
                total_chunks=0,
                start_pos=0,
                end_pos=len(chunk_content),
                metadata={"type": "code"}
            ))

        # 更新总块数
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks


# ======================================================================
# LangChain 集成：桥接类 + 工具函数 + 分割器适配器
# ======================================================================

def _try_import_langchain() -> Dict[str, Any]:
    """
    惰性导入 LangChain 文本分割相关类

    使 LangChain 成为可选依赖：未安装时返回空字典，调用方自动降级到自研分割器。

    返回:
        Dict[str, Any]: 已导入的类字典，键为类名，值为类对象。
                        导入失败时为空字典。
    """
    modules = {}

    # 核心分割器：langchain_text_splitters 提供（必需，任一缺失则该类不可用）
    try:
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            TokenTextSplitter,
            MarkdownHeaderTextSplitter,
        )
        modules.update({
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "TokenTextSplitter": TokenTextSplitter,
            "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter,
        })
        _vprint("[LANGCHAIN] LangChain 核心分割器导入成功")
    except ImportError as e:
        _vprint(f"[LANGCHAIN][WARN] LangChain 文本分割器未安装 ({e})，"
              f"langchain_recursive/token/markdown 策略将降级到自研分割器；"
              f"可通过 pip install langchain-text-splitters tiktoken 安装")
        warnings.warn(
            "LangChain 文本分割器未安装，相关策略将不可用。"
            "可通过 pip install langchain-text-splitters tiktoken 安装。",
            stacklevel=2,
        )

    # 语义分割器：SemanticChunker 在新版本位于 langchain_experimental，
    # 单独导入，失败时仅影响 langchain_semantic 策略，不影响其他 LangChain 策略
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        modules["SemanticChunker"] = SemanticChunker
        _vprint("[LANGCHAIN][DEBUG] SemanticChunker 导入成功（来自 langchain_experimental）")
    except ImportError:
        try:
            from langchain_text_splitters import SemanticChunker
            modules["SemanticChunker"] = SemanticChunker
            _vprint("[LANGCHAIN][DEBUG] SemanticChunker 导入成功（来自 langchain_text_splitters）")
        except ImportError:
            _vprint("[LANGCHAIN][WARN] SemanticChunker 不可用（需安装 langchain_experimental），"
                  "langchain_semantic 策略将降级到自研语义分割")

    # 公共依赖：langchain_core
    try:
        from langchain_core.embeddings import Embeddings as LCEmbeddings
        from langchain_core.documents import Document as LCDocument
        modules["LCEmbeddings"] = LCEmbeddings
        modules["LCDocument"] = LCDocument
        _vprint("[LANGCHAIN][DEBUG] langchain_core 导入成功 (LCEmbeddings / LCDocument)")
    except ImportError as e:
        _vprint(f"[LANGCHAIN][WARN] langchain_core 导入失败 ({e})，LangChain 策略将不可用")

    _vprint(f"[LANGCHAIN][DEBUG] LangChain 模块导入结果 keys={list(modules.keys())}")
    return modules


# 模块级惰性导入缓存，避免重复尝试导入
_langchain_modules = None


def _get_langchain_modules() -> Dict[str, Any]:
    """获取 LangChain 模块缓存，惰性初始化"""
    global _langchain_modules
    if _langchain_modules is None:
        _langchain_modules = _try_import_langchain()
        _vprint(f"[LANGCHAIN][DEBUG] LangChain 模块缓存已建立，模块数: {len(_langchain_modules)}")
    return _langchain_modules


class _LlamaIndexEmbeddingsAdapter:
    """
    LlamaIndex → LangChain Embeddings 接口桥接类

    项目使用 LlamaIndex 的 HuggingFaceEmbedding（接口为 get_text_embedding），
    而 LangChain SemanticChunker 需要 LangChain Embeddings 接口
    （embed_documents / embed_query）。本类完成接口适配。

    设计决策：
    - 不修改 embedding_service.py，通过组合方式适配接口
    - 若底层模型缺失（如 llama-index 未安装），embed 方法返回空列表，
      由上层语义分割器降级处理
    """

    def __init__(self, llama_index_model: Any):
        self._model = llama_index_model
        _vprint(f"[EMB][DEBUG] LlamaIndex Embedding 适配器已创建，底层模型可用: {self._model is not None}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本向量化（LangChain Embeddings 接口）"""
        if self._model is None:
            _vprint(f"[EMB][DEBUG] embed_documents 降级：底层 Embedding 模型不可用，返回空向量 (文本数={len(texts)})")
            return [[] for _ in texts]
        try:
            embeddings = self._model.get_text_embeddings(texts)
            _vprint(f"[EMB][DEBUG] embed_documents 批量向量化完成: 文本数={len(texts)}, 维度={len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except AttributeError:
            _vprint(f"[EMB][DEBUG] get_text_embeddings 接口不可用，逐个调用 get_text_embedding (文本数={len(texts)})")
            return [self._model.get_text_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本向量化（LangChain Embeddings 接口）"""
        if self._model is None:
            _vprint("[EMB][DEBUG] embed_query 降级：底层 Embedding 模型不可用，返回空向量")
            return []
        vector = self._model.get_text_embedding(text)
        _vprint(f"[EMB][DEBUG] embed_query 向量化完成: 向量维度={len(vector)}")
        return vector


def _get_project_embeddings_adapter():
    """
    从项目 embedding_service 获取并包装 LlamaIndex Embedding 模型

    返回:
        已适配的 LangChain Embeddings 兼容对象，或 None（embedding 不可用时）
    """
    try:
        from backend.rag.embedding_service import embedding_service
        model = embedding_service.embed_model
        _vprint(f"[EMB][DEBUG] 从 embedding_service 获取模型: type(model)={type(model).__name__ if model else None}")
        if model is None:
            _vprint("[EMB][WARN] Embedding 服务可用但其底层模型为 None，语义分割将降级")
            return None
        _vprint("[EMB][DEBUG] 成功获取项目 Embedding 模型并包装为 LangChain 适配器")
        return _LlamaIndexEmbeddingsAdapter(model)
    except Exception as e:
        import traceback
        _vprint(f"[EMB][ERROR] 获取项目 Embedding 适配器失败: {e}")
        traceback.print_exc()
        warnings.warn(f"获取项目 Embedding 适配器失败: {e}", stacklevel=2)
        return None


def _langchain_docs_to_chunks(
    lc_docs: List[Any],
    chunk_type: str,
    chunk_index_start: int = 0,
) -> List[TextChunk]:
    """
    将 LangChain Document 列表转换为项目 TextChunk 列表

    处理逻辑：
    1. 过滤空白块（strip 后为空则丢弃）
    2. 若 LangChain 分块未记录位置，则按累计长度重建 start_pos/end_pos
    3. 透传 LangChain 分块元数据（如 Markdown 标题）到 TextChunk.metadata

    参数:
        lc_docs: LangChain Document 列表
        chunk_type: 分割类型标识，写入 metadata.type
        chunk_index_start: 起始索引（用于拼接时保持连续）

    返回:
        List[TextChunk]: 转换后的文本块列表
    """
    _vprint(f"[LC2CHUNK][DEBUG] LangChain Document 转换开始: 输入文档数={len(lc_docs)}, 类型={chunk_type}, 起始索引={chunk_index_start}")
    chunks = []
    pos = 0
    chunk_index = chunk_index_start
    skipped_blank = 0

    for doc in lc_docs:
        # 兼容两种 LangChain 输出类型：
        # 1. split_text() 返回 List[str]（纯字符串块）
        # 2. split_documents() 返回 List[Document]（带 metadata 的对象）
        if isinstance(doc, str):
            content = doc
            meta = {}
        else:
            content = getattr(doc, "page_content", "") or ""
            meta = dict(getattr(doc, "metadata", {}) or {})
        # 过滤空白块
        if not content.strip():
            skipped_blank += 1
            continue

        # 提取 LangChain 自带的位置元数据（若存在），否则按累计长度重建
        start_pos = meta.pop("start_index", pos)
        end_pos = meta.pop("end_index", pos + len(content))
        meta["type"] = chunk_type

        chunks.append(TextChunk(
            content=content,
            chunk_index=chunk_index,
            total_chunks=0,
            start_pos=start_pos,
            end_pos=end_pos,
            metadata=meta,
        ))
        _vprint(f"  [LC2CHUNK][DEBUG] 块#{chunk_index}: 长度={len(content)}, "
              f"位置[{start_pos},{end_pos}), 元数据字段={list(meta.keys())}")
        chunk_index += 1
        pos = end_pos

    # 回填总块数
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    _vprint(f"[LC2CHUNK] LangChain Document 转换完成: 生成块数={len(chunks)}, 跳过空白块={skipped_blank}, 类型={chunk_type}")
    return chunks


class LangChainRecursiveSplitter(BaseSplitter):
    """
    LangChain 递归字符分割器适配器

    底层使用 LangChain 的 RecursiveCharacterTextSplitter，按分隔符优先级
    递归拆分，确保语义完整性。默认分隔符覆盖中英文标点，适配多语言内容。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        _vprint(f"[RECURSIVE][DEBUG] LangChainRecursiveSplitter 初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
              f"分隔符数={len(self.separators)}")

    def split_text(self, text: str) -> List[TextChunk]:
        _vprint(f"[RECURSIVE][DEBUG] langchain_recursive 开始分割: 输入字符数={len(text)}")
        mods = _get_langchain_modules()
        splitter_cls = mods.get("RecursiveCharacterTextSplitter")
        if splitter_cls is None:
            _vprint("[RECURSIVE][WARN] LangChain 未安装，langchain_recursive 策略降级为自研递归分割")
            warnings.warn("LangChain 未安装，langchain_recursive 策略降级为自研递归分割", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap, separators=self.separators
            ).split_text(text)

        splitter = splitter_cls(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        lc_docs = splitter.split_text(text)
        _vprint(f"[RECURSIVE][DEBUG] RecursiveCharacterTextSplitter 产出 {len(lc_docs)} 个原始文档")
        return _langchain_docs_to_chunks(lc_docs, "langchain_recursive")


class LangChainTokenSplitter(BaseSplitter):
    """
    LangChain Token 分割器适配器

    底层使用 LangChain 的 TokenTextSplitter，基于真实 BPE tokenizer 统计，
    精确控制每个块的 token 数，适合需要严格 LLM 上下文预算控制的场景。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        tokenizer: Optional[Any] = None,
        encoding_name: Optional[str] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.tokenizer = tokenizer
        self.encoding_name = encoding_name or "cl100k_base"
        _vprint(f"[TOKEN][DEBUG] LangChainTokenSplitter 初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
              f"encoding={self.encoding_name}, 自定义tokenizer={tokenizer is not None}")

    def split_text(self, text: str) -> List[TextChunk]:
        _vprint(f"[TOKEN][DEBUG] langchain_token 开始分割: 输入字符数={len(text)}, encoding={self.encoding_name}")
        mods = _get_langchain_modules()
        splitter_cls = mods.get("TokenTextSplitter")
        if splitter_cls is None:
            _vprint("[TOKEN][WARN] LangChain 未安装，langchain_token 策略降级为自研递归分割")
            warnings.warn("LangChain 未安装，langchain_token 策略降级为自研递归分割", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_text(text)

        try:
            kwargs = dict(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            if self.tokenizer is not None:
                kwargs["tokenizer"] = self.tokenizer
            else:
                kwargs["encoding_name"] = self.encoding_name
            splitter = splitter_cls(**kwargs)
            lc_docs = splitter.split_text(text)
            _vprint(f"[TOKEN][DEBUG] TokenTextSplitter 产出 {len(lc_docs)} 个原始文档")
        except Exception as e:
            import traceback
            _vprint(f"[TOKEN][ERROR] TokenTextSplitter 初始化或分割失败，降级为自研递归分割: {e}")
            traceback.print_exc()
            warnings.warn(f"TokenTextSplitter 初始化或分割失败，降级为自研递归分割: {e}", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_text(text)

        return _langchain_docs_to_chunks(lc_docs, "langchain_token")


class LangChainMarkdownSplitter(BaseSplitter):
    """
    LangChain Markdown 分割器适配器

    底层使用 LangChain 的 MarkdownHeaderTextSplitter（按 H1-H3 标题切分），
    再用递归字符分割器对超长标题块二次切分，保留标题上下文元数据。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        _vprint(f"[MARKDOWN][DEBUG] LangChainMarkdownSplitter 初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
              f"标题级别数={len(self.headers_to_split_on)}")

    def split_text(self, text: str) -> List[TextChunk]:
        _vprint(f"[MARKDOWN][DEBUG] langchain_markdown 开始分割: 输入字符数={len(text)}")
        mods = _get_langchain_modules()
        header_splitter_cls = mods.get("MarkdownHeaderTextSplitter")
        recursive_cls = mods.get("RecursiveCharacterTextSplitter")
        if header_splitter_cls is None or recursive_cls is None:
            _vprint("[MARKDOWN][WARN] LangChain 未安装，langchain_markdown 策略降级为自研递归分割")
            warnings.warn("LangChain 未安装，langchain_markdown 策略降级为自研递归分割", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_text(text)

        # 1. 按标题切分
        header_splitter = header_splitter_cls(headers_to_split_on=self.headers_to_split_on)
        header_docs = header_splitter.split_text(text)
        _vprint(f"[MARKDOWN][DEBUG] Markdown 按标题切分完成: 产出 {len(header_docs)} 个标题块")

        # 2. 对超长标题块二次递归切分，保留标题元数据
        recursive_splitter = recursive_cls(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        )
        final_docs = []
        oversize_count = 0
        for doc in header_docs:
            content = getattr(doc, "page_content", "") or ""
            if len(content) <= self.chunk_size:
                final_docs.append(doc)
            else:
                oversize_count += 1
                sub_chunks = recursive_splitter.split_text(content)
                for sub in sub_chunks:
                    final_docs.append(sub)

        _vprint(f"[MARKDOWN][DEBUG] Markdown 二次切分完成: 超长块={oversize_count}, 最终文档数={len(final_docs)}")
        return _langchain_docs_to_chunks(final_docs, "langchain_markdown")


class LangChainSemanticSplitter(BaseSplitter):
    """
    LangChain 语义分割器适配器

    底层使用 LangChain 的 SemanticChunker，基于 embedding 余弦距离识别语义边界，
    生成语义完整的块。内置两个安全兜底：
    1. 无 embedding 环境：自动降级为自研递归分割
    2. 语义块超 chunk_size：用递归分割器二次切分，避免超长块
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embeddings: Any = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        _vprint(f"[SEMANTIC][DEBUG] LangChainSemanticSplitter 初始化: chunk_size={chunk_size}, overlap={chunk_overlap}, "
              f"threshold_type={breakpoint_threshold_type}, threshold_amount={breakpoint_threshold_amount}, "
              f"外部注入embedding={embeddings is not None}")

    def split_text(self, text: str) -> List[TextChunk]:
        _vprint(f"[SEMANTIC][DEBUG] ====== langchain_semantic 开始分割 ======")
        _vprint(f"[SEMANTIC][DEBUG] 输入文本字符数={len(text)}, 文本前120字: {text[:120]!r}")
        _vprint(f"[SEMANTIC][DEBUG] 当前配置: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
              f"threshold_type={self.breakpoint_threshold_type}, threshold_amount={self.breakpoint_threshold_amount}")
        mods = _get_langchain_modules()
        semantic_cls = mods.get("SemanticChunker")
        _vprint(f"[SEMANTIC][DEBUG] 模块中是否含 SemanticChunker: {semantic_cls is not None}, 全部模块keys={list(mods.keys())}")
        if semantic_cls is None:
            _vprint("[SEMANTIC][WARN] LangChain 未安装，langchain_semantic 策略降级为自研语义分割")
            warnings.warn("LangChain 未安装，langchain_semantic 策略降级为自研语义分割", stacklevel=2)
            return SemanticSplitter(self.chunk_size, self.chunk_overlap).split_text(text)

        # 获取 embedding 适配器（外部注入优先，其次项目默认）
        embeddings = self.embeddings or _get_project_embeddings_adapter()
        _vprint(f"[SEMANTIC][DEBUG] 使用的 Embedding 适配器: type={type(embeddings).__name__ if embeddings else None}, "
              f"实例={embeddings}")
        if embeddings is None:
            _vprint("[SEMANTIC][WARN] 无可用 Embedding，langchain_semantic 降级为自研递归分割")
            warnings.warn("无可用 Embedding，langchain_semantic 降级为自研递归分割", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_text(text)

        try:
            _vprint(f"[SEMANTIC][DEBUG] 创建 SemanticChunker 实例: "
                  f"breakpoint_threshold_type={self.breakpoint_threshold_type}, "
                  f"breakpoint_threshold_amount={self.breakpoint_threshold_amount}")
            chunker = semantic_cls(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            )
            _vprint(f"[SEMANTIC][DEBUG] SemanticChunker 实例创建完成, type={type(chunker).__name__}")
            semantic_docs = chunker.split_text(text)
            _vprint(f"[SEMANTIC][DEBUG] SemanticChunker 产出 {len(semantic_docs)} 个语义块")
            for i, doc in enumerate(semantic_docs):
                _c = getattr(doc, "page_content", doc) if not isinstance(doc, str) else doc
                _vprint(f"  [SEMANTIC][DEBUG] 语义块#{i}: 长度={len(_c)}, 前60字={str(_c)[:60]!r}")
        except Exception as e:
            import traceback
            _vprint(f"[SEMANTIC][ERROR] SemanticChunker 分割失败，降级为自研递归分割: {e}")
            traceback.print_exc()
            warnings.warn(f"SemanticChunker 分割失败，降级为自研递归分割: {e}", stacklevel=2)
            return RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            ).split_text(text)

        # 安全兜底：对超 chunk_size 的语义块二次递归切分
        recursive_cls = mods.get("RecursiveCharacterTextSplitter")
        final_docs = []
        oversize_count = 0
        for doc in semantic_docs:
            content = getattr(doc, "page_content", "") or ""
            if len(content) <= self.chunk_size or recursive_cls is None:
                final_docs.append(doc)
            else:
                oversize_count += 1
                _vprint(f"[SEMANTIC][DEBUG] 语义块超长({len(content)} > {self.chunk_size})，触发二次递归切分")
                sub_splitter = recursive_cls(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
                )
                for sub in sub_splitter.split_text(content):
                    final_docs.append(sub)

        if oversize_count:
            _vprint(f"[SEMANTIC][DEBUG] 语义块超长二次切分: 超长块={oversize_count}, 最终文档数={len(final_docs)}")
        else:
            _vprint(f"[SEMANTIC][DEBUG] 语义块均未超长，无需二次切分，最终文档数={len(final_docs)}")
        return _langchain_docs_to_chunks(final_docs, "langchain_semantic")


# 文件类型/扩展名 -> 推荐分割策略 映射表
# 设计决策：集中维护路由规则，避免散落各处难维护；路由优先级由 auto_route 触发时覆盖默认策略。
_FILE_TYPE_STRATEGY_MAP = {
    # 结构化技术文档：保留标题层级上下文
    "md": "langchain_markdown",
    "markdown": "langchain_markdown",
    # 代码文档：按函数/类边界切分
    "py": "code",
    "java": "code",
    "js": "code",
    "ts": "code",
    "tsx": "code",
    "jsx": "code",
    "c": "code",
    "cpp": "code",
    "h": "code",
    "hpp": "code",
    "go": "code",
    "rs": "code",
    "rb": "code",
    "php": "code",
    "sql": "code",
    "sh": "code",
    "bash": "code",
    "json": "code",
    "yml": "code",
    "yaml": "code",
    "toml": "code",
    "xml": "code",
    "html": "code",
    "css": "code",
    # 表格数据：行数少但单行可能很长，适合递归 + 较小 overlap 即可（走默认递归/语义）
    # 其余类型（docx/pdf/txt/xlsx 等）默认交给语义/递归策略
}

# 无扩展名时的回退默认策略（其余所有文档类型）
_DEFAULT_ROUTE_STRATEGY = "langchain_semantic"

# ======================================================================
# Mermaid 围栏保护（doc_to_md 增强产物约束：mermaid 代码块必须完整
# 落在同一个 chunk，不能被切片截断）
#
# 实现：在切片前把 ```mermaid ... ``` 围栏块替换为**等长占位符**，
# 占位符无换行、无标点，作为独立"原子"参与任何分割策略都不会被切碎；
# 切片完成后统一还原。等长替换保证文本总长与每个字符的绝对坐标
# 全部不变，不影响现有 start_pos/end_pos 溯源。
#
# 兼容性保证：
#   1. 文本中不含 ```mermaid 时，保护函数直接原样返回，零开销零影响；
#   2. 占位符长度固定（19 字符），超出 100 万块时自动取模避免变长。
# ======================================================================
_MERMAID_FENCE_RE = re.compile(r"```mermaid[^\n]*\n.*?```", re.S | re.I)
_MERMAID_PH_PREFIX = "__MERMAID_"
_MERMAID_PH_SUFFIX = "__"
_MERMAID_PH_LEN = len(_MERMAID_PH_PREFIX) + 6 + len(_MERMAID_PH_SUFFIX)  # 19


def _protect_mermaid_fences(text: str):
    """
    把 ```mermaid 围栏块替换为等长占位符。

    返回:
        (protected_text, fence_map)
        fence_map: {占位符: 原始mermaid块}，为空表示无 mermaid 块
    """
    if not _MERMAID_FENCE_RE.search(text):
        return text, {}
    fence_map = {}

    def _repl(m):
        idx = len(fence_map) % 1000000  # 取模保证占位符长度恒定
        ph = f"{_MERMAID_PH_PREFIX}{idx:06d}{_MERMAID_PH_SUFFIX}"
        while ph in fence_map:  # 极端冲突时递增重试
            idx = (idx + 1) % 1000000
            ph = f"{_MERMAID_PH_PREFIX}{idx:06d}{_MERMAID_PH_SUFFIX}"
        fence_map[ph] = m.group(0)
        return ph

    return _MERMAID_FENCE_RE.sub(_repl, text), fence_map


def _restore_mermaid_fences(chunks: List[TextChunk], fence_map: dict) -> None:
    """把 chunk 内容中的 mermaid 占位符还原为原始围栏块。"""
    if not fence_map:
        return
    for ch in chunks:
        content = ch.content
        if _MERMAID_PH_PREFIX in content:
            for ph, orig in fence_map.items():
                if ph in content:
                    content = content.replace(ph, orig)
            ch.content = content


def _guess_code_language(text: str) -> str:
    """
    轻量内容嗅探：判断文本是否更接近代码而非自然语言。

    设计决策：仅用于在缺失类型信息时做兜底判断，宁可误判为自然语言
    走默认语义切分，也不轻易对自然语言施加代码切分。
    返回空字符串表示"不像代码"。

    参数:
        text: 待判断的文本内容

    返回:
        str: 命中的代码语言（如 "python"），未命中返回空字符串
    """
    if not text or len(text) < 50:
        return ""
    # 常见代码特征：花括号、def/class/import 等关键字、注释符、赋值箭头等
    low = text[:2000].lower()
    python_hits = sum(kw in low for kw in ("def ", "class ", "import ", "return ", "self."))
    code_hits = sum(kw in low for kw in ("{", "}", "=>", "->", "// ", "/*", "else if"))
    # 若命中的代码特征达到阈值，视为代码文本
    if python_hits >= 3 or code_hits >= 4:
        return "python" if python_hits >= 3 else "code"
    return ""


class TextSplitter:
    """
    文本分割器统一入口 - Text Splitter Facade
    
    根据策略名称创建对应的分割器实例，提供统一的文本分割接口。
    支持按文档类型自动路由分割策略（auto_route）。
    
    设计决策：采用门面模式（Facade），对外提供简洁的接口，
    隐藏内部多策略实现细节，便于调用方使用。
    """

    def __init__(
        self,
        strategy: str = "langchain_semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        auto_route: bool = False
    ):
        """
        初始化文本分割器
        
        参数:
            strategy: 分割策略，可选值：
                - "recursive"（自研递归字符）
                - "semantic"（自研语义，默认）
                - "code"（代码感知）
                - "langchain_recursive"（LangChain 递归字符）
                - "langchain_token"（LangChain Token）
                - "langchain_markdown"（LangChain Markdown）
                - "langchain_semantic"（LangChain 语义）
                - "auto"（按文档类型自动路由，配合 split_document 使用）
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            auto_route: 是否按文档类型自动路由策略（当为 True 时，strategy 会被
                        split_document 中传入的文档类型覆盖；strategy="auto" 时等价于 auto_route=True）
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_route = auto_route or (strategy == "auto")
        self.splitter = self._create_splitter()
        _vprint(f"[SPLITTER][DEBUG] TextSplitter 门面创建完成: strategy={strategy}, auto_route={self.auto_route}, "
              f"chunk_size={chunk_size}, overlap={chunk_overlap}, 实际分割器={type(self.splitter).__name__}")

    @staticmethod
    def pick_strategy(file_type: str = "", file_path: str = "", text: str = "") -> str:
        """
        根据文档类型/文件路径/内容智能选择分割策略（按文档类型路由）。

        设计决策：把"文档类型 -> 策略"的路由规则集中在这里，单一数据源，
        便于维护与扩展。路由优先级：显式扩展名映射 > 文件名扩展名推断 > 内容嗅探 > 默认。

        参数:
            file_type: 文档类型标识（如 "pdf"/"docx"/"md"/"xlsx"/"txt"/"py" 等）
            file_path: 文件路径（用于在 file_type 缺失时从扩展名推断）
            text: 文本内容（用于无类型信息时做简单内容嗅探）

        返回:
            str: 推荐的分割策略名称
        """
        # 1) 优先使用显式 file_type（小写、去点）
        ft = (file_type or "").strip().lower().lstrip(".")
        if not ft and file_path:
            # 2) 从文件路径/文件名推断扩展名
            ext = os.path.splitext(file_path)[1].lstrip(".").lower()
            ft = ext
        if ft and ft in _FILE_TYPE_STRATEGY_MAP:
            return _FILE_TYPE_STRATEGY_MAP[ft]

        # 3) 内容嗅探：无明显类型时，若内容呈代码特征则用代码切分，否则默认语义
        if text:
            code_hint = _guess_code_language(text)
            if code_hint:
                return "code"

        # 4) 默认：语义切分（检索质量优先）
        return _DEFAULT_ROUTE_STRATEGY

    def _create_splitter(self) -> BaseSplitter:
        """
        根据策略创建对应的分割器实例（内部方法）
        
        返回:
            BaseSplitter: 分割器实例
        """
        # ---- 自研策略 ----
        if self.strategy == "semantic":
            _vprint("[SPLITTER] 选择分割策略: 自研语义分割 (semantic)")
            return SemanticSplitter(self.chunk_size, self.chunk_overlap)
        elif self.strategy == "code":
            _vprint("[SPLITTER] 选择分割策略: 自研代码感知分割 (code)")
            return CodeSplitter(self.chunk_size, self.chunk_overlap)

        # ---- LangChain 策略 ----
        elif self.strategy == "langchain_recursive":
            _vprint("[SPLITTER] 选择分割策略: LangChain 递归字符分割 (langchain_recursive)")
            return LangChainRecursiveSplitter(self.chunk_size, self.chunk_overlap)
        elif self.strategy == "langchain_token":
            _vprint("[SPLITTER] 选择分割策略: LangChain Token 分割 (langchain_token)")
            return LangChainTokenSplitter(self.chunk_size, self.chunk_overlap)
        elif self.strategy == "langchain_markdown":
            _vprint("[SPLITTER] 选择分割策略: LangChain Markdown 分割 (langchain_markdown)")
            return LangChainMarkdownSplitter(self.chunk_size, self.chunk_overlap)
        elif self.strategy == "langchain_semantic":
            _vprint("[SPLITTER] 选择分割策略: LangChain 语义分割 (langchain_semantic)")
            #调用时 没有传 embeddings 参数，是因为构造函数中 self.embeddings 默认为 None ， 
            # split_text 里会通过 _get_project_embeddings_adapter() 自动获取项目的 bge-small-zh-v1.5 模型。
            # 降级保护链：
            #    1、外部传了 embeddings =》 用外部的
            #    2、没传 + embedding_service 可用 =》 用 bge-small-zh-v1.5
            #    3、没传 + embedding_service 不可用 =》 降级为 RecursiveCharacterSplitter 递归分割
            return LangChainSemanticSplitter(self.chunk_size, self.chunk_overlap)

        else:
            # 默认使用自研递归字符分割策略
            _vprint(f"[SPLITTER] 选择分割策略: 自研递归字符分割 (recursive，未知策略 {self.strategy!r} 回退默认)")
            return RecursiveCharacterSplitter(self.chunk_size, self.chunk_overlap)

    def split_text(self, text: str) -> List[TextChunk]:
        """
        分割文本
        
        参数:
            text: 待分割的文本内容
            
        返回:
            List[TextChunk]: 分割后的文本块列表
        """
        _vprint(f"[SPLITTER][DEBUG] 门面 split_text 调用: 输入字符数={len(text)}")
        chunks = self._split_text_protected(self.splitter, text)
        _vprint(f"[SPLITTER] 门面 split_text 完成: 策略={self.strategy}, 输入={len(text)}字符, 输出={len(chunks)}块")
        return chunks

    def _split_text_protected(self, splitter: "BaseSplitter", text: str) -> List[TextChunk]:
        """
        带 Mermaid 围栏保护的文本分割（门面统一入口）。

        切片前把 ```mermaid 围栏块替换为等长占位符，切片后还原，
        保证 mermaid 代码块完整落在同一个 chunk（不被任何分割策略切碎）。
        文本不含 mermaid 时完全走原逻辑，行为不变（零影响）。
        """
        protected, fence_map = _protect_mermaid_fences(text)
        chunks = splitter.split_text(protected)
        if fence_map:
            _restore_mermaid_fences(chunks, fence_map)
        return chunks

    def split_document(self, document) -> List[TextChunk]:
        """
        分割文档对象
        
        参数:
            document: 文档对象（需包含content属性，建议同时含 file_type/file_path 用于自动路由）
            
        返回:
            List[TextChunk]: 分割后的文本块列表
        """
        # ---- 结构化解析分支（docstruct）----
        # 当 document 带 structure（结构化中间产物，由 DocumentLoader.load(structured=True)
        # 填充）时，优先使用结构感知切片器：
        #   表格 -> 整表完整切片（可回答跨行列问题）
        #   流程图 -> 箭头关系文本化（可回答"什么条件下转诊"类问题）
        #   正文 -> 带章节锚点（可溯源"这是哪部分第几节"）
        # 切片失败自动降级旧文本链路，不阻断入库。
        structure = getattr(document, "structure", None)
        if structure is not None:
            chunks = self._split_structured_document(document)
            if chunks is not None:
                return chunks
            _vprint(f"[SPLITTER][WARN] 结构化切片失败，降级旧文本链路")

        content = document.content
        _vprint(f"[SPLITTER][DEBUG] 门面 split_document 调用: 文档内容字符数={len(content)}")

        # ---- 按文档类型自动路由策略 ----
        # 当启用 auto_route 时，根据文档 file_type/file_path/content 动态挑选最合适的策略，
        # 并创建一个临时分割器（不污染门面的默认 splitter）。
        if self.auto_route:
            file_type = getattr(document, "file_type", "") or ""
            file_path = getattr(document, "file_path", "") or ""
            _vprint(f"[SPLITTER][DEBUG] split_document 进入自动路由分支: auto_route=True, "
                  f"file_type={file_type!r}, file_path={file_path!r}")
            routed = self.pick_strategy(file_type=file_type, file_path=file_path, text=content)
            _vprint(f"[SPLITTER][DEBUG] pick_strategy 计算结果: routed={routed!r}, 原strategy={self.strategy!r}")
            if routed != self.strategy:
                _vprint(f"[SPLITTER] 门面 split_document 按文档类型路由策略: {self.strategy!r} -> {routed!r} "
                      f"(file_type={file_type or file_path or 'unknown'!r})")
            tmp = self._create_splitter_for(routed)
            chunks = self._split_text_protected(tmp, content)
            _vprint(f"[SPLITTER] 门面 split_document 完成: 路由策略={routed}, 输入={len(content)}字符, 输出={len(chunks)}块")
            self._export_chunk_metadata(document, chunks)
            return chunks

        chunks = self._split_text_protected(self.splitter, content)
        _vprint(f"[SPLITTER] 门面 split_document 完成: 策略={self.strategy}, 输入={len(content)}字符, 输出={len(chunks)}块")
        self._export_chunk_metadata(document, chunks)
        return chunks

    def _export_chunk_metadata(self, document, chunks) -> None:
        """解析日志：导出向量存储元数据到日志目录 metadata.json。

        每个切片一条记录（含全部元数据字段，与向量库实际入库元数据一致）。
        仅当 document 带 parse_log_dir（DocumentLoader.load 填充）时导出；
        日志写入失败不影响入库，只打印警告。
        """
        log_dir = getattr(document, "parse_log_dir", "") or ""
        if not log_dir:
            return
        try:
            from backend.rag.parse_logger import write_chunk_metadata
            write_chunk_metadata(log_dir, chunks)
        except Exception as exc:
            _vprint(f"[SPLITTER][WARN] 向量元数据日志导出失败: {exc}")

    def _split_structured_document(self, document) -> Optional[List[TextChunk]]:
        """
        使用结构感知切片器对结构化文档切片（docstruct-PDF / markdown 双路集成点）。

        分派规则：
            - document.structure.splitter_key == "md"  -> MdStructuredSplitter
              （Markdown：先图意化图片，再按表格/代码/图片/正文切片）
            - 其余                                  -> docstruct StructuredSplitter（PDF）

        参数:
            document: 带 .structure 的 Document 对象

        返回:
            List[TextChunk] 或 None（失败时返回 None，由调用方降级旧文本链路）
        """
        _vprint("[SPLITTER][DEBUG] 检测到结构化产物，使用结构感知切片器")
        structure = getattr(document, "structure", None)
        try:
            from backend.config.settings import settings

            if getattr(structure, "splitter_key", "") == "md":
                return self._split_md_document(document, settings)

            # ---- docstruct（PDF）路径 ----
            # 延迟导入：docstruct 库位于 aicode/lib 下，由 sys.path 注入
            from docstruct.slice.structured_splitter import StructuredSplitter

            splitter = StructuredSplitter(
                chunk_size=settings.STRUCTURED_PARSER_CHUNK_SIZE,
                chunk_overlap=settings.STRUCTURED_PARSER_CHUNK_OVERLAP,
                max_table_tokens=settings.STRUCTURED_PARSER_MAX_TABLE_TOKENS,
                embed_safe_tokens=getattr(
                    settings, "STRUCTURED_PARSER_EMBED_SAFE_TOKENS", 400),
            )
            raw_chunks = splitter.split_document(document)
            chunks = _to_text_chunks(raw_chunks)
            _vprint(f"[SPLITTER] 结构感知切片完成: {len(chunks)}块")
            self._export_chunk_metadata(document, chunks)
            return chunks
        except Exception as exc:
            # 任何异常都不阻断入库（返回 None 触发降级旧链路）
            _vprint(f"[SPLITTER][WARN] 结构感知切片失败: {exc}")
            return None

    def _split_md_document(self, document, settings) -> Optional[List[TextChunk]]:
        """
        Markdown 结构化切片（新增集成点）：

            1. 图片图意化（OCR/VL + 落盘 + md5 缓存）：此刻 document.id 已由
               IncrementalUpdater 分配，落盘目录以 doc_id 命名；
            2. 图意化后把 doc.content 渲染为纯文本（含图意文本），供图谱/旧链路受益；
            3. MdStructuredSplitter 结构感知切片；
            4. 回写 md 解析日志 info.json（图片统计以最终态为准）。

        任何一步失败都返回 None（由门面降级旧文本链路），不阻断入库。
        """
        try:
            # 1) 图片图意化（幂等：MdStruct.images_processed 防重复）
            from backend.rag.markdown_image_service import (
                enhance_document_images, render_document_content,
            )
            image_stats = enhance_document_images(document)
            structure = getattr(document, "structure", None)
            if structure is not None:
                try:
                    structure.image_stats = image_stats or {}
                except Exception:
                    pass
                # 2) doc.content 渲染为纯文本（含 OCR/VL 图意文本）
                rendered = render_document_content(document)
                if rendered:
                    document.content = rendered

            # 3) 结构感知切片
            from backend.rag.md_structured_splitter import MdStructuredSplitter
            splitter = MdStructuredSplitter(
                chunk_size=settings.STRUCTURED_PARSER_CHUNK_SIZE,
                chunk_overlap=settings.STRUCTURED_PARSER_CHUNK_OVERLAP,
                max_table_tokens=settings.STRUCTURED_PARSER_MAX_TABLE_TOKENS,
                embed_safe_tokens=getattr(
                    settings, "STRUCTURED_PARSER_EMBED_SAFE_TOKENS", 400),
            )
            chunks = splitter.split_structure(document)
            if not chunks:
                _vprint("[SPLITTER][md] 结构化切片产物为空，降级旧文本链路")
                return None
            _vprint(f"[SPLITTER][md] 结构感知切片完成: {len(chunks)}块")
            self._export_chunk_metadata(document, chunks)
            # 4) 回写 md 日志（图片统计以图意化最终态为准）
            try:
                log_dir = getattr(document, "parse_log_dir", "") or ""
                if log_dir:
                    from backend.rag.parse_logger import finalize_md_parse_log
                    finalize_md_parse_log(log_dir, document)
            except Exception:
                pass
            return chunks
        except Exception as exc:
            _vprint(f"[SPLITTER][md] 结构化切片失败（降级旧链路）: {exc}")
            return None

    def _create_splitter_for(self, strategy: str) -> "BaseSplitter":
        """
        根据指定策略名称创建分割器实例（供按文档类型自动路由时临时使用）。

        参数:
            strategy: 目标分割策略名称

        返回:
            BaseSplitter: 分割器实例
        """
        saved = self.strategy
        self.strategy = strategy
        try:
            return self._create_splitter()
        finally:
            self.strategy = saved


def _to_text_chunks(raw_chunks) -> List[TextChunk]:
    """
    把 docstruct 的 Chunk 转换为现有 TextChunk 格式（新增适配器）。

    docstruct 库为了保持"与 backend 解耦"，内部使用自己的 Chunk 结构；
    入库链路需要的是本文件定义的 TextChunk。两者字段一一对应，
    这里做一层轻量转换，并保留全部 metadata（含 section_path 等结构信息，
    它们会随 _prepare_document_data 自动并入向量库）。

    参数:
        raw_chunks: docstruct 切片产物（List[docstruct.schemas.Chunk]）

    返回:
        List[TextChunk]: 现有格式的切片列表
    """
    total = len(raw_chunks)
    result = []
    for chunk in raw_chunks:
        result.append(TextChunk(
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            total_chunks=total,
            start_pos=chunk.start_pos,
            end_pos=chunk.end_pos,
            metadata=dict(chunk.metadata),  # 拷贝一份，避免共享引用
        ))
    return result