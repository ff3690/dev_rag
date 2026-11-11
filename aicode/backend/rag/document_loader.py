"""
文档加载器模块 - Document Loader Module

该模块负责加载和解析多种格式的研发文档，包括PDF、Word、Markdown、Excel、纯文本等。
采用策略模式，每种文档格式对应一个Loader实现类，通过DocumentLoader统一入口调用。

设计决策：
1. 使用抽象基类(BaseLoader)定义统一接口，便于扩展新的文档格式
2. pandas采用可选导入，避免安装失败影响其他功能
3. Document数据类使用dataclass，简洁且支持类型提示
4. 批量加载时捕获异常继续处理，提高容错性
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
from docx import Document as DocxDocument
import re
import tempfile
import time
import warnings
import sys
import os

# 添加 lib 目录到 sys.path（存放本地安装的第三方库）
_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'lib')
if os.path.exists(_LIB_DIR) and _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# PDF 解析库配置：按优先级排序
# 1. pymupdf (fitz) - 兼容性最好，能处理各种格式的 PDF
# 2. pypdf - PyPDF2 的维护分支，修复了多个 bug
# 3. PyPDF2 - 经典方案，最后回退

PDF_LIB = None
PdfReader = None
FitzReader = None

# 尝试导入 pymupdf（最佳兼容性）
try:
    import fitz as FitzReader
    PDF_LIB = "pymupdf"
except ImportError:
    pass

# 尝试导入 pypdf
if PDF_LIB is None:
    try:
        from pypdf import PdfReader
        PDF_LIB = "pypdf"
    except ImportError:
        pass

# 最后回退到 PyPDF2
if PDF_LIB is None:
    try:
        from PyPDF2 import PdfReader
        PDF_LIB = "PyPDF2"
    except ImportError:
        raise ImportError("未找到任何可用的 PDF 解析库，请安装 pymupdf、pypdf 或 PyPDF2")

# pandas可选导入：Excel文件解析需要pandas，若未安装则该功能不可用
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

# OCR处理器可选导入：图像文本识别需要OCRProcessor，若未安装则该功能不可用
try:
    from backend.rag.ocr_processor import OCRProcessor
    OCR_AVAILABLE = True
    _ocr_processor = OCRProcessor(lang="chi_sim")
except Exception:
    OCR_AVAILABLE = False
    _ocr_processor = None


def extract_text_from_image(image_path: str) -> str:
    """
    从图像文件中提取文本（使用OCR）
    
    参数:
        image_path: 图像文件路径
        
    返回:
        str: 提取的文本内容，OCR不可用时返回空字符串
    """
    if not OCR_AVAILABLE or not _ocr_processor:
        return ""
    
    try:
        return _ocr_processor.extract_text(image_path)
    except Exception as e:
        print(f"OCR processing failed for {image_path}: {str(e)}")
        return ""


def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    """
    从图像字节流提取文本（内存处理，不落临时文件）
    
    用于缺陷3修复：图片在内存中直接 OCR，避免临时文件往返。
    
    参数:
        img_bytes: 图片二进制内容
        
    返回:
        str: 提取的文本内容，OCR不可用/失败时返回空字符串
    """
    if not OCR_AVAILABLE or not _ocr_processor:
        print("[OCR] 内存 OCR 不可用（OCR_AVAILABLE=False 或处理器初始化失败），返回空文本")
        return ""
    try:
        # 委托给 OCRProcessor 的内存接口（含灰度/对比度/放大预处理）。
        # 必须用包内相对导入：服务以 backend.main:app 从 aicode 目录启动时，
        # 写死 aicode.backend.* 会抛 ModuleNotFoundError（曾导致内存 OCR 恒失败、恒返回空）。
        from .ocr_processor import extract_text_from_image_bytes as _ocr_bytes
        return _ocr_bytes(img_bytes, lang=_ocr_processor.lang).strip()
    except Exception as e:
        print(f"[OCR] ✗ 内存 OCR 调用失败: {type(e).__name__}: {e}")
        return ""


def pytesseract_image_to_string(img) -> str:
    """
    对 PIL Image 对象执行 pytesseract 识别（延迟导入，避免顶层依赖）
    """
    import pytesseract
    return pytesseract.image_to_string(img, lang=_ocr_processor.lang)


@dataclass
class Document:
    """
    文档数据类 - Document Data Class
    
    封装解析后的文档信息，包含基本属性和元数据。
    
    属性:
        id: 文档唯一标识（UUID）
        title: 文档标题（默认取文件名）
        content: 文档内容（纯文本）
        file_path: 文件路径
        file_type: 文件类型（pdf/docx/md/xlsx/txt）
        category: 文档分类（接口文档/故障排查/需求文档/架构图纸）
        department: 所属部门
        pages: PDF页数（其他类型为0）
        metadata: 元数据字典（包含页数、段落数、行数等统计信息）
    """
    id: Optional[str] = None
    title: str = ""
    content: str = ""
    file_path: str = ""
    file_type: str = ""
    category: str = ""
    department: str = ""
    pages: int = 0
    metadata: dict = None
    # 结构化解析中间产物（docstruct），默认 None = 未启用/解析失败。
    # 仅 PDF + 显式开启结构化解析时才会被填充（见 DocumentLoader.load）。
    # 填充后，TextSplitter.split_document 会优先使用结构感知切片器。
    structure: object = None
    # 本 PDF 的解析统计日志目录（见 parse_logger.py），空字符串 = 未启用/写入失败。
    # 由 DocumentLoader.load 填充，切片阶段据此导出向量存储元数据。
    parse_log_dir: str = ""

    def __post_init__(self):
        """初始化后处理：确保metadata不为None"""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentChunk:
    """
    文档分块数据类 - Document Chunk Data Class
    
    封装文档分割后的单个块信息，用于后续向量化存储。
    
    属性:
        document_id: 所属文档ID
        chunk_id: 块唯一标识
        content: 块内容
        chunk_index: 块索引（从0开始）
        total_chunks: 总块数
        metadata: 元数据（包含位置信息等）
    """
    document_id: str
    chunk_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict = None

    def __post_init__(self):
        """初始化后处理：确保metadata不为None"""
        if self.metadata is None:
            self.metadata = {}


class BaseLoader(ABC):
    """
    文档加载器抽象基类 - Document Loader Abstract Base Class
    
    定义所有文档加载器必须实现的接口方法。
    
    设计决策：使用ABC抽象基类强制子类实现统一接口，便于扩展和维护。
    """

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        加载文档并返回Document对象
        
        参数:
            file_path: 文件路径
        
        返回:
            Document: 解析后的文档对象
        """
        pass

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """
        判断是否支持指定文件类型
        
        参数:
            file_type: 文件扩展名（不含点号）
        
        返回:
            bool: 是否支持
        """
        pass


class PDFLoader(BaseLoader):
    """
    PDF文档加载器 - PDF Document Loader
    
    使用多种 PDF 解析库（pymupdf/pypdf/PyPDF2）解析PDF文件，
    提取文本内容和页码信息，同时支持图片提取和OCR识别。
    
    设计决策：
    1. 优先使用 pymupdf (fitz)，兼容性最好，能处理各种格式的 PDF
    2. 自动回退到 pypdf 或 PyPDF2
    3. 添加逐页异常处理，单页失败不影响其他页
    4. 支持扫描版 PDF（仅有图片，无文本层）
    """

    def load(self, file_path: str) -> Document:
        """
        加载PDF文档
        
        参数:
            file_path: PDF文件路径
        
        返回:
            Document: 包含文本内容和页码信息的文档对象
        """
        content = ""
        image_text = ""
        image_count = 0
        total_pages = 0
        processed_pages = 0
        
        # 根据可用库选择解析方式
        if PDF_LIB == "pymupdf":
            content, image_text, image_count, total_pages, processed_pages = self._load_with_pymupdf(file_path)
        elif PDF_LIB in ("pypdf", "PyPDF2"):
            content, image_text, image_count, total_pages, processed_pages = self._load_with_pypdf(file_path)
        else:
            # 不应该到达这里，但为安全起见
            raise RuntimeError(f"未找到可用的 PDF 解析库")
        
        # 合并文本内容和图片OCR内容
        full_content = content + image_text
        
        return Document(
            title=Path(file_path).name,# 取文件名（含扩展名）作为标题
            content=full_content.strip(),
            file_path=file_path,
            file_type="pdf",
            pages=total_pages,
            metadata={
                "pages": total_pages, 
                "processed_pages": processed_pages,
                "images_found": image_count, 
                "images_with_text": len(image_text) > 0,
                "pdf_library": PDF_LIB
            }
        )
    
    def _load_with_pymupdf(self, file_path: str) -> tuple:
        """
        使用 pymupdf (fitz) 加载 PDF
        
        pymupdf 兼容性最好，能处理各种格式的 PDF，
        包括扫描版 PDF（通过 OCR 提取图片中的文本）。
        
        缺陷2/3修复（所有开关默认 False，关闭时行为与旧版本完全一致）：
          - LEGACY_PDF_LAYOUT_ORDER: page.get_text(sort=True) 按坐标排序，避免跨栏/表格错乱
          - LEGACY_PDF_HEADER_FOOTER_FILTER: 两遍法剔除页眉页脚（出现在 ≥80% 页面的重复行）
          - LEGACY_PDF_OCR_INLINE: OCR 文本按页插入（紧跟对应页正文），而非追加全文末尾
          - LEGACY_PDF_OCR_MIN_SIZE: 尺寸过滤，小于阈值的水印/logo/图标跳过 OCR
          - LEGACY_PDF_OCR_PARALLEL: 多线程并行 OCR
          - OCR 全程内存处理（BytesIO），不再落临时文件
        """
        from backend.config.settings import settings  # 延迟导入，避免顶层循环依赖

        content = ""
        image_text = ""
        image_count = 0
        processed_pages = 0
        
        # ---- 读取增强开关（全部默认 False，保持旧行为）----
        layout_order = getattr(settings, "LEGACY_PDF_LAYOUT_ORDER", False)
        hf_filter = getattr(settings, "LEGACY_PDF_HEADER_FOOTER_FILTER", False)
        ocr_inline = getattr(settings, "LEGACY_PDF_OCR_INLINE", False)
        ocr_min_size = getattr(settings, "LEGACY_PDF_OCR_MIN_SIZE", 50)
        ocr_parallel = getattr(settings, "LEGACY_PDF_OCR_PARALLEL", False)
        
        try:
            doc = FitzReader.open(file_path)
            total_pages = len(doc)
            
            # ---- 第一遍：收集每页原始文本行（页眉页脚过滤需全量统计）----
            page_raw_lines = []
            for page_idx in range(total_pages):
                try:
                    page = doc[page_idx]
                    # 缺陷2：阅读顺序（sort=True 按坐标排序，避免跨栏/表格错乱）
                    text = page.get_text(sort=layout_order)
                    page_raw_lines.append(text.splitlines())
                    processed_pages += 1
                except Exception as e:
                    print(f"Warning: Failed to process page {page_idx+1}: {str(e)}")
                    page_raw_lines.append([])
                    continue
            
            # ---- 页眉页脚候选行：出现在 ≥80% 页面的重复行（两遍法）----
            header_footer_lines = set()
            if hf_filter and total_pages >= 3:
                from collections import Counter
                line_freq = Counter()
                for lines in page_raw_lines:
                    seen_in_page = set()
                    for ln in lines:
                        stripped = ln.strip()
                        if stripped:
                            seen_in_page.add(stripped)
                    for s in seen_in_page:
                        line_freq[s] += 1
                threshold = max(2, int(total_pages * 0.8))
                header_footer_lines = {
                    s for s, cnt in line_freq.items()
                    if cnt >= threshold and len(s) <= 80  # 页眉页脚一般较短
                }
            
            # ---- 第二遍：拼接正文 + 收集 OCR 任务（图片去噪）----
            ocr_tasks = []  # (page_idx, img_bytes, img_ext)
            seen_img_md5 = set()
            for page_idx, lines in enumerate(page_raw_lines):
                if hf_filter:
                    lines = [ln for ln in lines if ln.strip() not in header_footer_lines]
                page_content = "\n".join(lines).strip()
                if page_content:
                    content += page_content + "\n\n"
                
                # 图片提取与去噪
                if OCR_AVAILABLE:
                    try:
                        page = doc[page_idx]
                        images = page.get_images(full=True)
                        for img_info in images:
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                img_bytes = base_image["image"]
                                img_ext = base_image.get("ext", "png")
                                
                                # 去噪规则1：md5 去重（同一背景图/logo 反复出现时只 OCR 一次）
                                import hashlib
                                digest = hashlib.md5(img_bytes).hexdigest()
                                if digest in seen_img_md5:
                                    continue
                                seen_img_md5.add(digest)
                                
                                # 去噪规则2：尺寸过滤（水印/图标/装饰图通常很小）
                                if ocr_min_size > 0:
                                    try:
                                        from PIL import Image as PILImage
                                        import io as _io
                                        with PILImage.open(_io.BytesIO(img_bytes)) as pimg:
                                            w, h = pimg.size
                                            if w < ocr_min_size or h < ocr_min_size:
                                                continue
                                    except Exception:
                                        pass  # 尺寸无法解析时保守放行
                                
                                image_count += 1
                                ocr_tasks.append((page_idx, img_bytes, img_ext))
                            except Exception as e:
                                print(f"Failed to process image on page {page_idx+1}: {str(e)}")
                    except Exception as e:
                        print(f"Warning: Failed to extract images from page {page_idx+1}: {str(e)}")
            
            # ---- OCR 执行（内存处理，可选并行）----
            ocr_results = {}  # page_idx -> [text, ...]
            if ocr_tasks:
                def _ocr_one(task):
                    pidx, bts, _ext = task
                    return pidx, extract_text_from_image_bytes(bts)
                
                if ocr_parallel and len(ocr_tasks) > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=min(4, len(ocr_tasks))) as ex:
                        for pidx, txt in ex.map(_ocr_one, ocr_tasks):
                            if txt:
                                ocr_results.setdefault(pidx, []).append(txt)
                else:
                    for task in ocr_tasks:
                        pidx, txt = _ocr_one(task)
                        if txt:
                            ocr_results.setdefault(pidx, []).append(txt)
            
            # ---- 拼接 OCR 结果：按页插入 或 追加全文末尾 ----
            for page_idx in range(total_pages):
                if page_idx not in ocr_results:
                    continue
                for ocr_t in ocr_results[page_idx]:
                    if ocr_inline:
                        # 缺陷3：OCR 紧跟对应页正文之后（保持版面上下文）
                        content += f"\n\n--- 第{page_idx+1}页图片OCR识别内容 ---\n{ocr_t}\n"
                    else:
                        image_text += f"\n\n--- 第{page_idx+1}页图片OCR识别内容 ---\n{ocr_t}\n"
            
            doc.close()
            return content, image_text, image_count, total_pages, processed_pages
            
        except Exception as e:
            print(f"Failed to read PDF file with pymupdf: {file_path}, error: {str(e)}")
            return "", "", 0, 0, 0
    
    def _load_with_pypdf(self, file_path: str) -> tuple:
        """
        使用 pypdf/PyPDF2 加载 PDF（回退方案）
        
        当 pymupdf 不可用时使用，兼容性稍差。
        """
        content = ""
        image_text = ""
        image_count = 0
        processed_pages = 0
        
        # 过滤已知警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
            except Exception as e:
                print(f"Failed to read PDF file with {PDF_LIB}: {file_path}, error: {str(e)}")
                return "", "", 0, 0, 0
        
            # 逐页提取文本
            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        content += text.strip() + "\n\n"
                    processed_pages += 1
                    
                    # 提取页面中的图片并进行OCR识别
                    if OCR_AVAILABLE:
                        try:
                            if hasattr(page, 'images'):
                                for img_idx, img in enumerate(page.images):
                                    image_count += 1
                                    temp_path = None
                                    try:
                                        img_data = img.data if hasattr(img, 'data') else img.get_data()
                                        img_name = f"image_{page_idx}_{img_idx}"
                                        if hasattr(img, 'name') and img.name:
                                            img_name = img.name
                                        
                                        with tempfile.NamedTemporaryFile(
                                            suffix=Path(img_name).suffix or '.png', 
                                            delete=False
                                        ) as f:
                                            f.write(img_data)
                                            temp_path = f.name
                                        
                                        ocr_result = extract_text_from_image(temp_path)
                                        if ocr_result:
                                            image_text += f"\n\n--- 图片{image_count}(第{page_idx+1}页) OCR识别内容 ---\n{ocr_result}\n"
                                    except Exception as e:
                                        print(f"Failed to process image {image_count} on page {page_idx+1}: {str(e)}")
                                    finally:
                                        if temp_path and Path(temp_path).exists():
                                            try:
                                                Path(temp_path).unlink()
                                            except Exception:
                                                pass
                        except Exception as e:
                            print(f"Warning: Failed to extract images from page {page_idx+1}: {str(e)}")
                            
                except Exception as e:
                    print(f"Warning: Failed to process page {page_idx+1}: {str(e)}")
                    continue
        
        return content, image_text, image_count, total_pages, processed_pages

    def supports(self, file_type: str) -> bool:
        """判断是否支持PDF类型"""
        return file_type.lower() == "pdf"


class DocxLoader(BaseLoader):
    """
    Word文档加载器 - Word Document Loader
    
    使用python-docx库解析.docx文件，提取段落和表格内容。
    同时支持从Word中提取图片并进行OCR识别。
    
    设计决策：python-docx是处理Word文档的标准库，能保留段落结构和表格数据。
    图片OCR作为可选功能，OCR不可用时仅提取文本。
    """

    def _extract_images_from_docx(self, doc: DocxDocument) -> tuple:
        """
        从Word文档中提取所有图片并进行OCR识别（内部方法）
        
        参数:
            doc: Word文档对象
        
        返回:
            tuple: (image_text, image_count) 图片OCR文本和图片数量
        """
        if not OCR_AVAILABLE:
            return "", 0

        image_text = ""
        image_count = 0

        # 遍历文档部件中的所有相关资源，提取图像
        for rel_id, part in doc.part.related_parts.items():
            # 检查是否为图像资源（通过 MIME 类型判断）
            if part.content_type and part.content_type.startswith('image/'):
                image_count += 1
                temp_path = None
                try:
                    # 获取图像二进制数据
                    image_bytes = part.blob
                    # 根据 content_type 确定文件扩展名
                    ext = part.content_type.split('/')[-1]
                    if ext == 'jpeg':
                        ext = 'jpg'
                    # 保存为临时文件
                    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                        f.write(image_bytes)
                        temp_path = f.name
                    # 调用 OCR 提取文本
                    ocr_result = extract_text_from_image(temp_path)
                    if ocr_result:
                        image_text += f"\n\n--- 图片{image_count} Word文件中OCR识别内容 ---\n{ocr_result}\n"
                except Exception as e:
                    print(f"Failed to process related image {image_count} in DOCX: {str(e)}")
                finally:
                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink()

        return image_text, image_count

    def load(self, file_path: str) -> Document:
        """
        加载Word文档
        
        参数:
            file_path: Word文件路径
        
        返回:
            Document: 包含段落和表格内容的文档对象
        """
        doc = DocxDocument(file_path)
        content = ""
        
        # 提取段落内容
        for para in doc.paragraphs:
            content += para.text + "\n\n"
        
        # 提取表格内容，单元格用制表符分隔
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join(cell.text for cell in row.cells)
                content += row_text + "\n"
            content += "\n"
        
        # 提取图片并进行OCR识别
        image_text, image_count = self._extract_images_from_docx(doc)
        
        # 合并文本内容和图片OCR内容
        full_content = content + image_text
        
        return Document(
            title=Path(file_path).name,  # 使用 .name 包含文件扩展名（如 .pdf、.docx）
            content=full_content.strip(),
            file_path=file_path,
            file_type="docx",
            metadata={"paragraphs": len(doc.paragraphs), "tables": len(doc.tables), "images_found": image_count, "images_with_text": len(image_text) > 0}
        )

    def supports(self, file_type: str) -> bool:
        """判断是否支持docx类型"""
        return file_type.lower() == "docx"


class MarkdownLoader(BaseLoader):
    """
    Markdown文档加载器 - Markdown Document Loader
    
    直接读取MD文件内容，保留原始格式（含编码回退：utf-8-sig -> gb18030 -> latin-1）。

    设计决策：
      1. Markdown格式简单，轻量读取即可，无需额外依赖；
      2. 编码回退保证非 UTF-8 的 md（如 gbk 导出）也能正常入库；
      3. 仅在 MD_STRUCTURED_PARSING_ENABLED 开启时补充块级统计 metadata
         （关开关时保持与旧版本完全一致：仅 line_count）。
      4. 块级结构化解析产物（MdStruct）由 DocumentLoader.load(structured=True) 填充，
         不在此处执行，避免重复解析。
    """

    @staticmethod
    def _read_text_robust(file_path: str) -> str:
        """读取 md 文本，含编码回退（utf-8-sig -> gb18030 -> latin-1）。"""
        with open(file_path, "rb") as f:
            raw = f.read()
        if not raw:
            return ""
        for enc in ("utf-8-sig", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        # 最后兜底 latin-1：永不失败，但仅用于极端无法解码的文件
        return raw.decode("latin-1", errors="replace")

    @staticmethod
    def _quick_stats(content: str) -> dict:
        """轻量块级统计（供 metadata 展示，不做完整解析）。"""
        lines = content.split("\n")
        headings = sum(1 for ln in lines if re.match(r"^#{1,6}\s", ln))
        fences = sum(1 for ln in lines if re.match(r"^\s*(```|~~~)", ln))
        code_blocks = fences // 2
        tables = sum(1 for ln in lines
                      if re.match(r"^[\s|:|-]+$", ln) and "|" in ln and "-" in ln)
        images = content.count("![")
        return {
            "headings": headings,
            "tables": tables,
            "code_blocks": code_blocks,
            "images_found": images,
            "line_count": len(lines),
        }

    def load(self, file_path: str) -> Document:
        """
        加载Markdown文档
        
        参数:
            file_path: Markdown文件路径
        
        返回:
            Document: 包含原始内容的文档对象
        """
        content = self._read_text_robust(file_path)

        metadata = {"line_count": content.count("\n") + 1}
        # 开启 md 结构化时补充块级统计（关闭时保持旧行为逐字节不变）
        try:
            from backend.config.settings import settings
            if getattr(settings, "MD_STRUCTURED_PARSING_ENABLED", False):
                metadata.update(self._quick_stats(content))
        except Exception:
            pass  # 配置读取失败仅跳过统计，不影响主流程

        return Document(
            title=Path(file_path).name,  # 使用 .name 包含文件扩展名（如 .pdf、.docx）
            content=content,
            file_path=file_path,
            file_type="md",
            metadata=metadata
        )

    def supports(self, file_type: str) -> bool:
        """判断是否支持md/markdown类型"""
        return file_type.lower() == "md" or file_type.lower() == "markdown"


class ExcelLoader(BaseLoader):
    """
    Excel文档加载器 - Excel Document Loader
    
    使用pandas解析.xlsx/.xls文件，将表格数据转换为文本。
    
    设计决策：pandas是数据处理的标准库，能高效读取和转换Excel数据。
    注意：pandas为可选依赖，未安装时会抛出ImportError。
    """

    def load(self, file_path: str) -> Document:
        """
        加载Excel文档
        
        参数:
            file_path: Excel文件路径
        
        返回:
            Document: 包含表格文本内容的文档对象
        
        异常:
            ImportError: 当pandas未安装时抛出
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel file processing. Please install pandas.")
        
        df = pd.read_excel(file_path)
        content = df.to_string(index=False)  # 转换为文本，不显示索引列
        
        return Document(
            title=Path(file_path).name,  # 使用 .name 包含文件扩展名（如 .pdf、.docx）
            content=content,
            file_path=file_path,
            file_type="xlsx",
            metadata={"rows": len(df), "columns": len(df.columns)}
        )

    def supports(self, file_type: str) -> bool:
        """判断是否支持xlsx/xls类型"""
        return file_type.lower() == "xlsx" or file_type.lower() == "xls"


class TextLoader(BaseLoader):
    """
    纯文本文件加载器 - Text File Loader
    
    直接读取TXT文件内容。
    
    设计决策：纯文本文件是最基础的格式，无需复杂解析。
    """

    def load(self, file_path: str) -> Document:
        """
        加载纯文本文件
        
        参数:
            file_path: 文本文件路径
        
        返回:
            Document: 包含文本内容的文档对象
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return Document(
            title=Path(file_path).name,  # 使用 .name 包含文件扩展名（如 .pdf、.docx）
            content=content,
            file_path=file_path,
            file_type="txt",
            metadata={"line_count": content.count("\n") + 1}
        )

    def supports(self, file_type: str) -> bool:
        """判断是否支持txt类型"""
        return file_type.lower() == "txt"


class ImageLoader(BaseLoader):
    """
    图像文件加载器 - Image File Loader
    
    使用OCR技术从图像中提取文本内容，支持jpg、png、bmp等常见格式。
    
    设计决策：OCR功能作为可选依赖，未安装时返回空内容，不影响其他功能。
    """

    def load(self, file_path: str) -> Document:
        """
        加载图像文件并提取文本
        
        参数:
            file_path: 图像文件路径
        
        返回:
            Document: 包含OCR识别文本的文档对象
        """
        content = extract_text_from_image(file_path)
        print(f"图片路径：{file_path}")
        return Document(
            title=Path(file_path).name,  # 使用 .name 包含文件扩展名（如 .pdf、.docx）
            content=content,
            file_path=file_path,
            file_type=Path(file_path).suffix[1:].lower(),
            metadata={"ocr_available": OCR_AVAILABLE, "has_text": len(content) > 0}
        )

    def supports(self, file_type: str) -> bool:
        """判断是否支持图像类型（jpg/jpeg/png/bmp/gif）"""
        supported_types = ["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif"]
        return file_type.lower() in supported_types


class DocumentLoader:
    """
    文档加载器统一入口 - Document Loader Facade
    
    根据文件扩展名自动选择合适的加载器，提供统一的文档加载接口。
    
    设计决策：采用门面模式（Facade），对外提供简洁的接口，隐藏内部多Loader实现细节。
    """

    def __init__(self):
        """初始化加载器列表，按优先级顺序排列"""
        self.loaders = [
            PDFLoader(),
            DocxLoader(),
            MarkdownLoader(),
            ExcelLoader(),
            TextLoader(),
            ImageLoader()
        ]

    def _get_loader(self, file_path: str) -> BaseLoader:
        """
        根据文件扩展名获取对应的加载器（内部方法）
        
        参数:
            file_path: 文件路径
        
        返回:
            BaseLoader: 对应的文档加载器
        
        异常:
            ValueError: 当文件类型不支持时抛出
        """
        ext = Path(file_path).suffix[1:].lower()  # 获取扩展名（不含点号）
        for loader in self.loaders:
            if loader.supports(ext):
                return loader
        raise ValueError(f"Unsupported file type: {ext}")

    def load(self, file_path: str, structured: bool = False) -> Document:
        """
        加载单个文档
        
        参数:
            file_path: 文件路径
            structured: 是否启用结构化解析（docstruct）。
                仅对 PDF 生效，且需 settings.STRUCTURED_PARSING_ENABLED=True。
                解析失败自动降级（structure=None），不阻断入库。
        
        返回:
            Document: 解析后的文档对象
        """
        loader = self._get_loader(file_path)
        doc = loader.load(file_path)

        # ---- 结构化解析分支（PDF: docstruct / MD: markdown parser）----
        # 仅 显式开启 + 配置总开关开启 时才执行；
        # 其他类型文档或未开启时保持原行为，零影响。
        structure_parse_ms = 0.0
        file_type = getattr(doc, "file_type", "")
        if structured and file_type in ("pdf", "md", "markdown"):
            try:
                from backend.config.settings import settings  # 延迟导入，避免顶层循环依赖
                if file_type == "pdf" and settings.STRUCTURED_PARSING_ENABLED:
                    _t0 = time.time()
                    doc.structure = self._load_structure(file_path, doc)
                    structure_parse_ms = (time.time() - _t0) * 1000
                elif file_type in ("md", "markdown") and settings.MD_STRUCTURED_PARSING_ENABLED:
                    print(f"[DOCLOADER][md] 开始块级结构化解析: {os.path.basename(file_path)}")
                    _t0 = time.time()
                    doc.structure = self._load_md_structure(file_path, doc)
                    structure_parse_ms = (time.time() - _t0) * 1000
                    print(f"[DOCLOADER][md] 结构化解析完成: "
                          f"块={len(getattr(doc.structure, 'blocks', []) or [])} "
                          f"图片={getattr(doc.structure, 'stats', {}).get('images_found', 0)} "
                          f"耗时={structure_parse_ms:.0f}ms")
                elif file_type in ("md", "markdown"):
                    print("[DOCLOADER][md] 跳过块级结构化解析"
                          "（MD_STRUCTURED_PARSING_ENABLED=False）：md 走旧文本链路，"
                          f"文档内图片不做 OCR/视觉大模型: {os.path.basename(file_path)}")
            except Exception as exc:
                # 配置读取失败等任何异常都不影响主流程
                print(f"[DOCLOADER] 结构化解析配置读取失败: {exc}")

        # 解析统计日志：上传/更新即写 info.json 基本信息；
        # 结构化解析成功时补全结构类型数量/图片统计等。
        # 日志目录挂到 doc.parse_log_dir，切片阶段据此导出向量存储元数据。
        # 日志写入失败只打印警告，绝不影响主链路。
        try:
            if file_type == "pdf":
                from backend.rag.parse_logger import write_pdf_parse_log
                doc.parse_log_dir = write_pdf_parse_log(file_path, doc, parse_ms=structure_parse_ms) or ""
            elif file_type in ("md", "markdown"):
                # md 日志仅在结构化总开关开启时创建（关闭时保持旧行为：无日志）
                from backend.config.settings import settings
                if getattr(settings, "MD_STRUCTURED_PARSING_ENABLED", False):
                    from backend.rag.parse_logger import write_md_parse_log
                    doc.parse_log_dir = write_md_parse_log(file_path, doc, parse_ms=structure_parse_ms) or ""
        except Exception as exc:
            print(f"[DOCLOADER] 解析日志写入失败: {exc}")

        return doc

    def _load_structure(self, file_path: str, doc: Document) -> object:
        """
        调用 docstruct 库解析 PDF 为结构化中间产物（内部方法）。

        设计要点：
            1. 延迟导入 docstruct / settings，避免顶层循环依赖；
            2. 任何异常都返回 None（走旧纯文本链路），保证入库不中断；
            3. 解析结果挂在 doc.structure 上，由 TextSplitter 消费。

        参数:
            file_path: PDF 文件路径
            doc: 已加载的 Document（提供 doc_id）

        返回:
            StructuredDocument 或 None（失败/降级）
        """
        try:
            # 延迟导入：settings 与 docstruct 均在需要时才加载
            from backend.config.settings import settings
            from docstruct.parse import ParserGate
            from docstruct.config import ParserConfig

            # 流程图 LLM 识别通道：开启时实例化多模态客户端（DeepSeek 无视觉能力，
            # 走独立的 OpenAI 兼容视觉服务，见 backend/llm/vision_client.py）。
            # 客户端缺失/配置失败时置 None，docstruct 自动回退规则分析。
            flow_llm_client = None
            if settings.STRUCTURED_PARSER_FLOW_LLM_ENABLED:
                try:
                    from backend.llm.vision_client import VisionLLMClient
                    flow_llm_client = VisionLLMClient()
                    print("[DOCLOADER] 多模态视觉客户端就绪，流程图将优先走 LLM 识别")
                except Exception as exc:
                    print(f"[DOCLOADER] 多模态客户端初始化失败，流程图 LLM 通道降级为规则识别: {exc}")
                    flow_llm_client = None

            # 缺陷7：普通图片 LLM 解读通道（默认关）。
            # 开启且客户端可用时，图片块 OCR 无文本会调用 describe_image 生成描述。
            image_llm_client = None
            if getattr(settings, "STRUCTURED_PARSER_IMAGE_LLM_ENABLED", False):
                try:
                    from backend.llm.vision_client import VisionLLMClient
                    image_llm_client = VisionLLMClient()
                    print("[DOCLOADER] 图片 LLM 解读通道就绪")
                except Exception as exc:
                    print(f"[DOCLOADER] 图片 LLM 客户端初始化失败: {exc}")
                    image_llm_client = None

            # 用 settings 中的 STRUCTURED_PARSER_* 配置初始化解析门面
            gate = ParserGate(ParserConfig(
                chunk_size=settings.STRUCTURED_PARSER_CHUNK_SIZE,
                chunk_overlap=settings.STRUCTURED_PARSER_CHUNK_OVERLAP,
                max_table_tokens=settings.STRUCTURED_PARSER_MAX_TABLE_TOKENS,
                flow_llm_enabled=settings.STRUCTURED_PARSER_FLOW_LLM_ENABLED,
                flow_llm_client=flow_llm_client,
                flow_llm_auto_recheck=getattr(
                    settings, "STRUCTURED_PARSER_FLOW_LLM_AUTO_RECHECK", False),
                image_llm_enabled=getattr(
                    settings, "STRUCTURED_PARSER_IMAGE_LLM_ENABLED", False),
                image_llm_client=image_llm_client,
                output_dir=settings.STRUCTURED_PARSER_OUTPUT_DIR,
                flow_image_dir=settings.STRUCTURED_PARSER_FLOW_IMAGE_DIR,
                parser_version=settings.STRUCTURED_PARSER_PARSER_VERSION,
            ))
            parsed, reason = gate.parse_with_fallback(file_path, doc_id=doc.id or "")
            if parsed is None:
                print(f"[DOCLOADER] 结构化解析降级（继续使用旧链路）: {reason}")
            return parsed
        except Exception as exc:
            print(f"[DOCLOADER] 结构化解析初始化失败: {exc}（继续使用旧链路）")
            return None

    def _load_md_structure(self, file_path: str, doc: Document) -> object:
        """
        Markdown 块级结构化解析（内部方法）。

        设计要点：
            1. 延迟导入 markdown parser，任何异常返回 None（走旧纯文本链路）；
            2. 解析成功返回 MdStruct（splitter_key="md"），挂在 doc.structure 上，
               由 TextSplitter 识别分流到 MdStructuredSplitter；
            3. 图片图意化（OCR/VL）不在加载期执行——此时 doc.id 尚未分配，
               统一推迟到切片阶段（doc.id 已生成）由 md 切片分支调用，避免目录错位。

        参数:
            file_path: md 文件路径
            doc: 已加载的 Document（content 为原始 md 文本）

        返回:
            MdStruct 或 None（失败/降级）
        """
        try:
            from backend.rag.markdown_parser import parse_markdown
            print(f"[DOCLOADER][md] 解析器输入: 字符={len(doc.content or '')} 文件={file_path}")
            struct = parse_markdown(doc.content or "", file_path)
            if struct is None:
                print("[DOCLOADER][md] 块级解析返回空（继续使用旧链路）")
                return None
            struct.doc_id = doc.id or ""
            print(f"[DOCLOADER][md] 块级解析统计: {struct.stats}")
            # 开启开关但无任何有效块（空文件/纯注释）：仍给结构，交由切片器产出空列表
            return struct
        except Exception as exc:
            print(f"[DOCLOADER][md] 块级解析失败（继续使用旧链路）: {exc}")
            return None

    def load_batch(self, file_paths: List[str], structured: bool = False) -> List[Document]:
        """
        批量加载多个文档
        
        参数:
            file_paths: 文件路径列表
            structured: 是否启用结构化解析（透传给 load，仅 PDF + 总开关开启时生效）
        
        返回:
            List[Document]: 文档对象列表（失败的文件会被跳过并打印错误信息）
        
        设计决策：批量处理时捕获异常继续执行，提高容错性和用户体验。
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load(file_path, structured=structured)
                documents.append(doc)
            except Exception as e:
                print(f"Failed to load {file_path}: {str(e)}")
        return documents

    def get_supported_types(self) -> List[str]:
        """
        获取支持的文件类型列表
        
        返回:
            List[str]: 支持的文件扩展名列表
        """
        types = []
        for loader in self.loaders:
            if isinstance(loader, PDFLoader):
                types.append("pdf")
            elif isinstance(loader, DocxLoader):
                types.append("docx")
            elif isinstance(loader, MarkdownLoader):
                types.extend(["md", "markdown"])
            elif isinstance(loader, ExcelLoader):
                types.extend(["xlsx", "xls"])
            elif isinstance(loader, TextLoader):
                types.append("txt")
            elif isinstance(loader, ImageLoader):
                types.extend(["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif"])
        return types