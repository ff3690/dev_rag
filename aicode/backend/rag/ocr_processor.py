"""
OCR处理器模块 - OCR Processor Module (基于 Tesseract)

使用 Tesseract OCR 引擎从图像中提取文本，支持中英文及多种语言。
设计上保持与原有 PaddleOCR 版本接口兼容。
"""

import pytesseract
from PIL import Image, ImageOps
from typing import List, Optional
from dataclasses import dataclass
import io
import logging
import os
import sys

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """
    OCR识别结果数据类
    """
    text: str
    confidence: float
    bbox: Optional[List[int]] = None


class OCRProcessor:
    """
    OCR处理器类 - 基于 Tesseract
    """

    def __init__(self, lang: str = "chi_sim"):
        """
        初始化OCR处理器
        
        Args:
            lang: Tesseract语言代码，默认简体中文（chi_sim）
                   可选 "eng"（英文），"chi_tra"（繁体中文）等
        """
        self.lang = lang
        self._set_tesseract_path()  # 自动检测并配置 Tesseract 可执行文件路径

    def _set_tesseract_path(self):
        """
        自动检测 Tesseract 安装路径（Windows 下常见）
        """
        # 如果系统 PATH 中已有 tesseract，则无需设置
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'D:\Program Files\Tesseract-OCR\tesseract.exe',
                r'D:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Tesseract found at {path}")
                    break
            else:
                logger.warning("Tesseract not found in default Windows paths. "
                               "Please ensure it's installed and added to PATH.")
        # Linux/macOS 通常可通过 PATH 自动发现

    def process_image(self, image_path: str) -> List[OCRResult]:
        """
        处理单张图像，提取文本及位置信息
        
        Returns:
            List[OCRResult]: 每个识别到的文本块
        """
        try:
            img = Image.open(image_path)
            # 使用 Tesseract 获取详细数据（含置信度和边界框）
            data = pytesseract.image_to_data(
                img, lang=self.lang, output_type=pytesseract.Output.DICT
            )
            results = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                # 过滤空白或置信度为0的项（-1表示无识别）
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        conf = float(data['conf'][i]) / 100.0  # 归一化到0-1
                        x, y, w, h = (data['left'][i], data['top'][i],
                                      data['width'][i], data['height'][i])
                        bbox = [x, y, x + w, y + h]
                        results.append(OCRResult(text=text, confidence=conf, bbox=bbox))
            return results
        except Exception as e:
            logger.error(f"Tesseract OCR processing failed for {image_path}: {str(e)}")
            return []

    def batch_process(self, image_paths: List[str]) -> List[List[OCRResult]]:
        """批量处理多张图像"""
        results = []
        for image_path in image_paths:
            results.append(self.process_image(image_path))
        return results

    def extract_text1(self, image_path: str) -> str:
        """从图像中提取纯文本（无置信度信息）"""
        """image_to_data 返回的是每个检测到的文本块（通常是单词或单个字符），而这个方法将这些文本块用换行符 \n 拼接起来"""
        results = self.process_image(image_path)
        return "\n".join([r.text for r in results])
		
    def extract_text(self, image_path: str) -> str:
        """从图像中提取整段文本（保留自然换行，避免每字符一行）"""
        """对于提取纯文本（不需要坐标和置信度）的场景，直接使用 pytesseract.image_to_string 更合适，它会返回按段落/行分组的自然文本，而不是每个字符独立。"""
        try:
            img = Image.open(image_path)
            # 直接使用 image_to_string 获得自然段落文本
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract text extraction failed for {image_path}: {str(e)}")
            return ""

    def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> str:
        """提取文本并过滤低置信度结果"""
        results = self.process_image(image_path)
        filtered = [r.text for r in results if r.confidence >= min_confidence]
        return "\n".join(filtered)

    @staticmethod
    def _preprocess_image(img: "Image.Image") -> "Image.Image":
        """OCR 前置处理：灰度化 + 对比度增强；短边过小的图放大 2 倍。

        经验上可显著提升小字号/扫描件的 Tesseract 识别率。
        """
        try:
            if img.mode != "L":
                img = ImageOps.grayscale(img)
            # 对比度拉伸（去除灰蒙蒙的扫描底色）
            img = ImageOps.autocontrast(img)
            # 小图放大 2 倍（双三次插值，降低文字粘连）
            w, h = img.size
            if min(w, h) < 600:
                scale = 2.0
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            return img
        except Exception:
            # 预处理失败时退回原图，不阻塞识别
            return img

    def extract_text_from_pil(self, img: "Image.Image") -> str:
        """从 PIL.Image 中提取整段文本（内存模式，兼容 ImageRecognizer）。

        Args:
            img: 已打开的 PIL Image 对象

        Returns:
            str: 识别出的文本（空串表示失败）
        """
        try:
            img = self._preprocess_image(img)
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract PIL OCR failed: {str(e)}")
            return ""


def extract_text_from_image_bytes(data: bytes, lang: str = "chi_sim") -> str:
    """从图片字节流中提取文本（内存模式，避免落盘临时文件）。

    Args:
        data: 原始图片字节（PNG/JPEG 等）
        lang: Tesseract 语言代码

    Returns:
        str: 识别出的文本（空串表示失败）
    """
    try:
        img = Image.open(io.BytesIO(data))
        # 统一转 RGB，避免 CMYK/P 模式的兼容问题
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return OCRProcessor(lang).extract_text_from_pil(img)
    except Exception as e:
        logger.error(f"Tesseract bytes OCR failed: {str(e)}")
        return ""