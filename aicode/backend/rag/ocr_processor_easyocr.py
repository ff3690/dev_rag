"""
OCR处理器模块 - 基于 EasyOCR
"""

import easyocr
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Optional[List[int]] = None


class OCRProcessor:
    def __init__(self, lang: str = "ch_sim"):
        """
        初始化 EasyOCR
        Args:
            lang: 语言代码，'ch_sim' 简体中文, 'en' 英文等
        """
        self.lang = lang
        # EasyOCR 支持多语言列表，这里使用包含中英文的列表
        self.reader = easyocr.Reader([lang, 'en'], gpu=False)  # 若需 GPU 可改为 True

    def process_image(self, image_path: str) -> List[OCRResult]:
        try:
            results = self.reader.readtext(image_path)
            ocr_results = []
            for bbox, text, confidence in results:
                # bbox 是四个点坐标，转为 [x1,y1,x2,y2]
                x1 = int(min(p[0] for p in bbox))
                y1 = int(min(p[1] for p in bbox))
                x2 = int(max(p[0] for p in bbox))
                y2 = int(max(p[1] for p in bbox))
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2]
                ))
            return ocr_results
        except Exception as e:
            logger.error(f"EasyOCR processing failed for {image_path}: {str(e)}")
            return []

    def batch_process(self, image_paths: List[str]) -> List[List[OCRResult]]:
        return [self.process_image(p) for p in image_paths]

    def extract_text(self, image_path: str) -> str:
        results = self.process_image(image_path)
        return "\n".join([r.text for r in results])

    def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> str:
        results = self.process_image(image_path)
        filtered = [r.text for r in results if r.confidence >= min_confidence]
        return "\n".join(filtered)