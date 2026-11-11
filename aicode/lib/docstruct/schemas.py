# -*- coding: utf-8 -*-
"""
schemas.py - docstruct 库的全部数据结构定义
================================================

本文件是 docstruct 库的"地基"，所有其他模块（版面识别、内容提取、
结构组装、切片）都会用到这里定义的数据结构。

为什么用 dataclass（数据类）？
    Python 的 dataclass 会自动生成 __init__ / __repr__ / __eq__ 等方法，
    写起来简洁、读起来直观，和现有 backend/rag/document_loader.py 中
    Document / DocumentChunk 的写法保持一致，便于团队理解。

新手提示：
    - 每个类上方都有一行"中文说明"，说明这个类是干嘛的；
    - 每个字段都有注释说明它的含义；
    - 不同"阶段"的产物用不同的类承载，方便追踪数据流转：
      版面元素(LayoutElement) -> 内容块(Block) -> 结构化文档(StructuredDocument) -> 切片(Chunk)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

# =============================================================================
# 1. 类型别名（Python 3.8+ 支持 Literal，用于限定取值集合）
# =============================================================================
# 1. heading — 标题
# 含义：文档的章节标题，如「第一部分」「一、」「（一）」「1.」四级标题
# 识别器：TextRecognizer（与正文共用，但会附带标题层级）
# 数据结构：data.level 表示标题层级（1=第一部分, 2=一、, 3=（一）, 4=1.）
# 作用：作为结构树的锚点（StructureNode），划分章节层级关系
# 切片模板：HEADING_TEMPLATE → # 标题\n【章节路径】xxx
# 2. paragraph — 普通正文段落
# 含义：无标题性质的普通文字段落，即正文内容
# 识别器：TextRecognizer
# 特殊规则：text_recognizer.py 中有一处逻辑——若某行文本以 1. 等数字开头但未构成独立标题，会显式强制转为 paragraph，避免「目录清单误判成标题」
# 切片模板：PARAGRAPH_PREFIX → 注入【所属章节】保证上下文自包含
# 3. flowchart — 流程图
# 含义：矢量流程图（由矩形 + 连线 + 箭头组成的图形，非位图）
# 识别器：FlowchartRecognizer（矢量分析），另有 llm_flow_recognizer.py 做语义补全
# 数据结构：data 包含 title、nodes（节点及文本）、edges（箭头关系）、path_text（路径序列化，每条分支一行）、mermaid（渲染描述）
# 切片模板：FLOW_TEMPLATE → 【流程图】标题\n流程路径：A →[条件] B
# 备注：检测时会把误覆盖整页的 bbox 收敛到实际图形区域
# 4. checkbox — 复选框区域
# 含义：带勾选框的条目区域，如「☑ 已建立预防接种证 / □ 需补种」
# 识别器：CheckboxRecognizer，依据 CHECKBOX_CHARS（☑/□ 等符号）定位
# 数据结构：data.items → [{"text": 条目文本, "checked": 是否勾选}]
# 切片模板：CHECKBOX_TEMPLATE → 【复选框】章节路径\n[x] 条目
# 版面元素 / 内容块的类型。整个库里所有"这是哪类内容"的判断都用这个集合。
BlockType = Literal[
    "heading",       # 标题（如"第一部分""一、""（一）""1." 四级标题）
    "paragraph",     # 普通正文段落
    "table",         # 表格（线框表）
    "flowchart",     # 流程图（矢量图，由矩形+连线+箭头组成）
    "image",         # 图片（位图）
    "checkbox",      # 复选框区域（☑/□ 符号）
    "page_header",   # 页眉（每页顶部重复的文本）
    "page_footer",   # 页脚（每页底部重复的文本，如页码）
]


# =============================================================================
# 2. 坐标框
# =============================================================================

@dataclass
class BBox:
    """
    矩形坐标框（bounding box）。

    PDF 坐标系原点在页面**左上角**，x 向右增大，y 向下增大，单位是点（pt）。
    一个完整的 PDF 页面通常是 595x842 点（A4）。

    参数:
        x0: 左边缘坐标
        y0: 上边缘坐标
        x1: 右边缘坐标
        y1: 下边缘坐标
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """宽度（右-左），用于判断元素尺寸。"""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """高度（下-上），用于判断元素尺寸。"""
        return self.y1 - self.y0

    def contains(self, other: "BBox") -> bool:
        """
        判断本框是否完全包含另一个框（用于节点识别、重叠仲裁）。

        参数:
            other: 另一个坐标框

        返回:
            True 表示 other 完全在本框内部（含边缘刚好相等）
        """
        return (self.x0 <= other.x0 and self.y0 <= other.y0
                and self.x1 >= other.x1 and self.y1 >= other.y1)

    def overlaps(self, other: "BBox", ratio: float = 0.5) -> bool:
        """
        判断本框与另一个框是否重叠（重叠率 >= ratio 才算重叠）。

        参数:
            other: 另一个坐标框
            ratio: 重叠判定阈值（默认 0.5，即相交面积占本框面积的一半以上）

        返回:
            True 表示两框重叠程度达到阈值
        """
        # 计算两框相交矩形的宽高（若无相交则为 0 或负数）
        inter_w = min(self.x1, other.x1) - max(self.x0, other.x0)
        inter_h = min(self.y1, other.y1) - max(self.y0, other.y0)
        if inter_w <= 0 or inter_h <= 0:
            return False
        inter_area = inter_w * inter_h
        # 用本框面积为分母计算重叠率
        return inter_area / max(self.width * self.height, 1e-9) >= ratio


# =============================================================================
# 3. 版面元素（第一层识别产物）
# =============================================================================

@dataclass
class LayoutElement:
    """
    版面元素：第一层识别的结果，回答"文档里哪里有哪类内容"。

    这一层只负责"定位"（类型 + 位置 + 置信度），不负责"提取内容"。
    真正的文本/表格内容由第二层的 Recognizer 负责提取。

    为什么分两层？
        版面识别（哪里有什么）与内容提取（具体是什么）关注点不同，
        分层后可以独立调试、独立测试、单层失败不影响另一层。

    参数:
        type: 元素类型，见 BlockType
        bbox: 元素所在区域
        page_no: 所在页码（从 1 开始）
        confidence: 置信度 0~1，默认 1.0（规则命中即满置信）
        source: 来源，rule=规则识别 / llm=多模态大模型 / ocr=OCR
        detail: 判定依据的附加信息（如字号、矢量统计），便于调试
    """
    type: BlockType
    bbox: BBox
    page_no: int
    confidence: float = 1.0
    source: str = "rule"
    detail: Dict = field(default_factory=dict)


# =============================================================================
# 4. 流程图边（箭头关系）
# =============================================================================

@dataclass
class FlowEdge:
    """
    流程图的一条边（箭头），记录"从哪个节点到哪个节点、条件是什么"。

    例如"血压高于正常值 -> 转诊上级医院"：
        from_node = "血压高于正常值"
        to_node   = "转诊上级医院"
        label     = ""（无条件标签时为空字符串）

    为什么把边单独建模？
        这是流程图检索的核心价值所在——问题"什么情况下转诊？"的答案
        恰恰存在于边的 label / 节点关系里，必须显式保留而不是混在文本里。
    """
    from_node: str
    to_node: str
    label: str = ""


# =============================================================================
# 5. 内容块（第二层提取产物）
# =============================================================================

@dataclass
class Block:
    """
    内容块：第二层提取的结果，扁平列表，按文档阅读顺序排列。

    每个 Block 代表"一个完整的内容单元"（一个段落 / 一张表 / 一个流程图……），
    并带有它在文档层级结构中的位置（parent_id、section_path）。

    参数:
        block_id: 块唯一标识，格式：{doc_id}__p{page_no}__{序号}
        type: 块类型（与 LayoutElement.type 对应）
        page_no: 所在页码
        bbox: 所在区域
        content: 该块的主文本内容（对表格是渲染后的 Markdown 表格文本）
        data: 类型专属的结构化数据（见下方约定）
        parent_id: 所属标题块的 block_id（None 表示直接挂根）
        section_path: 章节路径，如 "第一部分 居民健康档案管理/五、工作指标"
        confidence: 置信度
        source: 来源（rule/llm/ocr）
        image_path: 图片/流程图渲染成图片后的文件路径（用于 LLM 通道）
    """
    block_id: str
    type: BlockType
    page_no: int
    bbox: BBox
    content: str
    data: Dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    section_path: str = ""
    confidence: float = 1.0
    source: str = "rule"
    image_path: Optional[str] = None

    # ------------------------------------------------------------------
    # data 字段按类型约定的格式（写入方遵守、读取方依赖）：
    #
    # table（表格）:
    #   {
    #     "caption": 表格标题文本（如上方的"表 2-3 高血压患者随访表"），
    #     "header": [列名列表],
    #     "rows": [每一行的单元格列表],
    #     "col_count": 列数,
    #     "row_count": 行数,
    #     "page_end": 跨页表格的结束页码（不跨页则等于 page_no）,
    #     "leading_text": 表格前紧邻的正文段落（上下文说明）,
    #   }
    #
    # flowchart（流程图）:
    #   {
    #     "title": 流程图标题,
    #     "nodes": [{"id": 节点id, "text": 节点文本}],
    #     "edges": [FlowEdge, ...],      # 箭头关系列表
    #     "path_text": 路径序列化文本（每条分支一行）,
    #     "mermaid": Mermaid 图描述（便于人工查看）,
    #   }
    #
    # checkbox（复选框）:
    #   {
    #     "items": [{"text": 条目文本, "checked": 是否勾选}],
    #   }
    #
    # image（图片）:
    #   {
    #     "ocr_text": OCR 识别出的文本,
    #     "img_ext": 图片扩展名,
    #   }
    #
    # heading（标题）:
    #   {
    #     "level": 标题层级（1=第一部分, 2=一、, 3=（一）, 4=1.）,
    #   }
    # ------------------------------------------------------------------


# =============================================================================
# 6. 结构树节点
# =============================================================================

@dataclass
class StructureNode:
    """
    标题锚点树节点（标题层级结构）。

    以《国家基本公共卫生服务规范》为例：
        第一部分 居民健康档案管理（level=1）
        ├── 一、服务对象（level=2）
        │   └── （一）XX（level=3）
        │       └── 1. XX（level=4）
        └── 二、服务内容（level=2）

    参数:
        node_id: 节点唯一标识（如 "n0001"）
        title: 标题文本
        level: 标题层级（1~4）
        heading: 对应的标题 Block（保留坐标等原始信息）
        children: 子节点列表（按文档顺序）
        blocks: 该标题下直属的非标题内容块（按阅读顺序）
    """
    node_id: str
    title: str
    level: int
    heading: Block
    children: List["StructureNode"] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)


# =============================================================================
# 7. 解析问题与报告
# =============================================================================

@dataclass
class ParseIssue:
    """
    解析过程中记录的单个问题（用于质量报告）。

    参数:
        level: 级别，error=错误（可能导致内容缺失）/ warning=警告（内容不完整）/ info=提示
        code: 问题码，见 report.py 中的常量表（如 "W_TABLE_TRUNCATED"）
        message: 人类可读的问题描述
        page_no: 问题所在页码（可为 None）
        block_id: 问题所在块 id（可为 None）
        detail: 附加细节（如截断行数）
    """
    level: str
    code: str
    message: str
    page_no: Optional[int] = None
    block_id: Optional[str] = None
    detail: Dict = field(default_factory=dict)


@dataclass
class ParseReport:
    """
    解析质量报告：记录整个解析过程的所有问题 + 各类内容的统计。

    调用方（如入库流程）可以根据 errors/warnings 判断是否需要人工复核。

    参数:
        issues: 全部问题列表
        stats: 统计字典，如 {"table": 85, "flowchart": 12, "heading": 210, ...}
    """
    issues: List[ParseIssue] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def errors(self) -> List[ParseIssue]:
        """只返回 error 级别的问题列表。"""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ParseIssue]:
        """只返回 warning 级别的问题列表。"""
        return [i for i in self.issues if i.level == "warning"]

    def add(self, issue: ParseIssue) -> None:
        """追加一个问题并同时更新统计。"""
        self.issues.append(issue)
        key = f"issues_{issue.level}"
        self.stats[key] = self.stats.get(key, 0) + 1


# =============================================================================
# 8. 结构化文档（最终中间产物）
# =============================================================================

@dataclass
class StructuredDocument:
    """
    结构化中间产物：一次解析的最终结果。

    它同时提供三种视图，方便不同场景使用：
        1. tree:   树状结构（标题层级 + 内容块归属），用于逻辑理解
        2. json:   全量 JSON 字典，便于落盘、传输、对比
        3. markdown: 渲染后的 Markdown，便于人工阅读、调试

    在 RAG 集成中，这个对象会被挂在 backend 的 Document.structure 字段上，
    由 StructuredSplitter 消费生成切片。

    参数:
        doc_id: 文档标识（与 backend Document.id 对齐）
        file_path: 源 PDF 文件路径
        title: 文档标题
        parser_version: 解析器版本号（升级后据此触发重新解析）
        toc: 目录列表 [{level, title, node_id, page_no}]
        blocks: 全部内容块（扁平，按阅读顺序）
        tree: 标题锚点树的根节点
        markdown: Markdown 渲染结果
        json: JSON 字典渲染结果
        report: 解析质量报告
    """
    doc_id: str
    file_path: str
    title: str
    parser_version: str
    toc: List[Dict] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    tree: Optional["StructureNode"] = None
    markdown: str = ""
    json: Dict = field(default_factory=dict)
    report: ParseReport = field(default_factory=ParseReport)


# =============================================================================
# 9. 切片产物（结构感知切片的输出）
# =============================================================================

@dataclass
class Chunk:
    """
    结构感知切片产物（docstruct 内部定义，与 backend 解耦）。

    之所以在 docstruct 内部单独定义 Chunk，而不是直接依赖 backend 的 TextChunk：
        - 本库可以脱离 backend 独立运行、独立测试；
        - backend 侧通过适配器（见 text_splitter.py 的 _to_text_chunks）转换。

    字段含义与 backend 的 TextChunk 完全对应，方便一一转换：
        content: 切片文本（用于向量化与检索展示）
        chunk_index: 在整篇文档中的块序号（从 0 开始）
        total_chunks: 总块数（转换时统一回填）
        start_pos: 文本起始位置（相对全文内容，用于溯源）
        end_pos: 文本结束位置
        metadata: 附加元数据（会并入向量库，结构字段如 section_path 在此）
    """
    content: str
    chunk_index: int
    total_chunks: int
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict = field(default_factory=dict)
