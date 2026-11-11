"""
应用配置模块 - Application Settings Module

使用 Pydantic BaseSettings 实现配置管理，支持环境变量和 .env 文件。
配置项分为以下几类：
1. 基础配置：应用名称、环境、数据库、密钥
2. LLM 配置：大语言模型路径、名称、参数
3. RAG 配置：向量存储、Embedding 模型（核心配置）
4. 文件配置：上传限制、支持的文件类型
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用设置类 - Application Settings Class

    所有配置项都有默认值，可通过 .env 文件或环境变量覆盖。
    """

    # ==================== 基础配置 ====================
    APP_NAME: str = "dev-team-rag-agent"      # 应用名称
    ENV: str = "development"                   # 运行环境：development / production
    DATABASE_URL: str = "sqlite:///./backend/db/app.db"  # SQLite 数据库路径
    SECRET_KEY: str = "dev-team-rag-agent-secret-key-2026-change-in-production"  # JWT 密钥
    ALGORITHM: str = "HS256"                  # JWT 加密算法
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30      # Token 过期时间（分钟）

    # ==================== LLM 配置（本地模型 - 已弃用，保留兼容） ====================
    MODEL_PATH: str = "models/qwen2-7b-chat-q4_0.gguf"  # 模型文件路径（GGUF 格式）
    MODEL_NAME: str = "qwen2-7b-chat"          # 模型标识名称
    MAX_TOKENS: int = 2048                     # 最大生成 token 数
    TEMPERATURE: float = 0.7                   # 生成温度，越高越有创造性

    # ==================== LLM API 配置（可切换） ====================
    # 选择一个服务商，注释掉另一个即可切换
    
    # 方案一：七牛云 AI API（推荐）
    # 申请地址: https://developer.qiniu.com/aitokenapi
    # DEEPSEEK_API_KEY: str = "sk-f03f2864889f746c57435c3562bf557bbcdfa21935975aae2acc2b3c1ce24b3d"  # TODO: 填入七牛云 API Key
    # DEEPSEEK_API_BASE: str = "https://api.qnaigc.com/v1"
    # DEEPSEEK_MODEL: str = "deepseek-v3"        # 可选: deepseek-v3、deepseek-r1、deepseek-r1-32b
    
    # 方案二：DeepSeek 原生 API
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"      # 可选: deepseek-chat、deepseek-reasoner
    
    # 通用参数（两个服务商共用）
    DEEPSEEK_TEMPERATURE: float = 0.7          # 生成温度 (0-1)，越低越确定性
    DEEPSEEK_MAX_TOKENS: int = 2048            # 最大生成 token 数
    DEEPSEEK_TIMEOUT: int = 60                 # API 请求超时（秒）
    
    # Token 用量统计开关（默认开启）
    # 说明：流式请求中需要在请求体传入 stream_options={"include_usage": true}，
    #       DeepSeek/OpenAI 兼容模型才会在最后一个 chunk 返回 usage 字段。
    #       - 若换用不支持该参数的其他模型/网关，开启可能导致 400 报错，
    #         此时请将此开关改为 False（关闭后 token 用量将无法统计）。
    DEEPSEEK_STREAM_INCLUDE_USAGE: bool = True

    # ==================== 知识图谱专用 LLM 配置（独立于检索生成） ====================
    # 知识图谱构建（实体/关系抽取）与检索生成（RAG 问答）可分别使用不同的大模型。
    # 各字段留空时自动 fallback 到上方的 DEEPSEEK_* 配置，保证开箱即用、不改动即共用一个模型。
    #
    # 典型使用场景：
    # - 省钱：图谱批量抽取走本地模型（如 Ollama qwen2.5:7b），问答走 deepseek-chat
    # - 质量优先：图谱抽取用 deepseek-chat（低温度），问答用 deepseek-reasoner
    # - 离线建图：图谱走本地模型，联网问答走 DeepSeek API
    #
    # 若 GRAPH_LLM_API_BASE 指向 Ollama（如 http://localhost:11434/v1），
    # 模型名需填写 Ollama 中实际存在的名称（如 qwen2.5:7b）。
    # 示例：
    # GRAPH_LLM_MODEL=qwen2.5:7b
    # GRAPH_LLM_API_BASE=http://localhost:11434/v1
    # GRAPH_LLM_API_KEY=ollama
    # GRAPH_LLM_TEMPERATURE=0.2
    GRAPH_ENABLED: bool = False                   # 是否在上传文档后自动构建知识图谱 True 构建 / False 不构建
    GRAPH_LLM_MODEL: str = ""                     # 图谱模型，空则使用 DEEPSEEK_MODEL
    GRAPH_LLM_API_KEY: str = ""                   # 图谱 API Key，空则使用 DEEPSEEK_API_KEY
    GRAPH_LLM_API_BASE: str = ""                  # 图谱 API 地址，空则使用 DEEPSEEK_API_BASE
    GRAPH_LLM_TEMPERATURE: float = 0.3           # 抽取任务需要更低温度，保证 JSON 输出稳定
    GRAPH_LLM_MAX_TOKENS: int = 2048             # 图谱最大生成 token 数，空/0 则使用 DEEPSEEK_MAX_TOKENS
    GRAPH_LLM_TIMEOUT: int = 60                  # 图谱 API 请求超时（秒），空/0 则使用 DEEPSEEK_TIMEOUT

    # ==================== RAG 配置（核心） ====================
    # 向量存储路径 - LlamaIndex 索引持久化目录
    VECTOR_STORE_PATH: str = "vector_data/"
    
    # 集合名称 - 用于日志标识（LlamaIndex 使用目录存储，此项保留用于兼容）
    COLLECTION_NAME: str = "rag_documents"
    
    # Embedding 向量维度
    # 选择 512 维的原因：
    # 1. 匹配 BAAI/bge-small-zh-v1.5 模型的输出维度（该模型固定输出 512 维向量）
    # 2. bge-small 系列是轻量级模型，适合 CPU 环境运行，内存占用小
    # 3. 512 维在语义表达能力和计算效率之间取得良好平衡
    # 4. 维度越高表示能力越强，但存储和计算开销也越大
    # 5. 如果使用 bge-base 或 bge-large 系列，需要调整为 768 或 1024 维
    EMBEDDING_DIMENSION: int = 512
    
    # Embedding 模型名称
    # BAAI/bge-small-zh-v1.5 是专门针对中文优化的轻量级 Embedding 模型
    # 选择理由：
    # 1. 中文效果优秀：在 C-MTEB 中文评测榜排名前列
    # 2. 轻量高效：仅 30MB 参数，CPU 上也能快速推理
    # 3. 多语言支持：同时支持中英文混合场景
    # 4. Apache 2.0 许可：商业友好
    # 替代方案：BAAI/bge-base-zh-v1.5（768维，效果更好但更慢）
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    
    # Embedding 模型本地路径
    # 如果设置此路径，将直接从本地加载模型，不再从 HuggingFace 下载
    # 适用场景：已手动下载模型到本地、离线环境
    # 示例："models/embedding" 或 "E:/models/bge-small-zh-v1.5"
    # 留空则使用 EMBEDDING_MODEL_NAME 从 HuggingFace 自动下载/加载
    EMBEDDING_MODEL_PATH: str = "models/embedding"
    
    # 推理设备
    # 当前选择 cpu 的原因：
    # 1. 用户当前环境无 GPU，必须使用 CPU 推理
    # 2. bge-small 模型在 CPU 上表现尚可，单次向量化约 10-30ms
    # 3. 如需 GPU 加速，可改为 "cuda" 或 "cuda:0"
    # 4. MPS (Mac) 可使用 "mps"
    EMBEDDING_DEVICE: str = "cpu"

    # ==================== Reranker 精排配置 ====================
    # BGE-Reranker-v2-m3 模型配置（Cross-Encoder 精排）
    # 用于对向量粗排结果进行二次重排序，提升检索精度
    #
    # 技术说明：
    # - 模型架构：Cross-Encoder (XLM-RoBERTa-based)
    # - 参数量：~560M，内存占用约 1.1GB
    # - 输入格式：Query-Document 对拼接
    # - 输出格式：相关性分数（经 sigmoid 归一化为 0-1）
    #
    # CPU 优化要点（基于实测经验）：
    # 1. INPUT_TOP_K 设为 5（而非 20），避免 13 秒级延迟
    # 2. MAX_LENGTH 设为 256（而非 512），降低计算复杂度
    # 3. 必须关闭 FP16，使用 FP32（CPU 上 FP16 无加速且可能报错）
    # 4. BATCH_SIZE 设为 4，避免内存抖动
    # 5. 文档预截断至 800 字符，减少序列长度

    # 是否启用 Reranker 精排（总开关）
    # True: 启用精排，检索流程为 粗排 → 精排 → 返回
    # False: 关闭精排，仅使用向量粗排结果
    RERANKER_ENABLED: bool = True

    # Reranker 模型名称（HuggingFace 标识符）
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

    # Reranker 模型本地路径
    # 优先从本地加载，避免重复下载
    # 下载后会自动缓存到此目录
    RERANKER_MODEL_PATH: str = "models/reranker"

    # Reranker 推理设备
    # CPU 环境必须设为 "cpu"，GPU 环境可设为 "cuda"
    RERANKER_DEVICE: str = "cpu"

    # 粗排候选数量（进入 Reranker 的文档数量）
    # CPU 环境建议 5-10，GPU 环境可设为 20-50
    # 值越大召回率越高，但推理时间线性增长
    # 值由5扩大到10，优点是精排时有足够空间筛选，真正过滤噪声，缺点是CPU 耗时可能翻倍
    RERANKER_INPUT_TOP_K: int = 10

    # 精排输出数量（最终返回给 LLM 的文档数量）
    # 通常与 API 请求的 top_k 保持一致
    RERANKER_OUTPUT_TOP_K: int = 5

    # Reranker 最大序列长度（token 数）
    # CPU 环境建议 256（平衡精度与速度）
    # GPU 环境可设为 512（精度更高）
    # 注意：512 是默认值但会导致 CPU 推理极慢（13秒+）
    RERANKER_MAX_LENGTH: int = 256

    # Reranker 批处理大小
    # CPU 建议 4-8，GPU 建议 16-32
    # 过大会导致内存不足，过小会降低吞吐量
    RERANKER_BATCH_SIZE: int = 4

    # Reranker 置信度阈值
    # 低于此分数的文档将被过滤（认为不相关）
    # 分数范围：0-1（经 sigmoid 归一化）
    # - > 0.7：高度相关
    # - 0.5-0.7：中度相关
    # - 0.3-0.5：弱相关
    # - < 0.3：不相关
    RERANKER_SCORE_THRESHOLD: float = 0.3

    # 文档预截断字符数
    # 在送入 Reranker 前截断文档长度，减少序列计算量
    # 配合 MAX_LENGTH 使用，进一步优化 CPU 推理速度
    RERANKER_DOC_CHAR_LIMIT: int = 800

    # Reranker 超时时间（秒）
    # 超时后降级到粗排结果，避免请求阻塞
    RERANKER_TIMEOUT: int = 10

    # ==================== 文件配置 ====================
    MAX_FILE_SIZE: str = "50MB"                # 单文件上传大小限制
    SUPPORTED_EXTENSIONS: str = ".pdf,.docx,.md,.xlsx,.png,.jpg,.jpeg"  # 支持的文件格式

    # ==================== 文档结构化解析配置（docstruct） ====================
    # 针对《国家基本公共卫生服务规范》这类含表格/流程图/复选框的 PDF，
    # 使用自研 docstruct 库解析为结构化中间产物，再做结构感知切片。
    #
    # 集成原则：
    # - 总开关关闭（默认）时，PDF 走原有纯文本链路，行为与改动前完全一致；
    # - 开启后，仅 PDF 类型文档启用结构化解析，其他类型（md/docx/txt 等）不受影响；
    # - 解析失败自动降级回旧链路（structure=None），不阻断入库。

    # 结构化解析总开关（默认关 = 完全保持旧行为；开启 = PDF 走新链路）
    STRUCTURED_PARSING_ENABLED: bool = True

    # 常规文本切片 token 上限（与现有 RAG 的 chunk_size 保持一致）
    STRUCTURED_PARSER_CHUNK_SIZE: int = 512

    # 相邻切片重叠 token 数
    STRUCTURED_PARSER_CHUNK_OVERLAP: int = 64

    # 表格切片 token 上限（表格优先整表完整切片，超过才降级截断）
    # 说明：bge 模型输入上限 512 token，但表格"整表完整"才能回答跨行列问题，
    #       因此表格单独放行到 2000 token，向量化时由 embedding 服务自动截断。
    STRUCTURED_PARSER_MAX_TABLE_TOKENS: int = 2000

    # 表格切片的"embedding 安全 token 阈值"（默认 400 < 512）。
    # 整表 token 超过该值时，除整表块外额外生成"行组伴随块"（表头 + 若干行），
    # 保证大表后半部分的行在向量空间可检索，弥补 embedding 服务截断导致的盲区。
    STRUCTURED_PARSER_EMBED_SAFE_TOKENS: int = 400

    # 流程图 LLM 识别通道开关（默认关 = 仅用矢量规则分析，开=流程图优先走多模态 LLM）
    # 本 PDF 的流程图是矢量图（矩形+连线），规则分析已足够；
    # 该开关用于"图片型流程图"文档的扩展场景。
    STRUCTURED_PARSER_FLOW_LLM_ENABLED: bool = False

    # 流程图 LLM 识别通道 - 多模态模型 API 配置
    # DeepSeek 官方 API 无视觉能力，无法"看图"，因此这里独立配置一个
    # 多模态模型（OpenAI 兼容协议），与上方 DEEPSEEK_* 完全解耦。
    # 默认指向智谱 GLM-4V-Flash（免费额度）：
    #   Key 申请: https://open.bigmodel.cn/ （实名注册后创建 API Key）
    # 也可切换为其他 OpenAI 兼容视觉服务，只需改 base_url/model：
    #   阿里云百炼 Qwen-VL: base_url=https://dashscope.aliyuncs.com/compatible-mode/v1, model=qwen-vl-plus
    #   OpenAI GPT-4o:      base_url=https://api.openai.com/v1, model=gpt-4o
    # 注意：API Key 未配置时 LLM 通道自动降级为规则识别，不影响主流程。
    STRUCTURED_PARSER_FLOW_LLM_API_KEY: str = ""                    # 多模态模型 API Key
    STRUCTURED_PARSER_FLOW_LLM_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"  # OpenAI 兼容地址
    STRUCTURED_PARSER_FLOW_LLM_MODEL: str = "glm-4v-flash"          # 视觉模型名
    STRUCTURED_PARSER_FLOW_LLM_TIMEOUT: int = 120                   # 请求超时（秒）

    # 流程图规则通道低置信度自动 LLM 复核（默认关 = 只打 warning 不复核）
    # 开启时：规则通道识别出节点<3 或 边<2（confidence 降到 0.5）时，
    # 自动调用 flow_llm_client 对流程图区域图片做二次识别并替换结果。
    STRUCTURED_PARSER_FLOW_LLM_AUTO_RECHECK: bool = False

    # 普通图片 LLM 解读通道（默认关 = 图片块无 OCR 文本时仅【图片】占位）
    # 开启时：图片块 OCR 无文本会调用多模态客户端 describe_image 生成语义描述，
    # 避免图片在向量空间完全不可检索（缺陷7）。
    STRUCTURED_PARSER_IMAGE_LLM_ENABLED: bool = False

    # BM25 关键词混合检索开关（默认关 = 仅向量检索）
    # 开启后在入库时同步构建 SQLite FTS5 关键词索引，
    # 检索时与向量结果做 RRF 融合，提升"建档率/随访频率"这类单值查询的命中率。
    STRUCTURED_PARSER_BM25_ENABLED: bool = False

    # 结构化中间产物（md/json/流程图图片）落盘目录
    STRUCTURED_PARSER_OUTPUT_DIR: str = "artifacts/structured"

    # 流程图渲染图片目录（供 LLM 通道与人工抽检使用）
    STRUCTURED_PARSER_FLOW_IMAGE_DIR: str = "artifacts/structured/flow_images"

    # 解析器版本号：升级后入库侧据此提示用户重新解析入库（不自动重建）
    STRUCTURED_PARSER_PARSER_VERSION: str = "1.0.0"

    # 每个 PDF 的解析统计日志根目录。
    # 上传/更新 PDF 时会在此目录下新建 <PDF名>_<时间戳>/ 子目录，
    # 记录 info.json（基本信息+质量报告）、结构片段（表格md/流程图png/图片ocr txt）
    # 与 metadata.json（向量存储元数据）。每次上传新建，历史可追溯。
    PARSE_LOG_ROOT: str = "logs/解析日志"

    # ==================== Markdown 结构化解析配置 ====================
    # 针对含表格/代码块/图片的复杂 md 文档（如 doc_to_md 产物），
    # 入库前先做"块级结构化解析"，再做结构感知切片：
    #   - 表格    -> 整表完整切片（可回答跨列组合问题）
    #   - 代码块  -> 整块不切（含 mermaid）
    #   - 图片    -> auto 档语义化（OCR -> 必要时视觉大模型描述），落盘 + 静态托管
    #
    # 集成原则（与 PDF docstruct 完全一致）：
    # - 总开关关闭时，md 走原有文本链路，上传/切片行为与改动前逐字节一致
    #   （注意：关闭时 md 内图片既不落盘也不做 OCR/视觉大模型图意化）；
    # - 开启后，md 块级结构解析失败自动降级回旧链路（structure=None），不阻断入库。
    # 默认开启：与 STRUCTURED_PARSING_ENABLED(PDF) 对齐，md 图片才会进入 OCR/VL 图意化链路。
    MD_STRUCTURED_PARSING_ENABLED: bool = True

    # 图片增强档位：off / ocr / auto / all（决策：默认 auto）
    #   off  - 不做 OCR/视觉大模型，仅保留 alt/占位（保底兼容值）
    #   ocr  - 仅内存 OCR
    #   auto - 先 OCR；文本为空或过短时再调用视觉大模型（推荐，兼顾成本与召回）
    #   all  - OCR + 视觉大模型双写
    # 视觉大模型 API Key 未配置/调用失败时，auto/all 自动降级为仅 OCR/原文，不阻断入库。
    MD_IMAGE_ENHANCE_MODE: str = "auto"

    # auto 模式阈值：OCR 文本短于此长度（字符）时调用视觉大模型
    MD_IMAGE_AUTO_VL_MIN_TEXT: int = 20

    # 图片静态托管根目录（相对 CWD 或绝对路径）；服务挂载到 /md_images
    # 目录结构：
    #   <MD_IMAGE_STATIC_DIR>/<doc_id>/           每文档图片（删除文档时整目录清理）
    #   <MD_IMAGE_STATIC_DIR>/_cache/             内容寻址缓存（md5 去重，跨文档复用 OCR/VL 结果）
    MD_IMAGE_STATIC_DIR: str = "assets/md_images"

    # 是否下载 md 中引用的外链 http(s) 图片（默认关：外链图片不下载，仅保留 alt）
    MD_IMAGE_URL_DOWNLOAD: bool = False

    # 单张图片大小上限（KB），超过则跳过该图 OCR/VL（避免拖垮上传/索引）
    MD_IMAGE_MAX_KB: int = 5120

    # 是否复用 STRUCTURED_PARSER_FLOW_LLM_* 视觉模型配置（md 图片描述使用同一套视觉通道）
    MD_IMAGE_USE_DOCSTRUCT_VL_CONFIG: bool = True

    # ==================== 旧纯文本链路增强（缺陷2/3修复，默认关保持旧行为） ====================
    # 说明：以下开关全部默认 False，关闭时旧链路行为与改动前完全一致；
    #       开启后依次增强"阅读顺序、页眉页脚过滤、OCR 按页插入"。
    #
    # 旧链路 PDF 阅读顺序：page.get_text(sort=True)（按坐标排序，避免跨栏/表格错乱）
    LEGACY_PDF_LAYOUT_ORDER: bool = False

    # 旧链路页眉页脚过滤：两遍法统计"出现在 ≥80% 页面的重复行"并按页码位置剔除
    LEGACY_PDF_HEADER_FOOTER_FILTER: bool = False

    # 旧链路 OCR 增强（默认关）：
    # - 按页插入（OCR 文本紧跟对应页文本，而非追加全文末尾）
    # - 图片去噪（尺寸过滤 + md5 去重 + 水印/logo 黑名单区域）
    # - 内存处理（BytesIO，不落临时文件）
    LEGACY_PDF_OCR_INLINE: bool = False
    LEGACY_PDF_OCR_MIN_SIZE: int = 50          # 小于 50x50 px 的图片视为水印/装饰，跳过 OCR
    LEGACY_PDF_OCR_PARALLEL: bool = False      # 并行 OCR（ThreadPoolExecutor）

    # ==================== 邮件配置（Agent 请假场景） ====================
    # 通过 SMTP 发送请假申请等邮件。未配置时 send_email 工具会自动降级为生成预览。
    SMTP_HOST: str = "smtp.qq.com"                        # SMTP 服务器地址，如 smtp.qq.com
    SMTP_PORT: int = 587                       # SMTP 端口（SSL 通常 465，TLS 通常 587）
    SMTP_USERNAME: str = "123349332@qq.com"                    # 发件邮箱账号
    SMTP_PASSWORD: str = ""                    # 发件邮箱授权码（非登录密码）
    SMTP_FROM: str = ""                        # 发件人显示地址，空则使用 SMTP_USERNAME
    SMTP_USE_TLS: bool = True                  # 是否使用 TLS 加密（否则用 SSL）

    # ==================== Agent 请假流程追踪日志开关 ====================
    # 开启后在请假全链路（收到问题→意图识别→余额查询→合规检查→发送邮件）打印步骤日志
    # 便于分析和调试。关闭后不影响任何功能，仅不输出追踪 print。
    AGENT_LEAVE_TRACE: bool = True

    # ==================== 第三方办公用品系统配置（Agent 领用场景） ====================
    # Agent 的 submit_application 工具提交领用申请的目标 REST 接口。
    SUPPLY_API_URL: str = "http://localhost:8001/api/v1/supply/application"

    class Config:
        """Pydantic Settings 配置"""
        env_file = ".env"                     # 环境变量文件路径
        env_file_encoding = "utf-8"           # 环境变量文件编码


# 全局单例 - 应用启动时加载配置
settings = Settings()