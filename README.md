# dev-team-rag-agent — 研发团队私有化 RAG + Agent 智能助手

> 面向企业内部研发团队的私有化智能知识问答与自动化办公系统。
> 基于 **RAG（检索增强生成）** + **知识图谱（GraphRAG）** + **LangGraph Agent 多工具编排**，深度落地企业级业务场景（办公用品领用、请假合规审批等）。

---

## 一、项目简介

本项目是一套**可私有化部署**的企业级智能助手系统，解决传统 LLM 在内部知识问答中"缺乏企业私有数据、易产生幻觉、无法溯源"的核心痛点。系统围绕两条主线构建：

1. **知识问答线（RAG / GraphRAG）**：把企业内部文档（制度、流程、规范等）加载、切分、向量化，建立知识库；用户提问时执行"向量检索 → 重排 → LLM 生成"的完整 RAG 链路，并融合知识图谱增强复杂关系问题的回答能力，所有回答**可溯源、可引用**。
2. **Agent 自动化办公线**：基于 LangGraph 状态图编排多个业务工具，实现 **请假余额查询 → 合规检查 → 自动发送审批邮件**、**办公用品消耗预测 → 自动提交领用申请** 等端到端自动化流程，替代人工重复操作。

系统采用**前后端分离**架构，配套完整的管理后台（文档管理、用户管理、问答历史、Agent 工作台、图谱可视化），支持国产化 / 私有化 LLM 与 Embedding 模型，满足企业数据不出内网的安全合规要求。

---

## 二、核心功能

### 1. 企业知识库智能问答（RAG）
- **多格式文档接入**：支持 PDF、Word（docx）、Markdown、Excel、TXT，以及图片类文档（集成 Tesseract OCR / EasyOCR 文字识别）。
- **全链路处理流水线**：文档加载 → 智能文本切分 → Embedding 向量化 → 向量索引 → 检索召回 → 相关性过滤 → LLM 生成。
- **向量粗排 + 语义精排**：基于 `bge-small-zh` 向量粗排，可选 `bge-reranker` Cross-Encoder 精排，双阈值（粗排相似度 / 精排分数）过滤不相关内容。
- **来源溯源**：回答自动携带引用来源（文档标题、片段、相关度分数），前端可展开查看，杜绝"编造答案"。
- **流式输出**：基于 SSE（Server-Sent Events）流式返回回答，支持打字机效果，首字延迟低。
- **增量更新**：文档新增 / 更新 / 删除后，向量索引与变更日志自动增量维护，无需全量重建。
- **问答历史**：分页保存与查询历史问答记录，支持批量删除。

### 2. 知识图谱增强检索（GraphRAG）
- 文档上传后自动进行**实体与关系抽取**，构建企业知识图谱（节点 + 边 + 关系描述）。
- **双路融合检索**：向量语义检索 + 图谱实体匹配 → BFS 提取 N 跳子图 → 结构化图谱上下文与文档片段融合后交给 LLM。
- 前端支持**子图可视化**，展示实体、关系与命中实体。
- 通过 `GRAPH_ENABLED` 开关可独立开启 / 关闭图谱能力。

### 3. Agent 智能体自动化办公
基于 LangGraph 构建 `decide → tool → generate` 三节点编排图，由 LLM 动态决策调用哪些工具，已注册 6 个业务工具：

| 工具 | 能力 | 场景 |
| --- | --- | --- |
| `rag_search` | 检索知识库 | 制度 / 流程问答 |
| `query_leave_balance` | 查询员工各假别年度余额 | 请假场景 |
| `check_leave_compliance` | 依据制度校验请假天数、时间、审批链合规性 | 请假场景 |
| `send_email` | 自动生成并发送审批邮件 | 请假场景 |
| `predict_consumption` | 基于历史数据预测办公用品月度消耗 | 领用场景 |
| `submit_application` | 自动提交办公用品领用申请 | 领用场景 |

**典型端到端流程（请假）**：
```
用户："帮我请 3 天事假，9月1日到9月3日，家里有事"
  ↓ decide_node(意图决策)
  ↓ query_leave_balance → 查询余额
  ↓ check_leave_compliance → 合规校验
  ↓ 【合规通过】→ 自动追加 send_email（收件人按部门自动补全主管邮箱）
  ↓ generate_node → 汇总结果生成最终回复
```

**关键工程细节**：
- **确定性规则注入**：合规检查通过后在 `tool_node` 内自动追加 `send_email`（而非依赖 LLM 二次决策），保证关键步骤不遗漏；合规不通过则不发邮件。
- **启发式路由**：终端工具（发邮件 / 提交申请）执行后直接进入生成节点；非终端工具循环回决策节点，配合 `max_iterations` 与空跑检测双重防死循环。
- **系统信息透传（known_info）**：将用户姓名、部门、已调用工具列表自动注入 Prompt，避免 LLM 向用户重复询问系统已掌握的信息，防止跳步 / 重复调用。

### 4. 用户与权限管理
- JWT 认证体系，登录态校验，接口级权限控制。
- 用户 / 部门管理（含部门主管邮箱、上下级部门关系，供邮件自动补全收件人与抄送人）。

### 5. 前端管理后台
- **Dashboard**：数据概览。
- **Chat**：智能对话（流式回答 + 来源引用 + 图谱子图展示）。
- **Agent Workbench**：Agent 工具调用过程可视化。
- **Document Management**：文档上传 / 管理 / 预览 / 删除。
- **User Management**：用户与权限管理。
- **Settings**：系统设置。

---

## 三、技术栈

| 层次 | 技术选型 | 用途 |
| --- | --- | --- |
| 后端框架 | **Python 3 + FastAPI + Uvicorn** | RESTful API、SSE 流式响应、CORS、健康检查 |
| 大模型接入 | **DeepSeek API（OpenAI 兼容协议）** | 可替换为任意私有化 / 国产化模型（Qwen、GLM 等） |
| 向量检索 | **LlamaIndex + bge-small-zh（BAAI Embedding）** | 文档切分、向量化、向量索引与检索 |
| 重排精排 | **bge-reranker（Cross-Encoder）** | 检索结果语义精排 |
| Agent 编排 | **LangGraph（StateGraph）** | 状态图节点编排、条件路由、循环控制 |
| 知识图谱 | **自研图谱存储 + LLM 实体关系抽取** | 节点 / 边存储、BFS 子图检索、GraphRAG 融合 |
| OCR | **Tesseract OCR + EasyOCR** | 扫描件 / 图片类文档文字识别 |
| 数据库 | **SQLAlchemy + SQLite** | 用户、文档、问答历史、图谱数据持久化 |
| 认证鉴权 | **JWT（python-jose / PyJWT）** | 登录态与接口鉴权 |
| 前端 | **React 18 + TypeScript + Vite + TailwindCSS + Zustand** | SPA 管理后台、SSE 流式渲染、图谱可视化 |

---

## 四、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   前端（React + TS + Vite）               │
│  Dashboard / Chat / AgentWorkbench / 文档管理 / 用户管理 │
└───────────────────────┬─────────────────────────────────┘
                        │ RESTful / SSE (JSON)
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI 后端（/api）                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│  │ auth    │  │ rag     │  │ graph   │  │ agent     │  │
│  │ 认证鉴权 │  │ RAG问答  │  │ 知识图谱 │  │ Agent编排  │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────┬─────┘  │
└───────┼────────────┼────────────┼──────────────┼───────┘
        │            │            │              │
   SQLite(JWT)  向量存储     图谱存储      LangGraph 图
   users/dept  (LlamaIndex)  (节点/边)    decide/tool/generate
                 ▲   ▲                      │
                 │   └──── bge-reranker ─────┤
           Embedding(bge-small-zh)    6 个业务工具
                 ▲                          │
                 │                    SQLite(余额/部门)
   语料库文档 → OCR → 切分 → 向量化
```

**RAG 检索链路（`/rag/query`）**：
```
问题解析 → 检索策略选择 → 向量粗排(Top-K) → (可选)BGE-Reranker 精排
        → 相关性阈值过滤 → 来源格式化 → LLM 生成 → 历史保存
```

**GraphRAG 检索链路（`/rag/graphrag-query`）**：
```
向量粗排 → 精排 → 图谱实体匹配 → BFS 提取 N 跳子图
        → 图谱结构化上下文 + 文档片段融合 → LLM 生成 → 子图数据返回前端
```

---

## 五、目录结构

```
dev-team-rag-agent/
├── aicode/                        # 后端主工程
│   ├── backend/
│   │   ├── main.py                # FastAPI 入口（注册 auth/rag/graph/agent 路由）
│   │   ├── auth/                  # JWT 认证、用户/部门模型、登录注册
│   │   ├── rag/                   # RAG 核心
│   │   │   ├── document_loader.py # 多格式文档加载（PDF/Word/MD/Excel/TXT）
│   │   │   ├── ocr_processor.py   # Tesseract/EasyOCR 图片识别
│   │   │   ├── text_splitter.py   # 智能文本切分
│   │   │   ├── embedding_service.py # bge embedding
│   │   │   ├── vector_store.py    # LlamaIndex 向量索引
│   │   │   ├── retriever.py       # 向量粗排 + Reranker 精排
│   │   │   ├── reranker_service.py# bge-reranker 精排服务
│   │   │   ├── graphrag_retriever.py # 双路融合检索器
│   │   │   ├── incremental_updater.py # 向量增量更新 + change_log
│   │   │   ├── source_tracing.py  # 来源溯源
│   │   │   ├── llm_service.py     # LLM 统一接入（含流式/降级）
│   │   │   └── routes.py          # RAG API（上传/问答/流式/历史/GraphRAG）
│   │   ├── agent/                 # LangGraph Agent
│   │   │   ├── graph_builder.py   # decide/tool/generate 三节点编排图
│   │   │   ├── agent_state.py     # Agent 状态 + Prompt 模板
│   │   │   └── tools/             # 6 个业务工具（注册表模式，可插拔扩展）
│   │   ├── graph/                 # 知识图谱（实体抽取、图存储、BFS 检索、路由）
│   │   ├── llm/                   # LLM 配置与适配
│   │   ├── config/                # 系统配置（settings）
│   │   └── db/                    # SQLite 数据库会话与模型
│   ├── assets/                    # 上传文档存储
│   ├── vector_data/               # 向量索引目录 + change_log.json
│   ├── scripts/                   # 运维脚本
│   └── tests/                     # 测试用例
├── ui设计/                        # 前端 React 应用
│   └── src/
│       ├── pages/                 # Dashboard/Chat/AgentWorkbench/文档管理/用户管理/设置
│       ├── components/            # 通用组件（图谱可视化、流式对话等）
│       └── store/                 # Zustand 状态管理
├── 语料库/                        # 企业知识语料（制度/流程/规范文档）
└── *.md                           # 分阶段设计文档（含技术方案选型）
```

---

## 六、快速开始

### 后端

```bash
# 1. 安装依赖
pip install -r aicode/requirements.txt

# 2. 配置环境变量（.env）
#   LLM_API_KEY、LLM_API_BASE、EMBEDDING_MODEL=bge-small-zh-v1.5
#   RERANKER_ENABLED、GRAPH_ENABLED 等

# 3. 启动服务
cd aicode
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- API 文档：`http://localhost:8000/docs`（FastAPI 自动生成）
- 健康检查：`http://localhost:8000/health`

### 前端

```bash
cd ui设计
npm install
npm run dev
```

---

## 七、主要 API 概览

| 模块 | 方法 | 路径 | 说明 |
| --- | --- | --- | --- |
| 系统 | GET | `/health` | 健康检查 |
| 认证 | POST | `/auth/login` `/auth/register` | 登录 / 注册 |
| RAG | POST | `/rag/upload` | 批量上传文档（自动向量化 + 图谱构建） |
| RAG | GET | `/rag/documents` | 文档列表（分类/部门筛选） |
| RAG | POST | `/rag/query` | RAG 智能问答 |
| RAG | POST | `/rag/query-stream` | RAG 流式问答（SSE） |
| RAG | POST | `/rag/graphrag-query` | GraphRAG 双路融合问答 |
| RAG | POST | `/rag/graphrag-query-stream` | GraphRAG 流式问答（SSE） |
| RAG | GET | `/rag/history` | 问答历史（分页） |
| Agent | POST | `/agent/chat` | Agent 智能体对话（工具编排） |
| 图谱 | GET | `/graph/...` | 图谱节点/边查询、可视化数据 |

