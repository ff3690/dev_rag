"""
FastAPI应用主入口 - FastAPI Application Entry Point

该文件是研发团队私有化RAG+Agent项目的主入口，负责初始化FastAPI应用、
配置跨域中间件、注册路由等核心功能。

设计决策：
1. 使用FastAPI框架构建RESTful API，提供高性能和自动文档生成
2. 使用CORS中间件允许跨域请求，便于前端开发
3. 路由采用模块化管理，通过include_router注册各模块路由
4. 提供健康检查接口，便于监控和运维
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

# 导入各模块的路由
from backend.auth.routes import router as auth_router
from backend.rag.routes import router as rag_router
from backend.graph.routes import router as graph_router
from backend.agent.routes import router as agent_router

# 加载环境变量（从.env文件）
load_dotenv()

# 创建FastAPI应用实例
app = FastAPI(
    title="dev-team-rag-agent",    # API标题
    version="1.0.0"                 # API版本号
)

# 配置跨域资源共享（CORS）中间件
# 允许前端应用从任意域名访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（生产环境应限制具体域名）
    allow_credentials=True,        # 允许携带凭证（Cookie等）
    allow_methods=["*"],           # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],           # 允许所有请求头
)

# 注册认证模块路由（/auth前缀）
app.include_router(auth_router)

# 注册RAG模块路由（/rag前缀）
app.include_router(rag_router)

# 注册知识图谱模块路由（/graph前缀）
app.include_router(graph_router)

# 注册Agent智能体模块路由（/agent前缀）
app.include_router(agent_router)

# 挂载前端静态文件目录
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# 挂载 Markdown 图片静态目录（/md_images，目录存在时才挂载）
try:
    from backend.config.settings import settings
    md_img_dir = getattr(settings, "MD_IMAGE_STATIC_DIR", "assets/md_images")
    if not os.path.isabs(md_img_dir):
        md_img_dir = os.path.join(os.getcwd(), md_img_dir)
    if os.path.isdir(md_img_dir):
        app.mount("/md_images", StaticFiles(directory=md_img_dir), name="md_images")
except Exception:
    # 挂载失败仅打印提示，不影响 API 启动
    print("[MAIN] /md_images 静态目录挂载失败（忽略）")


@app.get("/health")
async def health_check():
    """
    健康检查接口 - Health Check Endpoint
    
    用于监控服务状态，返回服务健康信息。
    
    返回:
        dict: 包含状态和服务名称
    """
    return {"status": "healthy", "service": "dev-team-rag-agent"}


@app.get("/")
async def root():
    """
    根路径接口 - Root Endpoint
    
    返回欢迎信息，用于验证API是否正常运行。
    
    返回:
        dict: 包含欢迎消息
    """
    return {"message": "Welcome to dev-team-rag-agent API"}