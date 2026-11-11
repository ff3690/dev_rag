"""
RAG API路由模块 - RAG API Routes Module

该模块定义RAG系统的RESTful API接口，包含文档管理和智能问答功能。

设计决策：
1. 使用FastAPI的APIRouter组织路由，便于模块化管理
2. 使用Depends注入数据库会话和用户认证依赖
3. 文件上传使用UploadFile处理，支持批量上传
4. 流式响应使用StreamingResponse实现SSE（Server-Sent Events）
5. 全局初始化核心组件实例，确保复用
6. 集成 DeepSeek API 实现真正的 RAG 问答
"""

import json
import time
from contextlib import contextmanager
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import shutil


# ============================================================
# 检索链路步骤计时器 - 统一的步骤日志打印工具
# ------------------------------------------------------------
# 用法:
#   with _step_timer(1, "问题解析"):
#       # 执行步骤逻辑
#
#   _step_skip(4, "精排处理", "enable_rerank=False")
#
#   with _step_timer(5, "LLM生成", is_last=True):
#       # 执行最后一步
# ============================================================

@contextmanager
def _step_timer(step_num: int, name: str, is_last: bool = False):
    """
    步骤计时上下文管理器

    自动在步骤开始和结束时打印日志，并计算耗时（毫秒级）。
    性能影响极小：仅两次 perf_counter 调用 + 两次 print。

    参数:
        step_num: 步骤序号（从1开始）
        name: 步骤名称（如"问题解析"、"向量粗排"）
        is_last: 是否为最后一步（True 时前缀改为"最后一步"）
    """
    prefix = "最后一步" if is_last else f"第{step_num}步"
    print(f"{prefix} [{name}] 开始")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{prefix} [{name}] 结束 耗时: {elapsed_ms:.1f}ms")


def _step_skip(step_num: int, name: str, reason: str):
    """
    打印步骤跳过日志

    当某步骤因参数未开启而跳过时调用，保持日志链路完整性。

    参数:
        step_num: 步骤序号
        name: 步骤名称
        reason: 跳过原因（如"enable_rerank=False"）
    """
    print(f"第{step_num}步 [{name}] 跳过（{reason}）")

from backend.db.database import get_db
from backend.auth import models, dependencies
from backend.auth.models import Document as DB_Document
from backend.auth.models import Conversation as DB_Conversation
from backend.graph.models import KnowledgeNode, KnowledgeEdge
from backend.rag.document_loader import DocumentLoader, Document
from backend.rag.vector_store import LlamaVectorStore
from backend.rag.incremental_updater import IncrementalUpdater
from backend.rag.retriever import VectorRetriever, RetrievedDocument, SourceTracker
from backend.rag.graphrag_retriever import GraphRAGRetriever, GraphRAGResult
from backend.rag.embedding_service import embedding_service
from backend.rag.reranker_service import reranker_service
from backend.rag.llm_service import llm_service
from backend.auth.schemas import UserResponse
from backend.graph.graph_service import graph_service
from backend.config.settings import settings

from pydantic import BaseModel

class RagQueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    # 是否启用 BGE-Reranker 精排（默认关闭）
    # True: 检索流程为 粗排 → 精排 → 返回
    # False: 仅使用向量粗排结果
    enable_rerank: bool = False


def save_conversation_record(db: Session, user, question: str, answer: str,
                             sources, token_usage, model: str = "") -> None:
    """
    将一次问答保存到历史记录表。

    参数:
        db: 数据库会话
        user: 当前登录用户
        question: 用户问题
        answer: 回答内容（正常回答或降级返回的向量内容）
        sources: 引用来源（JSON 序列化后存储）
        token_usage: Token 用量（JSON 序列化后存储）
        model: 本次回答使用的模型名称
    """
    # sources 与 token_usage 以 JSON 字符串存储，便于序列化
    record = DB_Conversation(
        user_id=user.id,
        question=question,
        answer=answer or "",
        sources=json.dumps(sources, ensure_ascii=False) if sources else "[]",
        token_usage=json.dumps(token_usage, ensure_ascii=False) if token_usage else "{}",
        model=model or "",
        conversation_type="rag"
    )
    db.add(record)
    db.commit()
    print(f"[历史记录] 已保存问答记录 id={record.id}，用户={user.username}")


# 创建APIRouter实例，前缀为"/rag"，标签为"rag"
router = APIRouter(prefix="/rag", tags=["rag"])

# 全局初始化核心组件实例（使用 LlamaIndex + DeepSeek 版本）
# 检索流程：向量粗排 → BGE-Reranker精排 → LLM生成
vector_store = LlamaVectorStore()
updater = IncrementalUpdater(vector_store)

# VectorRetriever 注入 reranker_service，启用精排功能
# 如果 RERANKER_ENABLED=False 或模型不可用，自动降级到仅粗排
retriever = VectorRetriever(vector_store, reranker_service=reranker_service)
source_tracing = SourceTracker()
document_loader = DocumentLoader()

# GraphRAG 双路检索器（向量 + 知识图谱）
graphrag = GraphRAGRetriever(retriever)

# 文件上传目录
UPLOAD_DIR = "assets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===========================================================================
# Markdown + zip 上传支持（决策：放入现有上传接口）
# ===========================================================================
# zip 上传会话目录前缀：解压目录会保留（供 preview 读取），
# 删除文档时 delete_document 会清理前缀目录。
MDZIP_PREFIX = "mdzip_"


def _decode_zip_member_name(member) -> str:
    """
    还原 zip 成员文件名的原始编码，避免中文名解压后乱码。

    背景：Windows「发送到 → 压缩(zipped)文件夹」等打包器写中文名时使用本地
    ANSI(GBK) 字节，且不设置 UTF-8 标志位（flag_bits & 0x800）。此时 zipfile
    会按 CP437 解码，得到「╜ß╣╣╗»…」这类乱码文件名。

    策略：无 UTF-8 标志位时先按 CP437 还原出原始字节，再依次按
    UTF-8 → GBK → BIG5 解码（UTF-8 优先，避免 GBK 误判 UTF-8 字节）；
    全部失败则保持 zipfile 的原始结果，不阻断上传。
    """
    name = member.filename
    if member.flag_bits & 0x800:  # 打包器已声明 UTF-8，直接使用
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "gbk", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def _safe_zip_extract(zip_path: str, dest_root: str) -> None:
    """
    安全解压 zip：拒绝路径穿越（..）、绝对路径与盘符路径，
    并还原中文成员名的原始编码（见 _decode_zip_member_name）。

    异常: ValueError（非法成员）由调用方转 HTTPException
    """
    import re as _re
    import zipfile
    root_abs = os.path.normpath(os.path.abspath(dest_root))
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = _decode_zip_member_name(member).replace("\\", "/")
            # 拒绝绝对路径/盘符/上层目录穿越
            if name.startswith("/") or _re.match(r"^[A-Za-z]:", name):
                raise ValueError(f"zip 包含非法成员路径: {member.filename}")
            parts = name.split("/")
            if ".." in parts:
                raise ValueError(f"zip 包含路径穿越: {member.filename}")
            target = os.path.normpath(os.path.join(dest_root, name))
            if os.path.commonpath([root_abs, os.path.abspath(target)]) != root_abs:
                raise ValueError(f"zip 成员超出解压目录: {member.filename}")
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                import shutil as _sh
                _sh.copyfileobj(src, dst)


def _remove_upload_file_safely(file_path: str) -> bool:
    """
    删除上传的物理文件（仅允许删除 assets 目录内的文件，防越权）。
    返回是否删除了文件。
    """
    file_abs = file_path if os.path.isabs(file_path) else os.path.join(os.getcwd(), file_path)
    assets_abs = os.path.normpath(os.path.abspath(UPLOAD_DIR))
    try:
        inside = os.path.commonpath([assets_abs, os.path.normpath(os.path.abspath(file_abs))]) == assets_abs
    except Exception:
        inside = False
    if not inside or not os.path.isfile(file_abs):
        return False
    try:
        os.remove(file_abs)
        return True
    except Exception:
        return False


def _cleanup_upload_session_dir(file_path: str) -> None:
    """
    删除 zip 上传产生的会话目录（mdzip_* 前缀，已空时向上清理）。
    """
    cur = os.path.dirname(file_path) if os.path.isabs(file_path) \
        else os.path.dirname(os.path.join(os.getcwd(), file_path))
    assets_abs = os.path.normpath(os.path.abspath(UPLOAD_DIR))
    while cur and os.path.commonpath([assets_abs, os.path.normpath(cur)]) == assets_abs \
            and cur != assets_abs:
        if os.path.basename(cur).startswith(MDZIP_PREFIX) and os.path.isdir(cur):
            try:
                os.rmdir(cur)  # 仅删除空目录
            except OSError:
                break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent


def _schedule_graph_task(background_tasks, db_document_id, doc_content, doc_title) -> None:
    """GRAPH_ENABLED 开关控制：后台异步构建知识图谱（不阻塞上传响应）。"""
    if not settings.GRAPH_ENABLED:
        print(f"[上传] 跳过知识图谱构建（GRAPH_ENABLED=False）- document_id={db_document_id}")
        return

    def build_graph(document_id, content, title):
        import traceback
        print(f"[图谱后台任务] 开始执行 - document_id={document_id}, title={title}, "
              f"内容长度={len(content) if content else 0}")
        try:
            result = graph_service.extract_and_save(content, title, document_id, db)
            print(f"[图谱后台任务] 执行结果: {result}")
            if result.get("error"):
                print(f"[图谱后台任务] 图谱构建出错: {result['error']}")
        except Exception as e:
            print(f"[图谱后台任务] 异常: {str(e)}")
            traceback.print_exc()
        finally:
            print(f"[图谱后台任务] 执行结束 - document_id={document_id}")

    background_tasks.add_task(build_graph, db_document_id, doc_content, doc_title)


def _index_single_doc(doc, file_path: str, category: str, department: str,
                      db: Session, background_tasks) -> int:
    """
    单文档入库（向量 + DB 记录 + 图谱后台任务）公共逻辑。

    zip 内每个 md 文件与单文件上传共用本逻辑，保证两条入口行为一致。
    返回 DB_Document.id。
    """
    doc.category = category
    doc.department = department
    doc_id = updater.handle_document_add(doc)   # structured 切片（md 图意化在切片阶段执行）
    db_document = DB_Document(
        title=doc.title,
        file_path=file_path,
        file_type=doc.file_type,
        category=category,
        department=department,
        vector_index=doc_id,
        status="indexed"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    _schedule_graph_task(background_tasks, db_document.id, doc.content, doc.title)
    return db_document.id


async def _process_zip_upload(file, category: str, department: str,
                              db: Session, background_tasks) -> List[int]:
    """
    zip 上传处理：安全解压 -> 逐个 md 文件结构化入库 -> 返回 DB 文档 ID 列表。

    - 仅处理 zip 内的 *.md / *.markdown 文件（图片作为其资源由相对路径解析）；
    - 单个 md 失败不阻断整体（记录到 errors，全部失败才抛 400）；
    - 解压目录保留供 preview 使用，删除文档时清理。
    """
    import uuid
    session = f"{MDZIP_PREFIX}{uuid.uuid4().hex[:12]}"
    zip_path = os.path.join(UPLOAD_DIR, f"{session}.zip")
    dest_root = os.path.join(UPLOAD_DIR, session)

    # 读上传字节
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传的 zip 为空")
    try:
        os.makedirs(dest_root, exist_ok=True)
        with open(zip_path, "wb") as f:
            f.write(raw)
        _safe_zip_extract(zip_path, dest_root)
    except Exception as e:
        for p in (zip_path, dest_root):
            try:
                import shutil as _sh2
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    _sh2.rmtree(p, ignore_errors=True)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"zip 解压失败: {e}")

    md_files = []
    for root, _dirs, fnames in os.walk(dest_root):
        for fn in fnames:
            if fn.lower().endswith((".md", ".markdown")):
                md_files.append(os.path.join(root, fn))
    md_files.sort()

    if not md_files:
        import shutil as _sh3
        _sh3.rmtree(dest_root, ignore_errors=True)
        try:
            os.remove(zip_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="zip 中未找到 .md/.markdown 文件")

    document_ids: List[int] = []
    errors: List[str] = []
    for md_path in md_files:
        rel_path = os.path.relpath(md_path, os.getcwd()) if os.path.isabs(md_path) else md_path
        try:
            doc = document_loader.load(md_path, structured=True)
            if not doc or not (getattr(doc, "content", "") or "").strip():
                errors.append(f"{os.path.basename(md_path)}: 内容为空")
                continue
            document_ids.append(_index_single_doc(
                doc, rel_path, category, department, db, background_tasks))
            print(f"[上传][zip] ✓ {os.path.basename(md_path)} -> doc_id={document_ids[-1]}")
        except Exception as e:
            errors.append(f"{os.path.basename(md_path)}: {e}")
            print(f"[上传][zip] ✗ {os.path.basename(md_path)} 入库失败: {e}")

    # 删除临时 zip（解压目录保留供 preview）
    try:
        os.remove(zip_path)
    except Exception:
        pass

    if not document_ids:
        raise HTTPException(status_code=400,
                            detail=f"zip 内所有文档入库失败: {'; '.join(errors[:5])}")
    if errors:
        print(f"[上传][zip] 部分文档失败: {errors}")
    return document_ids


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    category: str = Form(...),
    department: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    上传文档并建立向量索引，同时自动构建知识图谱
    
    参数:
        files: 上传的文件列表，支持PDF、Word、Markdown、Excel、TXT格式
        category: 文档分类（接口文档/故障排查/需求文档/架构图纸）
        department: 所属部门
        background_tasks: FastAPI 后台任务
        current_user: 当前登录用户（通过认证依赖注入）
        db: 数据库会话（通过依赖注入）
        
    返回:
        dict: 包含成功状态、文档ID列表和消息
        
    异常:
        HTTPException: 文件处理失败时返回400错误
    """
    document_ids = []
    print(f"   开始上传文件")
    for file in files:
        # 文件名取 basename，防路径穿越
        filename = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not filename:
            continue

        # zip 上传（决策：放入现有上传接口）：内部 md 逐个结构化入库
        if filename.lower().endswith(".zip"):
            try:
                doc_ids = await _process_zip_upload(file, category, department,
                                                    db, background_tasks)
                document_ids.extend(doc_ids)
                print(f"   上传 zip 完成，共 {len(doc_ids)} 个 md 文档")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail=f"Failed to process {filename}: {str(e)}")
            continue

        # 保存上传的文件到本地
        file_path = os.path.join(UPLOAD_DIR, filename)
        print(f"   上传的文件路径{file_path}")
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        try:
            # 加载文档（structured=True：PDF/MD + 各自总开关开启时生效，
            # 解析失败自动降级回旧链路，不影响入库）
            doc = document_loader.load(file_path, structured=True)
            print(f"   加载文档{doc}")

            # 公共入库逻辑（向量 + DB 记录 + 图谱后台任务）
            db_id = _index_single_doc(doc, file_path, category, department,
                                      db, background_tasks)
            document_ids.append(db_id)

        except Exception as e:
            # 单文件入库失败：不阻断其他文件
            print(f"[上传] ✗ {filename} 入库失败: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process {filename}: {str(e)}")

    return {"success": True, "document_ids": document_ids, "message": "文档上传成功"}


@router.get("/documents")
async def get_document_list(
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db),
    category: Optional[str] = None,
    department: Optional[str] = None
):
    """
    获取文档列表（支持筛选）
    
    参数:
        current_user: 当前登录用户
        db: 数据库会话
        category: 文档分类筛选（可选）
        department: 部门筛选（可选）
        
    返回:
        list: 文档列表，包含文档ID、标题、路径、类型、分类、部门、状态和创建时间
    """
    query = db.query(DB_Document)
    
    # 根据分类筛选
    if category:
        query = query.filter(DB_Document.category == category)
    # 根据部门筛选
    if department:
        query = query.filter(DB_Document.department == department)
    
    documents = query.all()
    
    return [
        {
            "id": doc.id,
            "title": doc.title,
            "file_path": doc.file_path,
            "file_type": doc.file_type,
            "category": doc.category,
            "department": doc.department,
            "status": doc.status,
            "created_at": doc.created_at
        }
        for doc in documents
    ]


@router.get("/documents/{document_id}")
async def get_document_detail(
    document_id: int,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取文档详情
    
    参数:
        document_id: 文档ID
        current_user: 当前登录用户
        db: 数据库会话
        
    返回:
        dict: 文档详细信息
        
    异常:
        HTTPException: 文档不存在时返回404错误
    """
    doc = db.query(DB_Document).filter(DB_Document.id == document_id).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": doc.id,
        "title": doc.title,
        "file_path": doc.file_path,
        "file_type": doc.file_type,
        "category": doc.category,
        "department": doc.department,
        "status": doc.status,
        "created_at": doc.created_at
    }


@router.get("/documents/{document_id}/preview")
async def preview_document(
    document_id: int,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    预览文档 - 返回文档文件流，供前端在新标签页中打开预览
    
    参数:
        document_id: 文档ID
        current_user: 当前登录用户
        db: 数据库会话
        
    返回:
        FileResponse: 文档文件流
        
    异常:
        HTTPException: 文档不存在或文件缺失时返回404错误
    """
    doc = db.query(DB_Document).filter(DB_Document.id == document_id).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = doc.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document file not found")
    
    # 根据文件类型推断 media_type，保证浏览器可正常打开预览
    file_ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
    }
    media_type = media_types.get(file_ext, "application/octet-stream")
    
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


@router.put("/documents/{document_id}")
async def update_document(
    document_id: int,
    file: UploadFile = File(...),
    category: str = Form(None),
    department: str = Form(None),
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    更新文档内容
    
    参数:
        document_id: 文档ID
        file: 新的文件内容
        category: 新的分类（可选）
        department: 新的部门（可选）
        current_user: 当前登录用户
        db: 数据库会话
        
    返回:
        dict: 包含成功状态和消息
        
    异常:
        HTTPException: 文档不存在时返回404错误，处理失败时返回400错误
    """
    doc = db.query(DB_Document).filter(DB_Document.id == document_id).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 保存新文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        # 加载新文档内容（structured=True：PDF/MD + 各自总开关开启时生效，失败自动降级）
        loaded_doc = document_loader.load(file_path, structured=True)

        # 更新前清理旧 md 图片目录（同 doc_id 目录内旧图不残留）
        if doc.vector_index and getattr(loaded_doc, "file_type", "") in ("md", "markdown"):
            try:
                from backend.rag.markdown_image_service import cleanup_doc_images
                cleanup_doc_images(doc.vector_index)
            except Exception as e:
                print(f"[update_document] ✗ 清理 md 图片目录失败: {e}")

        # 更新向量存储（structured=True：保留结构化中间产物供结构感知切片）
        if doc.vector_index:
            updater.handle_document_update(doc.vector_index, loaded_doc)
        
        # 更新数据库记录
        if category:
            doc.category = category
        if department:
            doc.department = department
        doc.file_path = file_path
        doc.file_type = loaded_doc.file_type
        doc.status = "updated"
        
        db.commit()
        db.refresh(doc)
        
        return {"success": True, "message": "文档更新成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除文档
    
    参数:
        document_id: 文档ID
        current_user: 当前登录用户
        db: 数据库会话
        
    返回:
        dict: 包含成功状态和消息
        
    异常:
        HTTPException: 文档不存在时返回404错误
    """
    print(f"[delete_document] ===== 开始删除文档，document_id={document_id} =====")
    doc = db.query(DB_Document).filter(DB_Document.id == document_id).first()
    
    if not doc:
        print(f"[delete_document] ✗ 文档不存在 document_id={document_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    print(f"[delete_document] 找到文档: title={doc.title}, vector_index={doc.vector_index}, file_path={doc.file_path}")
    
    # 删除向量存储中的数据
    if doc.vector_index:
        print(f"[delete_document] → 开始删除向量，传入 vector_index={doc.vector_index}")
        try:
            updater.handle_document_delete(doc.vector_index)
            print(f"[delete_document] ✓ 向量删除调用完成")
        except Exception as e:
            print(f"[delete_document] ✗ 向量删除调用异常: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[delete_document] ⚠ 该文档 vector_index 为空，跳过向量删除（可能从未建立向量索引）")
    
    # 删除该文档关联的知识图谱数据（节点、边）
    try:
        result = graph_service.delete_by_document(document_id, db)
        print(f"[delete_document] ✓ 知识图谱数据已删除: {result}")
    except Exception as e:
        # 图谱删除失败不应阻断文档删除，记录异常继续执行
        print(f"[delete_document] ✗ 删除知识图谱数据失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 统一记录删除变更日志（无论是否建立向量索引，都保证 change_log 与删除状态一致）
    # 注意：record_change(delete) 会直接移除该 document_id 的所有日志记录，而非追加 delete 标志
    try:
        updater.record_change(str(doc.vector_index or document_id), "delete")
        print(f"[delete_document] ✓ 变更日志已更新: 移除该 document_id 的所有记录")
    except Exception as e:
        print(f"[delete_document] ✗ 更新变更日志失败: {e}")
        import traceback
        traceback.print_exc()

    # 删除 Markdown 图片文档目录（<MD_IMAGE_STATIC_DIR>/<doc_id>/）
    # doc_id 即 doc.vector_index（md 结构化切片阶段以 document.id 命名落盘）
    try:
        from backend.rag.markdown_image_service import cleanup_doc_images
        if doc.vector_index:
            cleaned = cleanup_doc_images(doc.vector_index)
            if cleaned:
                print(f"[delete_document] ✓ 已删除 md 图片目录: {doc.vector_index}")
    except Exception as e:
        print(f"[delete_document] ✗ 删除 md 图片目录失败: {e}")

    # 删除上传时保存的物理文件（支持单文件平铺与 zip 解压的嵌套路径，仅限 assets 内）
    if doc.file_path:
        try:
            file_abs = doc.file_path
            if not os.path.isabs(file_abs):
                file_abs = os.path.join(os.getcwd(), file_abs)
            if os.path.isfile(file_abs):
                if _remove_upload_file_safely(doc.file_path):
                    print(f"[delete_document] ✓ 已删除物理文件: {file_abs}")
                else:
                    print(f"[delete_document] ⚠ 物理文件不在 assets 内或删除失败，跳过: {file_abs}")
            else:
                print(f"[delete_document] ⚠ 物理文件不存在，跳过: {file_abs}")
            # zip 解压会话目录（mdzip_*，删空后向上清理）仅保留该文件的子目录
            _cleanup_upload_session_dir(doc.file_path)
        except Exception as e:
            print(f"[delete_document] ✗ 删除物理文件失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[delete_document] ⚠ 该文档无 file_path，跳过物理文件删除")

    # 删除数据库记录
    db.delete(doc)
    db.commit()
    print(f"[delete_document] ✓ 数据库记录已删除，document_id={document_id}")
    
    return {"success": True, "message": "文档删除成功"}


@router.post("/clear-all")
async def clear_all_data(
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    清除所有数据（重置系统到初始状态）

    执行以下清理操作：
    1. 删除整个向量索引目录 vector_data/llama_index
    2. 清空 assets 目录中的文件（保留目录本身）
    3. 将 change_log.json 内容清空为 []
    4. 清空数据库中的 conversations、documents、knowledge_edges、knowledge_nodes 表数据

    参数：
        current_user: 当前登录用户（仅允许已登录用户操作）
        db: 数据库会话

    返回：
        dict: 各清理步骤的统计信息
    """
    result = {
        "success": True,
        "llama_index_deleted": False,
        "assets_files_deleted": 0,
        "conversations_deleted": 0,
        "documents_deleted": 0,
        "knowledge_edges_deleted": 0,
        "knowledge_nodes_deleted": 0,
    }

    # 1. 删除向量索引目录 vector_data/llama_index（整个文件夹）
    llama_index_path = os.path.join(settings.VECTOR_STORE_PATH, "llama_index")
    try:
        if os.path.isdir(llama_index_path):
            shutil.rmtree(llama_index_path)
            result["llama_index_deleted"] = True
            print(f"[clear-all] ✓ 已删除向量索引目录: {llama_index_path}")
        else:
            print(f"[clear-all] ⚠ 向量索引目录不存在，跳过: {llama_index_path}")
    except Exception as e:
        print(f"[clear-all] ✗ 删除向量索引目录失败: {e}")

    # 2. 清空 assets 目录内容（保留目录本身）
    assets_dir = "assets"
    try:
        if os.path.isdir(assets_dir):
            deleted_count = 0
            for name in os.listdir(assets_dir):
                item_path = os.path.join(assets_dir, name)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"[clear-all] ✗ 删除 assets 下 {name} 失败: {e}")
            result["assets_files_deleted"] = deleted_count
            print(f"[clear-all] ✓ 已清空 assets 目录内容，删除 {deleted_count} 项")
        else:
            os.makedirs(assets_dir, exist_ok=True)
            print(f"[clear-all] ⚠ assets 目录不存在，已创建: {assets_dir}")
    except Exception as e:
        print(f"[clear-all] ✗ 清空 assets 目录失败: {e}")

    # 2.1 清空 Markdown 图片静态根目录（含内容寻址缓存；若位于 assets 内已被上一步清理）
    try:
        from backend.rag.markdown_image_service import cleanup_md_image_root
        cleanup_md_image_root()
        print(f"[clear-all] ✓ 已清理 md 图片静态目录（含缓存）")
    except Exception as e:
        print(f"[clear-all] ✗ 清理 md 图片静态目录失败: {e}")

    # 3. 清空 change_log.json 为 []
    change_log_path = os.path.join(settings.VECTOR_STORE_PATH, "change_log.json")
    try:
        os.makedirs(os.path.dirname(change_log_path), exist_ok=True)
        with open(change_log_path, "w", encoding="utf-8") as f:
            f.write("[]")
        print(f"[clear-all] ✓ 已清空变更日志: {change_log_path}")
    except Exception as e:
        print(f"[clear-all] ✗ 清空变更日志失败: {e}")

    # 4. 清空数据库表数据
    try:
        # 先清空有外键依赖的边表，再清空节点表，避免外键约束冲突
        result["knowledge_edges_deleted"] = db.query(KnowledgeEdge).delete()
        result["knowledge_nodes_deleted"] = db.query(KnowledgeNode).delete()
        result["conversations_deleted"] = db.query(DB_Conversation).delete()
        result["documents_deleted"] = db.query(DB_Document).delete()
        db.commit()
        print("[clear-all] ✓ 已清空数据库表: conversations/documents/knowledge_edges/knowledge_nodes")
    except Exception as e:
        db.rollback()
        print(f"[clear-all] ✗ 清空数据库表失败: {e}")
        result["success"] = False

    return result


# 最低相似度阈值 - 低于此分数的结果被认为不相关
# bge-small-zh 模型在中文场景下（粗排向量相似度）：
# - 相关文档: 0.5-0.8 分
# - 弱相关文档: 0.35-0.5 分
# - 不相关文档: 0.2-0.35 分
# 因此设为 0.45 可以有效过滤不相关结果
MIN_SIMILARITY_THRESHOLD = 0.45

# 精排分数阈值（bge-reranker Cross-Encoder 的 sigmoid 输出，与粗排向量相似度不同量纲）
# 相关文档精排分通常落在 0.1-0.9，这里取 0.1 作为最低相关线，避免误杀可用结果
RERANK_SCORE_RELEVANT_THRESHOLD = 0.1


def _is_relevant(r) -> bool:
    """判断检索结果是否相关。精排与粗排分数量纲不同，必须按来源分别使用阈值。"""
    if getattr(r, "rerank_score", None) is not None:
        # 经过精排：使用精排分数阈值
        return r.score >= RERANK_SCORE_RELEVANT_THRESHOLD
    # 仅粗排/精排降级：使用向量相似度阈值
    return r.score >= MIN_SIMILARITY_THRESHOLD

@router.post("/query")
async def rag_query(
    request: RagQueryRequest,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    RAG智能问答接口（集成 DeepSeek API）

    完整的 RAG 流程：
    1. 使用 VectorRetriever 检索相关文档
    2. 过滤低质量结果（低于相似度阈值）
    3. 将高质量检索结果作为上下文传给 DeepSeek LLM
    4. LLM 基于上下文生成回答
    5. 返回回答、来源引用和 Token 使用量

    参数:
        request: 查询请求体，包含question、conversation_id、top_k
        current_user: 当前登录用户
        db: 数据库会话

    返回:
        dict: 包含回答内容、来源信息、对话ID、Token使用量和延迟
    """
    pipeline_start = time.perf_counter()

    # 第1步: 问题解析
    with _step_timer(1, "问题解析"):
        question = request.question
        top_k = request.top_k
        use_rerank = request.enable_rerank
        print(f"  问题: \"{question}\"")
        print(f"  参数: top_k={top_k}, enable_rerank={use_rerank}, 图谱=N/A（纯向量接口）")

    # 第2步: 检索策略选择
    with _step_timer(2, "检索策略选择"):
        if use_rerank:
            strategy = "向量粗排 → BGE-Reranker精排 → 结果返回"
        else:
            strategy = "向量粗排 → 结果返回（精排关闭）"
        print(f"  策略: {strategy}")

    # 第3步: 检索执行（内部包含粗排+可选精排，详细日志见 retriever.py）
    with _step_timer(3, "检索执行"):
        results = retriever.retrieve(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        print(f"  检索返回: {len(results)} 条结果")
        for i, r in enumerate(results):
            score_type = "精排分" if r.rerank_score is not None else "粗排分"
            print(f"  [{i+1}] {score_type}={r.score:.4f} | {r.content[:60]}...")

    # 第4步: 结果过滤与相关性判断
    with _step_timer(4, "结果过滤"):
        has_relevant_results = any(_is_relevant(r) for r in results)
        relevant_count = sum(1 for r in results if _is_relevant(r))
        print(f"  相关结果: {relevant_count}/{len(results)} 条")

    if not results or not has_relevant_results:
        # 无相关文档，返回友好提示
        no_result_answer = f"抱歉，在当前知识库中未找到相关文档。查询\"{request.question}\"未命中任何相关内容。"
        print(f"  结果: 无相关文档，返回降级提示")

        if not request.conversation_id:
            conversation_id = str(uuid.uuid4())
        else:
            conversation_id = request.conversation_id

        # 保存历史记录（无命中结果）
        save_conversation_record(db, current_user, request.question, no_result_answer, [],
                                 {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms（无结果降级返回） ==========")
        return {
            "answer": no_result_answer,
            "sources": [],
            "conversation_id": conversation_id,
            "total_sources": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "",
            "is_fallback": True,
            "latency_ms": 0.0
        }

    # 最后一步: LLM生成回答
    with _step_timer(5, "LLM生成", is_last=True):
        # 说明：若 API Key 不可用/无网络/调用失败，llm_service 内部会降级，
        # 不调用大模型，直接基于向量检索内容生成回答（is_fallback=True）。
        llm_response = llm_service.generate(question, results)
        print(f"  模型: {llm_response.model}, 降级: {llm_response.is_fallback}")
        print(f"  Token: {llm_response.usage}")

    # 结果格式化与历史保存
    with _step_timer(6, "结果格式化与历史保存"):
        sources = source_tracing.format_sources(results)

        if not request.conversation_id:
            conversation_id = str(uuid.uuid4())
        else:
            conversation_id = request.conversation_id

        save_conversation_record(db, current_user, question, llm_response.answer,
                                 sources, llm_response.usage, llm_response.model)

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    print(f"========== 检索链路总耗时: {total_ms:.1f}ms ==========")

    return {
        "answer": llm_response.answer,
        "sources": sources,
        "conversation_id": conversation_id,
        "total_sources": len(results),
        "token_usage": llm_response.usage,
        "model": llm_response.model,
        "is_fallback": llm_response.is_fallback,
        "latency_ms": round(llm_response.latency_ms, 2)
    }


@router.post("/query-stream")
async def rag_query_stream(
    question: str,
    conversation_id: Optional[str] = None,
    top_k: int = 5,
    # 是否启用 BGE-Reranker 精排（默认关闭，仅粗排）
    enable_rerank: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    RAG智能问答流式接口（SSE，集成 DeepSeek API）

    完整的流式 RAG 流程：
    1. 发送开始事件
    2. 检索相关文档并发送来源信息
    3. 过滤低质量结果
    4. 调用 DeepSeek API 流式生成回答
    5. 逐步返回生成的文本片段
    6. 发送结束事件

    参数:
        question: 用户问题
        conversation_id: 对话ID（可选）
        top_k: 返回的最大文档数量，默认5
        current_user: 当前登录用户
        db: 数据库会话

    返回:
        StreamingResponse: 流式响应，包含多个SSE事件

    事件类型：
    - start: 开始处理
    - source: 检索到的来源文档
    - token: LLM 生成的文本片段
    - end: 处理完成
    """
    pipeline_start = time.perf_counter()

    # 第1步: 问题解析
    with _step_timer(1, "问题解析"):
        print(f"  问题: \"{question}\"")
        print(f"  参数: top_k={top_k}, enable_rerank={enable_rerank}, 图谱=N/A（纯向量接口）")

    # 第2步: 检索策略选择
    with _step_timer(2, "检索策略选择"):
        if enable_rerank:
            strategy = "向量粗排 → BGE-Reranker精排 → 流式LLM生成"
        else:
            strategy = "向量粗排 → 流式LLM生成（精排关闭）"
        print(f"  策略: {strategy}")

    # 第3步: 检索执行（内部包含粗排+可选精排，详细日志见 retriever.py）
    with _step_timer(3, "检索执行"):
        results = retriever.retrieve(question, top_k=top_k, use_rerank=enable_rerank)
        print(f"  检索返回: {len(results)} 条结果")
        for i, r in enumerate(results):
            score_type = "精排分" if r.rerank_score is not None else "粗排分"
            print(f"  [{i+1}] {score_type}={r.score:.4f} | {r.content[:60]}...")

    # 第4步: 结果过滤
    with _step_timer(4, "结果过滤"):
        has_relevant_results = any(_is_relevant(r) for r in results)
        relevant_count = sum(1 for r in results if _is_relevant(r))
        print(f"  相关结果: {relevant_count}/{len(results)} 条")

    if not results or not has_relevant_results:
        # 无相关文档，直接返回提示
        conv_id = conversation_id or str(uuid.uuid4())
        no_result_answer = f"抱歉，在当前知识库中未找到相关文档。查询\"{question}\"未命中任何相关内容。"
        print(f"  结果: 无相关文档，返回降级提示")

        async def generate_no_result():
            yield json.dumps({
                "type": "start",
                "message": "开始处理您的问题...",
                "conversation_id": conv_id
            }) + "\n"

            yield json.dumps({
                "type": "end",
                "message": no_result_answer,
                "answer": no_result_answer,
                "total_sources": 0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": ""
            }, ensure_ascii=False)

            # 保存历史记录（无命中结果，独立会话保存）
            from backend.db.database import SessionLocal
            save_db = SessionLocal()
            try:
                save_conversation_record(
                    save_db, current_user, question, no_result_answer, [],
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            except Exception as e:
                print(f"[历史记录] 保存失败: {e}")
            finally:
                save_db.close()

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms（无结果降级返回） ==========")
        return StreamingResponse(generate_no_result(), media_type="text/event-stream")

    # 第5步: 来源格式化
    with _step_timer(5, "来源格式化"):
        sources = source_tracing.format_sources(results)
        conv_id = conversation_id or str(uuid.uuid4())

    # 最后一步: 流式LLM生成
    print(f"最后一步 [流式LLM生成] 开始")
    llm_t0 = time.perf_counter()

    async def generate():
        # 发送开始事件
        yield json.dumps({
            "type": "start",
            "message": "开始处理您的问题...",
            "conversation_id": conv_id
        }) + "\n"

        # 发送来源信息
        yield json.dumps({
            "type": "sources",
            "message": f"找到 {len(results)} 个相关文档",
            "sources": sources
        }, ensure_ascii=False) + "\n"

        # 流式调用 LLM 生成回答
        answer_chunks = []
        async for chunk in llm_service.agenerate_stream(question, results):
            answer_chunks.append(chunk)
            yield json.dumps({
                "type": "token",
                "content": chunk
            }, ensure_ascii=False) + "\n"

        # 发送结束事件
        full_answer = "".join(answer_chunks)
        token_usage = llm_service._last_usage or {}
        print(f"  [诊断] end 事件发送前 _last_usage = {token_usage}")
        yield json.dumps({
            "type": "end",
            "message": "生成完成",
            "answer": full_answer,
            "total_sources": len(results),
            "token_usage": token_usage,
            "model": llm_service.model
        }, ensure_ascii=False) + "\n"

        llm_elapsed_ms = (time.perf_counter() - llm_t0) * 1000
        print(f"最后一步 [流式LLM生成] 结束 耗时: {llm_elapsed_ms:.1f}ms")

        # 自动保存问答历史
        def save_conversation():
            from backend.db.database import SessionLocal
            save_db = SessionLocal()
            try:
                save_conversation_record(save_db, current_user, question,
                                         full_answer, sources, token_usage, llm_service.model)
            except Exception as e:
                print(f"[历史记录] 保存失败: {e}")
            finally:
                save_db.close()

        background_tasks.add_task(save_conversation)

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms ==========")

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/sources/{conversation_id}")
async def get_sources(
    conversation_id: str,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取对话来源信息（预留接口）
    
    参数:
        conversation_id: 对话ID
        current_user: 当前登录用户
        db: 数据库会话
        
    返回:
        dict: 包含消息和对话ID
        
    设计决策：当前为预留接口，后续可实现完整的溯源功能，
    根据对话ID查询历史来源信息。
    """
    return {"message": "溯源功能开发中", "conversation_id": conversation_id}


# ============================================================
# 问答历史记录 API
# ============================================================

@router.get("/history")
async def get_history(
    page: int = 1,
    page_size: int = 20,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    分页查询当前用户的问答历史记录（按提问时间倒序）

    参数:
        page: 页码，从1开始，默认1
        page_size: 每页条数，默认20
        current_user: 当前登录用户
        db: 数据库会话

    返回:
        dict: 包含 records（历史记录列表）、total（总条数）、page、page_size
    """
    # 校验分页参数
    page = max(1, page)
    page_size = max(1, min(100, page_size))

    # 查询当前用户的历史记录总数（用于前端分页）
    total = db.query(DB_Conversation).filter(
        DB_Conversation.user_id == current_user.id
    ).count()

    # 按时间倒序分页查询
    conversations = (
        db.query(DB_Conversation)
        .filter(DB_Conversation.user_id == current_user.id)
        .order_by(DB_Conversation.created_at.desc(), DB_Conversation.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    records = []
    for conv in conversations:
        # 解析 sources 与 token_usage（以 JSON 字符串存储，避免解析失败则返回原始/空）
        sources = conv.sources
        token_usage = conv.token_usage
        try:
            if conv.sources:
                sources = json.loads(conv.sources)
        except (json.JSONDecodeError, TypeError):
            sources = None
        try:
            if conv.token_usage:
                token_usage = json.loads(conv.token_usage)
        except (json.JSONDecodeError, TypeError):
            token_usage = None

        records.append({
            "id": conv.id,
            "question": conv.question,
            "answer": conv.answer or "",
            "sources": sources or [],
            "token_usage": token_usage,
            "model": conv.model or "",
            "created_at": conv.created_at,
        })

    return {
        "records": records,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.delete("/history")
async def delete_history(
    ids: List[int] = Body(...),
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    批量删除当前用户的问答历史记录

    参数:
        ids: 待删除的历史记录ID列表
        current_user: 当前登录用户
        db: 数据库会话

    返回:
        dict: 包含删除成功状态和删除条数
    """
    if not ids:
        raise HTTPException(status_code=400, detail="No record ids provided")

    # 仅删除属于当前用户的记录，保证用户隔离
    result = db.query(DB_Conversation).filter(
        DB_Conversation.id.in_(ids),
        DB_Conversation.user_id == current_user.id
    ).delete(synchronize_session=False)
    db.commit()

    return {"success": True, "deleted": result}


# ============================================================
# GraphRAG 双路融合检索接口
# ============================================================

class GraphRAGQueryRequest(BaseModel):
    """GraphRAG 查询请求"""
    question: str
    top_k: int = 5
    graph_hops: int = 1
    include_graph: bool = True
    # 是否启用 BGE-Reranker 精排（默认关闭）
    enable_rerank: bool = False


@router.post("/graphrag-query")
async def graphrag_query(
    request: GraphRAGQueryRequest,
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    GraphRAG 双路融合检索问答接口

    融合向量检索（语义匹配）和知识图谱检索（实体关系），提供增强检索能力。

    处理流程：
    1. 向量检索 Top-K 文档片段
    2. 实体链接：匹配查询中的实体到知识图谱节点
    3. 子图提取：从匹配节点 BFS 提取 N 跳邻域
    4. 融合双路上下文，LLM 生成回答

    参数:
        request: 包含 question、top_k、graph_hops、include_graph
        current_user: 当前登录用户
        db: 数据库会话

    返回:
        dict: 包含回答、来源、子图数据、Token 用量
    """
    pipeline_start = time.perf_counter()

    # 第1步: 问题解析
    with _step_timer(1, "问题解析"):
        question = request.question
        top_k = request.top_k
        use_rerank = request.enable_rerank
        include_graph = request.include_graph
        graph_hops = request.graph_hops
        print(f"  问题: \"{question}\"")
        print(f"  参数: top_k={top_k}, enable_rerank={use_rerank}, "
              f"include_graph={include_graph}, graph_hops={graph_hops}")

    # 第2步: 检索策略选择
    with _step_timer(2, "检索策略选择"):
        parts = ["向量粗排"]
        if use_rerank:
            parts.append("BGE-Reranker精排")
        else:
            parts.append("精排关闭")
        if include_graph:
            parts.append(f"图谱检索({graph_hops}跳)")
        else:
            parts.append("图谱关闭")
        parts.append("LLM生成")
        strategy = " → ".join(parts)
        print(f"  策略: {strategy}")

    # 第3步: 双路检索执行（内部包含向量粗排+可选精排+图谱检索，详细日志见各模块）
    with _step_timer(3, "双路检索执行"):
        result: GraphRAGResult = graphrag.retrieve(
            query=question,
            db=db,
            top_k=top_k,
            graph_hops=graph_hops,
            include_graph=include_graph,
            use_rerank=use_rerank,
        )
        vector_results = result.vector_results
        subgraph = result.subgraph
        print(f"  向量检索: {len(vector_results)} 条结果")
        print(f"  图谱子图: {len(subgraph.nodes)} 个节点, {len(subgraph.edges)} 条边, "
              f"{len(subgraph.matched_entities)} 个匹配实体")
        if vector_results:
            for i, r in enumerate(vector_results):
                score_type = "精排分" if r.rerank_score is not None else "粗排分"
                print(f"  [{i+1}] {score_type}={r.score:.4f} | {r.content[:60]}...")

    # 第4步: 结果过滤
    with _step_timer(4, "结果过滤"):
        has_relevant_results = any(_is_relevant(r) for r in vector_results)
        relevant_count = sum(1 for r in vector_results if _is_relevant(r))
        print(f"  相关结果: {relevant_count}/{len(vector_results)} 条")

    # 第5步: 上下文整合
    with _step_timer(5, "上下文整合"):
        # 修复：不再把"图谱上下文 + 全部文档片段"拼成单个超长字符串。
        # 原实现下，_build_context 对超长文档是"整体跳过"，
        # 图谱拼接串一旦超过上下文限额，唯一文档被丢弃 → 上下文为空（"（无相关文档）"），
        # LLM 依据系统提示词误判"未找到相关信息"而输出 fallback，
        # 这是 A/B 评测中 completeness_on=0（开图谱反而答不出）的根因。
        # 现改为：图谱上下文与各文档片段拆成独立的 RetrievedDocument 列表，
        # 由 _build_context 按限额对各文档分别截断，保证文档片段始终进入上下文。
        enhanced_docs = []
        if result.graph_context_text and result.graph_context_text != "（知识图谱中未找到相关实体）":
            from backend.rag.retriever import RetrievedDocument
            enhanced_docs.append(
                RetrievedDocument(
                    document_id="graphrag_kg",
                    chunk_id="graphrag_kg_0",
                    content=f"## 知识图谱结构化上下文\n{result.graph_context_text}",
                    score=1.0,
                    metadata={
                        "document_id": "graphrag_kg",
                        "chunk_index": -1,
                        "search_type": "graphrag",
                    }
                )
            )
            enhanced_docs.extend(vector_results)
            print(f"  整合: 图谱上下文 + {len(vector_results)} 个文档片段（拆分传递）")
        else:
            if not include_graph:
                print(f"  整合: 仅文档片段（include_graph=False, 图谱检索未执行）")
            else:
                print(f"  整合: 仅文档片段（图谱未匹配到实体）")

    # 无相关结果处理
    if not vector_results or not has_relevant_results:
        no_result_answer = f"抱歉，在当前知识库中未找到与「{question}」相关的内容。"
        print(f"  结果: 无相关文档，返回降级提示")

        conversation_id = str(uuid.uuid4())
        save_conversation_record(db, current_user, question, no_result_answer, [],
                                 {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                 model="graphrag")

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms（无结果降级返回） ==========")
        return {
            "answer": no_result_answer,
            "sources": [],
            "subgraph": {
                "nodes": [{"id": n.id, "name": n.name, "type": n.node_type}
                           for n in subgraph.nodes],
                "edges": [{"id": e.id, "source": e.source_id, "target": e.target_id,
                            "relation": e.relation, "source_name": e.source_name,
                            "target_name": e.target_name}
                           for e in subgraph.edges],
            },
            "conversation_id": conversation_id,
            "total_sources": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "graphrag",
            "is_fallback": True,
            "latency_ms": 0.0,
        }

    # 最后一步: LLM生成回答
    with _step_timer(6, "LLM生成", is_last=True):
        if enhanced_docs:
            # 使用增强上下文（图谱 + 文档片段，拆分列表）调用 LLM。
            # 修复说明：enhanced_docs 由"图谱上下文文档 + 各文档片段"构成，
            # _build_context 会按限额分别截断，避免单个超长文档被整体丢弃。
            print(f"  使用增强上下文（图谱+文档，共 {len(enhanced_docs)} 条）调用 LLM")
            llm_response = llm_service.generate(question, enhanced_docs)
        else:
            print(f"  使用纯文档片段调用 LLM")
            llm_response = llm_service.generate(question, vector_results)
        print(f"  模型: {llm_response.model}, 降级: {llm_response.is_fallback}")
        print(f"  Token: {llm_response.usage}")

    # 结果格式化与历史保存
    with _step_timer(7, "结果格式化与历史保存"):
        sources = source_tracing.format_sources(vector_results)
        conversation_id = str(uuid.uuid4())
        save_conversation_record(db, current_user, question, llm_response.answer,
                                 sources, llm_response.usage, llm_response.model)

    # 构建子图数据（供前端可视化）
    with _step_timer(8, "子图数据构建"):
        subgraph_data = {
            "nodes": [
                {
                    "id": f"kg_{n.id}",
                    "name": n.name,
                    "type": n.node_type,
                    "description": n.description,
                    "is_matched": any(
                        me.node_id == n.id for me in subgraph.matched_entities
                    )
                }
                for n in subgraph.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": f"kg_{e.source_id}",
                    "target": f"kg_{e.target_id}",
                    "relation": e.relation,
                    "source_name": e.source_name,
                    "target_name": e.target_name,
                }
                for e in subgraph.edges
            ],
            "matched_entities": [
                {"name": me.name, "type": me.node_type, "score": me.match_score}
                for me in subgraph.matched_entities
            ],
            "graph_context_text": result.graph_context_text,
        }

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    print(f"========== 检索链路总耗时: {total_ms:.1f}ms ==========")

    return {
        "answer": llm_response.answer,
        "sources": sources,
        "subgraph": subgraph_data,
        "conversation_id": conversation_id,
        "total_sources": len(vector_results),
        "token_usage": llm_response.usage,
        "model": f"graphrag_{llm_response.model}",
        "is_fallback": llm_response.is_fallback,
        "latency_ms": round(llm_response.latency_ms, 2),
    }


# ============================================================
# GraphRAG 双路检索流式接口
# ============================================================

def _build_subgraph(result: GraphRAGResult) -> dict:
    """将 GraphRAGResult 的子图序列化为前端可用的字典结构"""
    return {
        "nodes": [
            {
                "id": f"kg_{n.id}",
                "name": n.name,
                "type": n.node_type,
                "description": n.description,
                "is_matched": any(
                    me.node_id == n.id for me in result.subgraph.matched_entities
                )
            }
            for n in result.subgraph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "source": f"kg_{e.source_id}",
                "target": f"kg_{e.target_id}",
                "relation": e.relation,
                "source_name": e.source_name,
                "target_name": e.target_name,
            }
            for e in result.subgraph.edges
        ],
        "matched_entities": [
            {"name": me.name, "type": me.node_type, "score": me.match_score}
            for me in result.subgraph.matched_entities
        ],
        "graph_context_text": result.graph_context_text,
    }


@router.post("/graphrag-query-stream")
async def graphrag_query_stream(
    request: GraphRAGQueryRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: models.User = Depends(dependencies.get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    GraphRAG 双路检索流式接口（SSE）

    融合向量检索（语义匹配）和知识图谱检索（实体关系），并以流式方式输出回答。

    处理流程：
    1. graphrag.retrieve 阻塞执行双路检索（向量 Top-K + 图谱 BFS 子图）
    2. 过滤低质量结果（score >= MIN_SIMILARITY_THRESHOLD）
    3. 构建增强上下文（图谱结构化上下文 + 文档片段）
    4. 无相关结果：流式返回 start -> subgraph -> end（无 token）
    5. 有相关结果：start -> sources -> subgraph -> token(逐块) -> end

    事件类型：
    - start: 开始处理，携带 conversation_id
    - sources: 检索到的来源文档
    - subgraph: 知识图谱子图摘要数据（nodes/edges/matched_entities）
    - token: LLM 生成的文本片段
    - end: 处理完成，携带完整 answer / token_usage / model
    """
    pipeline_start = time.perf_counter()

    # 第1步: 问题解析
    with _step_timer(1, "问题解析"):
        question = request.question
        top_k = request.top_k
        use_rerank = request.enable_rerank
        include_graph = request.include_graph
        graph_hops = request.graph_hops
        print(f"  问题: \"{question}\"")
        print(f"  参数: top_k={top_k}, enable_rerank={use_rerank}, "
              f"include_graph={include_graph}, graph_hops={graph_hops}")

    # 第2步: 检索策略选择
    with _step_timer(2, "检索策略选择"):
        parts = ["向量粗排"]
        if use_rerank:
            parts.append("BGE-Reranker精排")
        else:
            parts.append("精排关闭")
        if include_graph:
            parts.append(f"图谱检索({graph_hops}跳)")
        else:
            parts.append("图谱关闭")
        parts.append("流式LLM生成")
        strategy = " → ".join(parts)
        print(f"  策略: {strategy}")

    # 第3步: 双路检索执行（阻塞，内部含向量粗排+可选精排+图谱检索）
    with _step_timer(3, "双路检索执行"):
        result: GraphRAGResult = graphrag.retrieve(
            query=question,
            db=db,
            top_k=top_k,
            graph_hops=graph_hops,
            include_graph=include_graph,
            use_rerank=use_rerank,
        )
        vector_results = result.vector_results
        subgraph = result.subgraph
        print(f"  向量检索: {len(vector_results)} 条结果")
        print(f"  图谱子图: {len(subgraph.nodes)} 个节点, {len(subgraph.edges)} 条边, "
              f"{len(subgraph.matched_entities)} 个匹配实体")
        if vector_results:
            for i, r in enumerate(vector_results):
                score_type = "精排分" if r.rerank_score is not None else "粗排分"
                print(f"  [{i+1}] {score_type}={r.score:.4f} | {r.content[:60]}...")

    # 第4步: 结果过滤
    with _step_timer(4, "结果过滤"):
        has_relevant_results = any(_is_relevant(r) for r in vector_results)
        relevant_count = sum(1 for r in vector_results if _is_relevant(r))
        print(f"  相关结果: {relevant_count}/{len(vector_results)} 条")

    # 第5步: 上下文整合与格式化
    with _step_timer(5, "上下文整合与格式化"):
        conv_id = str(uuid.uuid4())
        subgraph_data = _build_subgraph(result)

        # 修复：不再把"图谱上下文 + 全部文档片段"拼成单个超长字符串。
        # 原实现下，_build_context 对超长文档是"整体跳过"，
        # 图谱拼接串一旦超过上下文限额，唯一文档被丢弃 → 上下文为空（"（无相关文档）"），
        # LLM 依据系统提示词误判"未找到相关信息"而输出 fallback。
        # 现改为：图谱上下文与各文档片段拆成独立的 RetrievedDocument 列表，
        # 由 _build_context 按限额对各文档分别截断，保证文档片段始终进入上下文。
        enhanced_docs = []
        if result.graph_context_text and result.graph_context_text != "（知识图谱中未找到相关实体）":
            from backend.rag.retriever import RetrievedDocument
            enhanced_docs.append(
                RetrievedDocument(
                    document_id="graphrag_kg",
                    chunk_id="graphrag_kg_0",
                    content=f"## 知识图谱结构化上下文\n{result.graph_context_text}",
                    score=1.0,
                    metadata={
                        "document_id": "graphrag_kg",
                        "chunk_index": -1,
                        "search_type": "graphrag",
                    }
                )
            )
            enhanced_docs.extend(vector_results)
            print(f"  整合: 图谱上下文 + {len(vector_results)} 个文档片段（拆分传递）")
        else:
            if not include_graph:
                print(f"  整合: 仅文档片段（include_graph=False, 图谱检索未执行）")
            else:
                print(f"  整合: 仅文档片段（图谱未匹配到实体）")

    # 无相关结果：仅返回 start -> subgraph -> end
    if not vector_results or not has_relevant_results:
        no_result_answer = f"抱歉，在当前知识库中未找到与「{question}」相关的内容。"
        print(f"  结果: 无相关文档，返回降级提示")

        async def generate_no_result():
            yield json.dumps({
                "type": "start",
                "message": "开始处理您的问题...",
                "conversation_id": conv_id
            }, ensure_ascii=False) + "\n"

            yield json.dumps({
                "type": "subgraph",
                "subgraph": subgraph_data
            }, ensure_ascii=False) + "\n"

            yield json.dumps({
                "type": "end",
                "message": no_result_answer,
                "answer": no_result_answer,
                "total_sources": 0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "graphrag"
            }, ensure_ascii=False) + "\n"

            # 后台保存历史记录
            from backend.db.database import SessionLocal
            save_db = SessionLocal()
            try:
                save_conversation_record(
                    save_db, current_user, question, no_result_answer, [],
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    model="graphrag")
            except Exception as e:
                print(f"[历史记录] 保存失败: {e}")
            finally:
                save_db.close()

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms（无结果降级返回） ==========")
        return StreamingResponse(generate_no_result(), media_type="text/event-stream")

    # 第6步: 来源格式化与增强文档构建
    with _step_timer(6, "来源格式化与增强文档构建"):
        sources = source_tracing.format_sources(vector_results)

        # 修复说明：enhanced_docs 已在第5步拆分构建（图谱上下文文档 + 各文档片段），
        # 此处直接沿用，不再二次拼接成单个超长文档，
        # 避免单个超长文档被 _build_context 整体丢弃导致上下文为空。
        if enhanced_docs:
            print(f"  构建增强文档（图谱+文档融合，共 {len(enhanced_docs)} 条）")
        else:
            print(f"  使用原始向量结果（无图谱增强）")

    # 最后一步: 流式LLM生成
    print(f"最后一步 [流式LLM生成] 开始")
    llm_t0 = time.perf_counter()

    async def generate():
        # 1. 开始事件
        yield json.dumps({
            "type": "start",
            "message": "开始处理您的问题...",
            "conversation_id": conv_id
        }, ensure_ascii=False) + "\n"

        # 2. 来源事件
        yield json.dumps({
            "type": "sources",
            "message": f"找到 {len(vector_results)} 个相关文档",
            "sources": sources
        }, ensure_ascii=False) + "\n"

        # 3. 子图摘要事件
        yield json.dumps({
            "type": "subgraph",
            "subgraph": subgraph_data
        }, ensure_ascii=False) + "\n"

        # 4. 流式生成回答（使用增强上下文或原始向量结果）
        docs = enhanced_docs if enhanced_docs else vector_results
        answer_chunks = []
        async for chunk in llm_service.agenerate_stream(question, docs):
            answer_chunks.append(chunk)
            yield json.dumps({
                "type": "token",
                "content": chunk
            }, ensure_ascii=False) + "\n"

        # 5. 结束事件
        full_answer = "".join(answer_chunks)
        token_usage = llm_service._last_usage or {}
        yield json.dumps({
            "type": "end",
            "message": "生成完成",
            "answer": full_answer,
            "total_sources": len(vector_results),
            "token_usage": token_usage,
            "model": f"graphrag_{llm_service.model}"
        }, ensure_ascii=False) + "\n"

        llm_elapsed_ms = (time.perf_counter() - llm_t0) * 1000
        print(f"最后一步 [流式LLM生成] 结束 耗时: {llm_elapsed_ms:.1f}ms")

        # 6. 后台保存历史记录
        from backend.db.database import SessionLocal
        save_db = SessionLocal()
        try:
            save_conversation_record(
                save_db, current_user, question, full_answer, sources,
                token_usage, model=f"graphrag_{llm_service.model}")
        except Exception as e:
            print(f"[历史记录] 保存失败: {e}")
        finally:
            save_db.close()

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        print(f"========== 检索链路总耗时: {total_ms:.1f}ms ==========")

    return StreamingResponse(generate(), media_type="text/event-stream")