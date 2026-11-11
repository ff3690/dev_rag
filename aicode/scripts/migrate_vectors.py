"""
数据迁移脚本 - Vector Store Migration Script

将旧的 SQLite 向量存储迁移到 LlamaIndex SimpleVectorStore。
支持从数据库读取文档记录，重新建立向量索引。

使用方法:
    python scripts/migrate_vectors.py
    python scripts/migrate_vectors.py --force  # 强制重新索引
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config.settings import settings
from backend.db.database import SessionLocal
from backend.auth.models import Document as DB_Document
from backend.rag.vector_store import LlamaVectorStore
from backend.rag.incremental_updater import IncrementalUpdater
from backend.rag.document_loader import DocumentLoader


def migrate_vectors(force: bool = False):
    """
    迁移向量数据
    
    参数:
        force: 是否强制重新索引（即使已有索引）
    """
    print("=" * 60)
    print("向量存储迁移脚本")
    print("=" * 60)
    
    # 初始化组件
    print("\n[1/4] 初始化组件...")
    vector_store = LlamaVectorStore()
    updater = IncrementalUpdater(vector_store)
    document_loader = DocumentLoader()
    
    # 检查现有索引
    existing_count = vector_store.get_vector_count()
    print(f"  当前向量存储中已有 {existing_count} 条记录")
    
    if existing_count > 0 and not force:
        print("\n  提示：向量存储中已有数据。")
        print("  使用 --force 参数可以强制重新索引。")
        return
    
    # 从数据库读取文档记录
    print("\n[2/4] 读取数据库文档记录...")
    db = SessionLocal()
    try:
        documents = db.query(DB_Document).filter(DB_Document.status == "indexed").all()
        print(f"  共找到 {len(documents)} 条已索引文档")
        
        if len(documents) == 0:
            print("  没有需要迁移的文档。")
            return
        
        # 获取所有唯一的文件路径
        file_paths = list(set([doc.file_path for doc in documents if doc.file_path]))
        print(f"  需要处理 {len(file_paths)} 个唯一文件")
        
        # 清理旧数据（如果强制）
        if force and existing_count > 0:
            print("\n  清理旧向量数据...")
            # 重新初始化向量存储
            import shutil
            index_path = os.path.join(settings.VECTOR_STORE_PATH, "llama_index")
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                print("  已清理旧索引")
        
        # 重新索引文档
        print("\n[3/4] 重新建立向量索引...")
        success_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(file_paths, 1):
            if not os.path.exists(file_path):
                print(f"  [{i}/{len(file_paths)}] 跳过不存在的文件: {file_path}")
                failed_count += 1
                continue
            
            try:
                print(f"  [{i}/{len(file_paths)}] 处理文件: {os.path.basename(file_path)}")
                
                # 加载文档
                doc = document_loader.load(file_path)
                
                # 获取该文件对应的数据库记录
                db_docs = [d for d in documents if d.file_path == file_path]
                if db_docs:
                    doc.category = db_docs[0].category
                    doc.department = db_docs[0].department
                
                # 添加到向量存储
                updater.handle_document_add(doc)
                success_count += 1
                
            except Exception as e:
                print(f"    失败: {str(e)}")
                failed_count += 1
        
        # 验证结果
        print("\n[4/4] 验证迁移结果...")
        final_count = vector_store.get_vector_count()
        print(f"  迁移完成！")
        print(f"    成功: {success_count} 个文件")
        print(f"    失败: {failed_count} 个文件")
        print(f"    最终向量数: {final_count}")
        
        # 更新数据库状态
        print("\n  更新数据库文档状态...")
        for db_doc in documents:
            if db_doc.file_path in [d.file_path for d in documents]:
                db_doc.status = "indexed"
        db.commit()
        
        print("\n" + "=" * 60)
        print("迁移完成！")
        print("=" * 60)
        
    finally:
        db.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="向量存储迁移脚本")
    parser.add_argument("--force", action="store_true", help="强制重新索引")
    args = parser.parse_args()
    
    try:
        migrate_vectors(force=args.force)
    except KeyboardInterrupt:
        print("\n\n迁移已取消。")
        sys.exit(0)
    except Exception as e:
        print(f"\n迁移失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()