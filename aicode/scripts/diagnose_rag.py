"""
RAG 系统诊断脚本

用于排查向量检索问题，检查：
1. 索引加载状态
2. 文档数量
3. 向量数据完整性
4. 检索功能测试
"""

import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose():
    print("=" * 60)
    print("RAG 系统诊断")
    print("=" * 60)
    
    # 1. 检查索引目录
    print("\n【1】检查索引目录...")
    index_path = "vector_data/llama_index"
    if os.path.exists(index_path):
        files = os.listdir(index_path)
        print(f"  ✓ 索引目录存在")
        print(f"  文件列表: {files}")
    else:
        print(f"  ✗ 索引目录不存在: {index_path}")
        return
    
    # 2. 检查 docstore.json 中的文档数量
    print("\n【2】检查文档数量...")
    docstore_path = os.path.join(index_path, "docstore.json")
    if os.path.exists(docstore_path):
        with open(docstore_path, "r", encoding="utf-8") as f:
            docstore = json.load(f)
        
        data = docstore.get("docstore/data", {})
        total_docs = len(data)
        docs_with_embedding = sum(
            1 for doc in data.values() 
            if doc.get("__data__", {}).get("embedding") is not None
        )
        
        print(f"  总文档数: {total_docs}")
        print(f"  有向量的文档: {docs_with_embedding}")
        print(f"  无向量的文档: {total_docs - docs_with_embedding}")
        
        if total_docs > 0:
            # 显示第一个文档的信息
            first_doc_id = list(data.keys())[0]
            first_doc = data[first_doc_id]["__data__"]
            print(f"\n  第一个文档示例:")
            print(f"    - ID: {first_doc_id}")
            print(f"    - 内容长度: {len(first_doc.get('text', ''))} 字符")
            print(f"    - 向量维度: {len(first_doc.get('embedding') or [])}")
            print(f"    - 有嵌入: {'是' if first_doc.get('embedding') else '否 ⚠️'}")
    else:
        print(f"  ✗ docstore.json 不存在")
    
    # 3. 检查 vector_store.json
    print("\n【3】检查向量存储...")
    vs_path = os.path.join(index_path, "default__vector_store.json")
    if os.path.exists(vs_path):
        with open(vs_path, "r", encoding="utf-8") as f:
            vs_data = json.load(f)
        
        # 检查向量数据
        embeddings = vs_data.get("embedding_dict", {})
        print(f"  向量数量: {len(embeddings)}")
        
        if len(embeddings) > 0:
            # 显示第一个向量的维度
            first_key = list(embeddings.keys())[0]
            first_vector = embeddings[first_key]
            print(f"  第一个向量维度: {len(first_vector)}")
            print(f"  ✅ 向量数据正常")
        else:
            print(f"  ⚠️ 向量存储为空！")
            print(f"  问题: 文档已添加但未生成向量")
            print(f"  原因: 可能是添加文档时 embedding 服务未正确初始化")
    else:
        print(f"  ✗ 向量存储文件不存在")
    
    # 4. 测试加载索引
    print("\n【4】测试加载索引...")
    try:
        from backend.rag.vector_store import LlamaVectorStore
        vs = LlamaVectorStore()
        
        if vs._index is not None:
            print(f"  ✓ 索引加载成功")
            
            # 测试检索
            print("\n【5】测试检索功能...")
            test_query = "心胸刺痛，心悸、怔忡，胁肋隐痛，腹胀、食少"
            results = vs.search(test_query, top_k=3)
            
            print(f"  查询: '{test_query}'")
            print(f"  检索结果数: {len(results)}")
            
            if len(results) > 0:
                for i, r in enumerate(results):
                    print(f"\n  结果 {i+1}:")
                    print(f"    - 分数: {r['score']:.4f}")
                    print(f"    - 内容: {r['content'][:100]}...")
                    if r.get('metadata'):
                        print(f"    - 文档ID: {r['metadata'].get('document_id', 'N/A')}")
            else:
                print(f"  ⚠️ 未检索到任何结果！")
                print(f"  可能原因:")
                print(f"    1. 向量库为空（文档未正确嵌入）")
                print(f"    2. 查询文本与文档内容不匹配")
                print(f"    3. 嵌入模型与索引时不一致")
        else:
            print(f"  ✗ 索引加载失败")
            
    except Exception as e:
        print(f"  ✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 5. 诊断总结
    print("\n" + "=" * 60)
    print("【诊断总结】")
    print("=" * 60)
    
    total_docs = 0
    has_vectors = False
    
    if os.path.exists(docstore_path):
        with open(docstore_path, "r", encoding="utf-8") as f:
            docstore = json.load(f)
            total_docs = len(docstore.get("docstore/data", {}))
    
    if os.path.exists(vs_path):
        with open(vs_path, "r", encoding="utf-8") as f:
            vs_data = json.load(f)
            has_vectors = len(vs_data.get("embedding_dict", {})) > 0
    
    if total_docs == 0:
        print("  ⚠️ 向量库为空，请先上传文档")
    elif not has_vectors:
        print("  ⚠️ 文档存在但向量为空！")
        print("  解决方案: 需要重建索引")
        print("  运行: python scripts/rebuild_index.py")
    elif not results:
        print("  ⚠️ 有向量但检索不到结果")
        print("  可能原因: 嵌入模型版本不匹配")
        print("  建议: 重新构建索引")
    else:
        print("  ✅ RAG 系统正常！")
    
    print("\n【建议操作】")
    print("  1. 如果向量为空: 删除 vector_data/llama_index 目录")
    print("  2. 重新上传文档触发向量化")
    print("  3. 或运行: python scripts/migrate_vectors.py --force")


if __name__ == "__main__":
    diagnose()
