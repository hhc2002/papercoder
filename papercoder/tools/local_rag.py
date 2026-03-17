"""
本地 RAG 工具 — LlamaIndex + ChromaDB
检索本地 rag/（手动放置）和 papers/（arXiv 自动下载）目录中的论文 PDF
向量索引持久化在 chroma_db/
"""
import os
from pathlib import Path
from langchain_core.tools import tool

RAG_DIR = Path(__file__).parent.parent / "rag"      # 用户手动放置 PDF 的目录
PAPERS_DIR = Path(__file__).parent.parent / "papers"  # arXiv 自动下载缓存目录
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

_index = None  # 懒加载


def _collect_pdf_files() -> list[Path]:
    """从 rag/ 和 papers/ 两个目录收集所有 PDF"""
    pdfs = []
    for d in (RAG_DIR, PAPERS_DIR):
        if d.exists():
            pdfs.extend(d.glob("*.pdf"))
    return pdfs


def _build_index():
    """构建或加载本地 RAG 索引"""
    global _index
    if _index is not None:
        return _index

    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        # 检查是否有本地 PDF（rag/ 或 papers/ 均可）
        pdf_files = _collect_pdf_files()
        if not pdf_files:
            return None

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        chroma_collection = chroma_client.get_or_create_collection("papers")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 如果 Chroma 集合为空则重新索引
        if chroma_collection.count() == 0:
            print(f"[LocalRAG] 正在索引 {len(pdf_files)} 篇本地 PDF...")
            # 从 rag/ 和 papers/ 分别加载
            documents = []
            for d in (RAG_DIR, PAPERS_DIR):
                if d.exists() and list(d.glob("*.pdf")):
                    documents.extend(
                        SimpleDirectoryReader(str(d), required_exts=[".pdf"]).load_data()
                    )
            _index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
        else:
            _index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )

        return _index

    except ImportError as e:
        print(f"[LocalRAG] 依赖未安装: {e}")
        return None
    except Exception as e:
        print(f"[LocalRAG] 索引构建失败: {e}")
        return None


@tool
def local_rag_tool(query: str) -> str:
    """检索本地 rag/ 或 papers/ 目录中已存储的论文，获取相关段落"""
    index = _build_index()

    if index is None:
        pdf_files = _collect_pdf_files()
        if not pdf_files:
            return "[本地RAG] 无本地 PDF，跳过本地检索。将论文 PDF 放入 rag/ 目录可启用本地检索。"
        return "[本地RAG] 索引构建失败，请检查依赖（llama-index, chromadb）"

    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        return f"[本地RAG 检索结果]\n{str(response)}"
    except Exception as e:
        return f"[本地RAG] 检索失败: {e}"


def index_local_papers():
    """手动触发本地 PDF 索引（可在 main.py 调用）"""
    global _index
    _index = None  # 强制重建
    result = _build_index()
    if result:
        pdf_count = len(_collect_pdf_files())
        print(f"[LocalRAG] 已索引 {pdf_count} 篇本地论文（来源：rag/ + papers/）")
    return result is not None
