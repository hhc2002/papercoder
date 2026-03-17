"""
FAISS 跨会话长期记忆
记住历史研究过的论文、算法结论和代码骨架
"""
import json
import numpy as np
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


MEMORY_DIR = Path(__file__).parent.parent / "memory_store"
INDEX_PATH = MEMORY_DIR / "faiss.index"
RECORDS_PATH = MEMORY_DIR / "records.json"
EMBED_DIM = 384


class LongTermMemory:
    """跨会话语义记忆：历史论文研究知识积累"""

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.records: list = []
        self._encoder = None
        self._index = None
        self._available = FAISS_AVAILABLE and ST_AVAILABLE

        if self._available:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            self._load()
        else:
            print("[Memory] FAISS 或 sentence-transformers 未安装，长期记忆降级为 JSON 简单搜索")
            self._load_json_only()

    # ── 写入 ──────────────────────────────────────────────────────

    def save(self, query: str, summary: str, code_snippet: str = ""):
        record = {"query": query, "summary": summary, "code": code_snippet}

        if self._available:
            embedding = self._encode(query)
            self._index.add(np.array([embedding], dtype=np.float32))

        self.records.append(record)
        self._persist()

    # ── 检索 ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.records:
            return ""

        if self._available:
            embedding = self._encode(query)
            _, indices = self._index.search(
                np.array([embedding], dtype=np.float32), min(top_k, len(self.records))
            )
            hits = [self.records[i] for i in indices[0] if 0 <= i < len(self.records)]
        else:
            # 简单关键词回退
            hits = [r for r in self.records if any(w in r["query"] for w in query.split())][:top_k]

        if not hits:
            return ""

        lines = ["【历史研究记录】"]
        for r in hits:
            lines.append(f"- 论文：{r['query']}\n  结论：{r['summary'][:300]}")
        return "\n".join(lines)

    # ── 内部 ──────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        vec = self._encoder.encode([text])[0]
        # L2 归一化（用于内积检索等效余弦相似度）
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _persist(self):
        if self._available:
            faiss.write_index(self._index, str(INDEX_PATH))
        with open(RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def _load(self):
        if INDEX_PATH.exists() and RECORDS_PATH.exists():
            self._index = faiss.read_index(str(INDEX_PATH))
            with open(RECORDS_PATH, encoding="utf-8") as f:
                self.records = json.load(f)

    def _load_json_only(self):
        if RECORDS_PATH.exists():
            with open(RECORDS_PATH, encoding="utf-8") as f:
                self.records = json.load(f)


# 全局单例
_memory_instance: LongTermMemory | None = None


def get_memory() -> LongTermMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance
