"""
PaperCoderState — LangGraph 全局状态定义
支持 Researcher 和 Coder 并行写入不同字段
"""
from typing import TypedDict, List, Optional, Annotated
import operator


def _last_value(a, b):
    """并行节点写同一字段时取最新值"""
    return b if b is not None else a


class PaperCoderState(TypedDict, total=False):
    # ── 用户输入 ──────────────────────────────────────────────────
    query: str                        # 用户输入（论文标题 / arXiv ID / 研究问题）
    pdf_path: Optional[str]           # 本地 PDF 路径（可选）

    # ── Planner 输出 ──────────────────────────────────────────────
    subtasks: List[dict]              # 分解的子任务列表（含 type: research/code）
    memory_context: str               # FAISS 历史论文上下文

    # ── Researcher 输出 ───────────────────────────────────────────
    retrieved_docs: List[dict]        # 检索结果（arXiv / S2 / Tavily / RAG）

    # ── Coder 输出 ────────────────────────────────────────────────
    paper_content: str                # 解析后的论文全文
    algo_description: str             # 提取的算法描述
    pseudocode: str                   # 提取的伪代码
    code_draft: str                   # 生成的 Python 代码骨架
    diagram: str                      # Mermaid 流程图
    github_refs: List[dict]           # GitHub 开源实现引用

    # ── Reviewer 输出 ─────────────────────────────────────────────
    draft_report: str                 # 报告草稿（Reviewer 评估前生成）
    feedback: str                     # Reviewer 改进建议
    score: float                      # Reviewer 综合评分（0-10）
    iteration: int                    # 当前迭代次数（Circuit Breaker）

    # ── Survey 模式专用 ──────────────────────────────────────────
    survey_type: Optional[str]        # "topic" | "followup"
    base_paper: Optional[str]         # followup 模式的基础论文名

    # ── Reporter 输出 ─────────────────────────────────────────────
    final_output: dict                # 最终三件套：{text_review, diagram, code}
