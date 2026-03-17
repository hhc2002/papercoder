"""
Planner Node — Plan-and-Execute 模式
1. 查询 FAISS Memory 获取历史上下文
2. 将用户输入分解为 research 和 code 两类子任务
3. 输出 Pydantic 校验的任务列表
"""
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PaperCoderState
from ..memory.long_term import get_memory
from ..llm_factory import get_llm
from ..tools.arxiv_tool import download_arxiv_pdf


class SubTask(BaseModel):
    type: str = Field(description="任务类型: research 或 code")
    description: str = Field(description="具体任务描述")
    priority: int = Field(description="优先级 1-3，1最高", ge=1, le=3)


class PlannerOutput(BaseModel):
    research_tasks: List[SubTask] = Field(description="检索类子任务列表")
    code_tasks: List[SubTask] = Field(description="实现类子任务列表")
    paper_focus: str = Field(description="论文核心聚焦点，一句话概括")


PLANNER_SYSTEM = """你是一位资深研究员，擅长论文精读和算法分析。
你的任务是分析用户的研究目标，制定详细的研究计划。

请将任务分解为两类：
1. **research 类**：需要检索的信息（背景知识、相关论文、引用关系、技术博客）
2. **code 类**：需要实现的内容（算法提取、代码生成、流程图、开源对比）

输出严格遵循 JSON 格式，包含 research_tasks、code_tasks、paper_focus 三个字段。"""

PLANNER_HUMAN = """用户研究目标：{query}

历史研究记录（如有）：
{memory_context}

请制定详细的研究计划，输出 JSON 格式的任务分解。"""


def planner_node(state: PaperCoderState) -> dict:
    """
    Planner Node
    - 查询长期记忆获取历史上下文
    - 用 LLM 分解任务
    - 返回更新后的状态片段
    """
    query = state.get("query", "")
    pdf_path = state.get("pdf_path", "")
    iteration = state.get("iteration", 0)

    print(f"\n{'='*60}")
    print(f"[Planner] 开始规划 | 查询: {query[:80]}... | 迭代: {iteration}")

    # 没有提供 PDF 时，自动从 arXiv 下载
    if not pdf_path:
        print("[Planner] 未提供 PDF，尝试从 arXiv 自动下载...")
        pdf_path = download_arxiv_pdf(query)
        if pdf_path:
            print(f"[Planner] 已获取 PDF: {pdf_path}")
        else:
            print("[Planner] arXiv 下载失败，将基于检索文本分析")

    # 查询长期记忆
    memory = get_memory()
    memory_context = memory.search(query, top_k=3)
    if memory_context:
        print(f"[Planner] 发现历史记录:\n{memory_context[:200]}...")

    # 构建 LLM 调用
    llm = get_llm()

    prompt = PLANNER_HUMAN.format(
        query=query,
        memory_context=memory_context or "无历史记录"
    )

    # 带结构化输出
    try:
        structured_llm = llm.with_structured_output(PlannerOutput)
        plan: PlannerOutput = structured_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=prompt),
        ])

        subtasks = (
            [t.model_dump() | {"type": "research"} for t in plan.research_tasks] +
            [t.model_dump() | {"type": "code"} for t in plan.code_tasks]
        )

        print(f"[Planner] 任务分解完成: {len(plan.research_tasks)} 个检索任务, {len(plan.code_tasks)} 个实现任务")
        print(f"[Planner] 聚焦点: {plan.paper_focus}")

    except Exception as e:
        print(f"[Planner] 结构化输出失败 ({e})，使用默认计划")
        subtasks = [
            {"type": "research", "description": f"检索 {query} 相关论文", "priority": 1},
            {"type": "research", "description": f"搜索 {query} 背景知识", "priority": 2},
            {"type": "code", "description": f"提取 {query} 核心算法", "priority": 1},
            {"type": "code", "description": f"生成代码骨架和流程图", "priority": 1},
        ]

    return {
        "subtasks": subtasks,
        "memory_context": memory_context,
        "iteration": iteration,
        "pdf_path": pdf_path,
    }
