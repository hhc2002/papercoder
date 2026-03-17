"""
LangGraph 图定义
拓扑：planner → [researcher ‖ coder] → reviewer → (refine | reporter)
支持并行执行 Researcher 和 Coder 节点
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import PaperCoderState
from .nodes.planner import planner_node
from .nodes.researcher import researcher_node
from .nodes.coder import coder_node
from .nodes.reviewer import reviewer_node, should_refine
from .nodes.reporter import reporter_node
from .nodes.surveyor import surveyor_node


def build_survey_graph():
    """
    Survey 模式图（简化）
    planner → researcher → surveyor → END
    不走 Coder / Reviewer 流程
    """
    graph = StateGraph(PaperCoderState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("surveyor", surveyor_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "surveyor")
    graph.add_edge("surveyor", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    print("[Graph] Survey 图编译完成")
    return compiled


def build_graph():
    """
    构建并编译 PaperCoder 状态机

    图结构：
    ┌──────────────────────────────────────────────┐
    │                  planner                     │
    │                 /       \\                    │
    │           researcher   coder                 │
    │                 \\       /                    │
    │                 reviewer                     │
    │                /        \\                   │
    │           (refine)    (report)               │
    │              ↓            ↓                  │
    │            coder       reporter              │
    │                            ↓                 │
    │                           END                │
    └──────────────────────────────────────────────┘
    """
    graph = StateGraph(PaperCoderState)

    # ── 注册节点 ──────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reporter", reporter_node)

    # ── 设置入口节点 ──────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── 并行分叉：planner → researcher & coder ────────────────────
    # LangGraph 会自动并行执行，并等待两个节点都完成后才进入 reviewer
    graph.add_edge("planner", "researcher")
    graph.add_edge("planner", "coder")

    # ── 汇合：researcher & coder → reviewer ──────────────────────
    graph.add_edge("researcher", "reviewer")
    graph.add_edge("coder", "reviewer")

    # ── 条件边：reviewer → refine(coder) 或 report ───────────────
    graph.add_conditional_edges(
        "reviewer",
        should_refine,
        {
            "refine": "coder",    # score < 6 且 iteration < 3 → 回到 Coder
            "report": "reporter", # score ≥ 6 或达到最大迭代 → 生成报告
        }
    )

    # ── 终止 ──────────────────────────────────────────────────────
    graph.add_edge("reporter", END)

    # ── 编译：加入 MemorySaver 支持会话级断点续跑 ─────────────────
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    print("[Graph] LangGraph 状态机编译完成")
    return compiled


# 全局图实例（懒加载）
_graph = None
_survey_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_survey_graph():
    global _survey_graph
    if _survey_graph is None:
        _survey_graph = build_survey_graph()
    return _survey_graph
