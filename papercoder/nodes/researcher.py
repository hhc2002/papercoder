"""
Researcher Node — 四路学术检索
1. arXiv API 论文检索
2. Semantic Scholar 引用关系
3. Tavily 网络搜索（背景知识）
4. 本地 RAG（LlamaIndex + ChromaDB）
"""
from langchain_core.messages import SystemMessage, HumanMessage
try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import PaperCoderState
from ..llm_factory import get_llm
from ..tools.arxiv_tool import arxiv_tool
from ..tools.semantic_scholar import semantic_scholar_tool
from ..tools.web_search import web_search_tool
from ..tools.local_rag import local_rag_tool


RESEARCHER_SYSTEM = """你是一位专业的学术文献研究员。

使用工具检索后，输出一份**简洁的结构化摘要**，严格控制在 800 字以内，包含：
1. 核心论文信息（标题、年份、核心贡献 2-3 句）
2. 相关工作（最多 3 篇，每篇 1 句）
3. 技术背景（2-3 句概括）
4. 重要引用/链接（最多 3 条）

禁止将原始搜索结果、URL 列表、摘要全文直接粘贴到输出中。"""

COMPRESS_SYSTEM = """你是一位信息压缩专家。将以下检索结果压缩为结构化摘要，严格控制在 600 字以内。

格式：
## 核心论文
（标题、年份、1-2句核心贡献）

## 相关工作
（最多3篇，每篇一行）

## 技术背景
（2-3句）

## 关键资源
（最多3条链接）

禁止包含原始数据、重复内容。"""


def researcher_node(state: PaperCoderState) -> dict:
    query = state.get("query", "")
    subtasks = state.get("subtasks", [])
    feedback = state.get("feedback", "")

    research_tasks = [t for t in subtasks if t.get("type") == "research"]
    print(f"\n[Researcher] 开始检索 | {len(research_tasks)} 个检索任务")

    task_descriptions = "\n".join(
        f"- [{t['priority']}] {t['description']}" for t in research_tasks
    ) if research_tasks else f"- 检索关于 '{query}' 的相关论文和背景知识"

    human_prompt = (
        f"研究目标：{query}\n\n"
        f"检索任务：\n{task_descriptions}\n\n"
        f"{f'Reviewer反馈：{feedback}' if feedback else ''}\n\n"
        "使用工具检索，最终输出简洁结构化摘要（800字以内）。"
    )

    tools = [arxiv_tool, semantic_scholar_tool, web_search_tool, local_rag_tool]
    llm = get_llm()

    raw_output = ""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESEARCHER_SYSTEM),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
        result = executor.invoke({"input": human_prompt})
        raw_output = result.get("output", "")

    except Exception as e:
        print(f"[Researcher] Agent 失败 ({e})，直接调用工具")
        parts = []
        for tool_fn, label in [
            (arxiv_tool, "arXiv"),
            (semantic_scholar_tool, "S2"),
            (web_search_tool, "Web"),
        ]:
            try:
                parts.append(tool_fn.invoke(query))
            except Exception as e2:
                parts.append(f"[{label} 失败: {e2}]")
        raw_output = "\n\n".join(parts)

    # ── 压缩：确保不超过 1200 字 ──────────────────────────────────
    if len(raw_output) > 1200:
        print(f"[Researcher] 输出过长({len(raw_output)}字)，压缩中...")
        try:
            compress_resp = llm.invoke([
                SystemMessage(content=COMPRESS_SYSTEM),
                HumanMessage(content=f"论文主题：{query}\n\n原始检索内容：\n{raw_output[:6000]}"),
            ])
            research_summary = compress_resp.content[:1200]
        except Exception as e:
            print(f"[Researcher] 压缩失败: {e}")
            research_summary = raw_output[:1200]
    else:
        research_summary = raw_output

    print(f"[Researcher] 完成，摘要长度: {len(research_summary)} 字符")
    return {"retrieved_docs": [{"source": "researcher", "content": research_summary}]}
