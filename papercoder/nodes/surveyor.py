"""
Surveyor Node — 多论文调研综述模式
输入：研究主题 或 基础论文
输出：对比表格 + 各论文摘要卡 + 领域综述
"""
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PaperCoderState
from ..llm_factory import get_llm
from ..tools.arxiv_tool import arxiv_tool
from ..tools.semantic_scholar import semantic_scholar_tool
from ..memory.long_term import get_memory


# ── Step 1: 发现相关论文 ──────────────────────────────────────────

DISCOVER_SYSTEM = """你是一位学术调研专家。根据用户的研究主题，制定检索策略，找出最相关的论文列表。"""

DISCOVER_HUMAN = """研究主题：{query}
调研类型：{survey_type}

检索结果：
{search_results}

请从检索结果中挑选出最相关的 5-8 篇论文，输出结构化列表。

每篇格式：
### [序号]. 论文标题
- **年份**: xxxx
- **一句话贡献**: ...
- **与主题的关联**: ...
- **PDF/链接**: ...（如有）"""


# ── Step 2: 逐篇分析 ─────────────────────────────────────────────

ANALYZE_SYSTEM = """你是一位论文分析专家，对每篇论文进行简洁的结构化分析，严格控制在 300 字以内。"""

ANALYZE_HUMAN = """论文信息：
{paper_info}

研究主题：{query}

请输出：
**核心方法**：（1-2句）
**主要创新**：（1-2句）
**实验结果**：（关键指标，1句）
**局限性**：（1句）
**与主题关联**：（1句）"""


# ── Step 3: 生成综述报告 ──────────────────────────────────────────

SURVEY_SYSTEM = """你是一位资深学术综述作者，生成结构严谨、内容详实的调研报告，使用 Markdown 格式。

**格式要求（必须严格遵守）：**
- 表格单元格内容简洁，每格不超过 20 字，禁止空格填充对齐
- 论文标题在表格中可缩写（保留关键词即可）
- 总字数 2500-4000 字，各节均匀分布"""

SURVEY_HUMAN = """研究主题：{query}
调研类型：{survey_type}

已发现的论文列表：
{paper_list}

各论文分析摘要：
{paper_analyses}

历史研究记忆：
{memory_context}

请生成完整的调研综述报告，严格按以下结构，表格单元格必须简洁（每格≤20字）：

## 调研背景
（该领域的核心问题和研究动机，100-150字）

## 论文概览
（表格：标题缩写 | 年份 | 核心方法(≤15字) | 主要贡献(≤15字) | 关键指标）

## 各论文详析
（每篇 200-300 字，含方法、创新点、实验结论）

## 横向对比分析
（从方法论、性能、适用场景等维度对比，含简洁对比表格）

## 研究趋势与发展脉络
（按时间线梳理技术演进，100-200字）

## 开放问题与未来方向
（3-5个尚未解决的核心问题）

## 推荐阅读顺序
（给研究者的阅读建议）"""


def surveyor_node(state: PaperCoderState) -> dict:
    """
    Surveyor Node — 多论文调研模式
    不生成代码，专注于文献发现和横向对比分析
    """
    query = state.get("query", "")
    survey_type = state.get("survey_type", "topic")  # "topic" | "followup"
    base_paper = state.get("base_paper", "")
    memory_context = state.get("memory_context", "")
    retrieved_docs = state.get("retrieved_docs", [])

    print(f"\n[Surveyor] 开始多论文调研 | 主题: {query[:60]} | 类型: {survey_type}")

    llm = get_llm()

    # ── Step 1: 检索论文 ──────────────────────────────────────────
    search_results = []

    # 用已有检索结果
    existing = "\n\n".join(d.get("content", "") for d in retrieved_docs)[:3000]
    if existing:
        search_results.append(existing)

    # 针对 followup 模式：用 Semantic Scholar 找引用论文
    if survey_type == "followup" and base_paper:
        print(f"[Surveyor] 查找 '{base_paper}' 的后续工作...")
        try:
            s2_result = semantic_scholar_tool.invoke(f"{base_paper} optimization improvement")
            search_results.append(s2_result)
        except Exception as e:
            print(f"[Surveyor] S2 检索失败: {e}")

    # 补充 arXiv 检索
    try:
        arxiv_result = arxiv_tool.invoke(query)
        search_results.append(arxiv_result[:2000])
    except Exception as e:
        print(f"[Surveyor] arXiv 检索失败: {e}")

    all_search = "\n\n---\n\n".join(search_results)[:5000]

    # ── Step 2: 发现并整理论文列表 ───────────────────────────────
    print("[Surveyor] 整理论文列表...")
    try:
        discover_resp = llm.bind(max_output_tokens=2048).invoke([
            SystemMessage(content=DISCOVER_SYSTEM),
            HumanMessage(content=DISCOVER_HUMAN.format(
                query=query,
                survey_type=survey_type,
                search_results=all_search,
            )),
        ])
        paper_list = discover_resp.content
    except Exception as e:
        print(f"[Surveyor] 论文列表生成失败: {e}")
        paper_list = all_search[:2000]

    # ── Step 3: 逐篇分析 ─────────────────────────────────────────
    print("[Surveyor] 逐篇分析...")
    try:
        analyze_resp = llm.bind(max_output_tokens=2048).invoke([
            SystemMessage(content=ANALYZE_SYSTEM),
            HumanMessage(content=ANALYZE_HUMAN.format(
                paper_info=paper_list,
                query=query,
            )),
        ])
        paper_analyses = analyze_resp.content
    except Exception as e:
        print(f"[Surveyor] 逐篇分析失败: {e}")
        paper_analyses = "（分析失败）"

    # ── Step 4: 生成完整综述 ──────────────────────────────────────
    print("[Surveyor] 生成综述报告...")
    try:
        survey_resp = llm.bind(max_output_tokens=4096).invoke([
            SystemMessage(content=SURVEY_SYSTEM),
            HumanMessage(content=SURVEY_HUMAN.format(
                query=query,
                survey_type="跟进研究/优化工作" if survey_type == "followup" else "领域综述",
                paper_list=paper_list[:2000],
                paper_analyses=paper_analyses[:2000],
                memory_context=memory_context or "无",
            )),
        ])
        survey_report = survey_resp.content
        # 只有真的超长才截断，正常报告不应触发
        if len(survey_report) > 40000:
            survey_report = survey_report[:40000] + "\n\n*（报告已截断）*"
    except Exception as e:
        print(f"[Surveyor] 综述生成失败: {e}")
        survey_report = f"综述生成失败: {e}"

    # ── 写入长期记忆 ──────────────────────────────────────────────
    try:
        memory = get_memory()
        memory.save(
            query=query,
            summary=survey_report[:400],
            code_snippet="",
        )
        print("[Surveyor] 已写入长期记忆")
    except Exception as e:
        print(f"[Surveyor] 记忆写入失败: {e}")

    print(f"[Surveyor] 完成 | 综述: {len(survey_report)} 字符")

    final_output = {
        "text_review": survey_report,
        "diagram": "",
        "code": "",
        "paper_list": paper_list,
        "paper_analyses": paper_analyses,
        "metadata": {"query": query, "survey_type": survey_type},
    }

    return {"final_output": final_output}
