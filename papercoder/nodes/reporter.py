"""
Reporter Node — 生成最终三件套输出
1. 文字综述（论文核心贡献、方法论、相关工作对比）
2. Mermaid 流程图（直接使用 Coder 生成的结果）
3. 代码骨架（带 GitHub 开源对比）
同时将本次研究写入 FAISS 长期记忆
"""
import re
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PaperCoderState
from ..llm_factory import get_llm
from ..memory.long_term import get_memory


REPORTER_SYSTEM = """你是一位技术写作专家，输出详尽、结构化的论文精读报告，使用 Markdown 格式。

固定结构（每节充分展开，总计 1500-2500 字）：

## 核心贡献
（4-5个要点，每点2-3句，说清楚创新在哪、解决了什么问题）

## 方法论详解
（详细说明算法核心思路、关键设计决策，含关键公式，300字以上）

## 与现有方法对比
（Markdown 表格，对比3-4个维度，附1-2段文字分析）

## 实现要点
（代码实现时的5个关键注意事项，每点2-3句）

## 局限性与未来方向
（2-3点）

## 延伸阅读
（3-5个推荐资源，含论文和博客）

禁止复制粘贴原始检索内容、URL列表或重复信息。"""

REPORTER_HUMAN = """论文主题：{query}

检索摘要：
{retrieved_docs}

算法描述：
{algo_description}

GitHub 开源实现：
{github_refs}

Reviewer评分：{score}/10 | 意见：{feedback}

请生成结构化综述报告，充分展开每个章节。"""


def reporter_node(state: PaperCoderState) -> dict:
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    algo_description = state.get("algo_description", "")
    pseudocode = state.get("pseudocode", "")
    code_draft = state.get("code_draft", "")
    diagram = state.get("diagram", "")
    github_refs = state.get("github_refs", [])
    score = state.get("score", 0.0)
    feedback = state.get("feedback", "")

    print(f"\n[Reporter] 生成最终报告 | 综合评分: {score:.1f}")

    llm = get_llm()

    research_text = "\n\n".join(d.get("content", "") for d in retrieved_docs)[:3000]
    github_text = "\n".join(r.get("content", "") for r in github_refs)[:800]

    # ── 生成文字综述 ──────────────────────────────────────────────
    try:
        bounded_llm = llm.bind(max_output_tokens=6144)
        review_response = bounded_llm.invoke([
            SystemMessage(content=REPORTER_SYSTEM),
            HumanMessage(content=REPORTER_HUMAN.format(
                query=query,
                retrieved_docs=research_text,
                algo_description=algo_description[:2000],
                github_refs=github_text,
                score=f"{score:.1f}",
                feedback=feedback[:500],
            )),
        ])
        text_review = review_response.content
    except Exception as e:
        print(f"[Reporter] 综述生成失败: {e}")
        text_review = f"# {query}\n\n综述生成失败: {e}"

    # ── 整理 GitHub 引用（格式化为 Markdown）─────────────────────
    github_md = _format_github_refs(github_refs)

    # ── 整理代码（提取纯 Python，去除 markdown 包裹）──────────────
    code_output = f"# PaperCoder — {query}\n\n{_extract_python_code(code_draft)}"

    # ── 构建最终三件套 ────────────────────────────────────────────
    final_output = {
        "text_review": text_review,
        "diagram": diagram,
        "code": code_output,
        "github_refs_md": github_md,
        "metadata": {
            "query": query,
            "score": score,
        }
    }

    # ── 写入长期记忆 ──────────────────────────────────────────────
    try:
        memory = get_memory()
        memory.save(
            query=query,
            summary=text_review[:400],
            code_snippet=code_draft[:800],
        )
        print("[Reporter] 已写入长期记忆")
    except Exception as e:
        print(f"[Reporter] 长期记忆写入失败: {e}")

    print(f"[Reporter] 完成 | 综述: {len(text_review)} 字符")
    print(f"\n{'='*60}\n✅ PaperCoder 分析完成！\n{'='*60}")

    return {"final_output": final_output}


def _extract_python_code(code_draft: str) -> str:
    """从 LLM 响应中提取纯 Python 代码，去除 markdown 代码块标记和前言文字"""
    # 优先提取 ```python ... ``` 代码块
    match = re.search(r'```python\s*(.*?)\s*```', code_draft, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 其次提取 ``` ... ``` 代码块（无语言标记）
    match = re.search(r'```\s*(.*?)\s*```', code_draft, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 无完整代码块：跳过开头非代码行，去除结尾残留的 ```
    lines = code_draft.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "def ", "class ", "#", "@")):
            start = i
            break
    result_lines = lines[start:]
    # 去除结尾的 ``` 行
    while result_lines and result_lines[-1].strip() in ("```", "```python", "```py"):
        result_lines.pop()
    return "\n".join(result_lines).strip()


def _format_github_refs(github_refs: list) -> str:
    """将 GitHub 检索结果格式化为 Markdown 列表"""
    content = "\n".join(r.get("content", "") for r in github_refs)
    lines = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(("1.", "2.", "3.", "4.", "5.")):
            lines.append(f"\n### {line}")
        elif line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
            lines.append(f"- 链接：{url}")
        elif line.startswith("⭐"):
            lines.append(f"- {line}")
        elif line.startswith("描述:"):
            lines.append(f"- {line}")
        elif line and not line.startswith("[GitHub"):
            lines.append(line)
    return "\n".join(lines)[:2000]
