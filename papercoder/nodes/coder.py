"""
Coder Node — 算法提取 + 代码生成
1. 两阶段 PDF 解析（Marker + 视觉LLM）
2. Python 代码骨架生成
3. Mermaid 流程图生成（结构化生成，稳定可靠）
4. GitHub 开源实现检索
"""
import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PaperCoderState
from ..llm_factory import get_llm, get_vision_llm
from ..tools.paper_parser import parse_paper
from ..tools.github_mcp import github_search_tool


# ── Mermaid 结构化模型 ────────────────────────────────────────────

class MermaidNode(BaseModel):
    id: str = Field(description="节点ID，只用单个大写字母，如 A、B、C")
    label: str = Field(description="节点标签文本，不含英文括号 () []")
    shape: Literal["rect", "round", "diamond"] = Field(
        default="rect",
        description="rect=矩形步骤, round=开始/结束椭圆, diamond=条件判断菱形"
    )


class MermaidEdge(BaseModel):
    from_id: str = Field(description="起始节点ID")
    to_id: str = Field(description="目标节点ID")
    label: str = Field(default="", description="边标签，条件分支用 Yes 或 No")


class MermaidDiagram(BaseModel):
    nodes: list[MermaidNode] = Field(description="所有节点，不超过12个，只保留核心步骤")
    edges: list[MermaidEdge] = Field(description="所有有向边，确保每个节点都被连接")


def _build_mermaid(diagram: MermaidDiagram) -> str:
    """从结构化数据构造合法 Mermaid 字符串，完全程序化，无语法错误风险"""
    lines = ["flowchart TD"]
    for node in diagram.nodes:
        label = node.label.replace('"', "'")
        if node.shape == "diamond":
            lines.append(f'    {node.id}{{"{label}"}}')
        elif node.shape == "round":
            lines.append(f'    {node.id}(["{label}"])')
        else:
            lines.append(f'    {node.id}["{label}"]')
    for edge in diagram.edges:
        if edge.label:
            lines.append(f'    {edge.from_id} -->|{edge.label}| {edge.to_id}')
        else:
            lines.append(f'    {edge.from_id} --> {edge.to_id}')
    return "```mermaid\n" + "\n".join(lines) + "\n```"


def _extract_mermaid_block(raw: str) -> str:
    """兜底：从 LLM 自由文本中提取 mermaid 代码块，修复常见语法问题"""
    match = re.search(r'```mermaid\s*(.*?)\s*```', raw, re.DOTALL)
    content = match.group(1).strip() if match else raw.strip()
    if not content.startswith("flowchart"):
        content = "flowchart TD\n" + content
    # 修复：节点标签含英文括号时加引号，如 A(text) → A["text"]
    content = re.sub(
        r'\b([A-Z])\(([^)"{}\[\]]+)\)',
        lambda m: f'{m.group(1)}["{m.group(2)}"]',
        content
    )
    return "```mermaid\n" + content + "\n```"


CODE_GEN_SYSTEM = """你是一位资深算法工程师，专门将学术论文算法转化为可运行代码。

代码生成要求：
1. 保留完整的函数签名、类定义和数据结构
2. 核心算法逻辑处用详细注释说明意图，标注 TODO（需实现）
3. 包含完整的类型注解（Python 3.10+）
4. 函数开头写 docstring，说明对应论文的哪一部分
5. 代码结构完整，可作为实现起点直接运行
6. 依赖库用 # pip install xxx 注释标注

重要：只输出纯 Python 代码，不要有任何 markdown 代码块标记（不要 ```python 或 ```），不要有任何解释性前言或总结文字。"""

CODE_GEN_HUMAN = """论文标题/主题：{query}

算法描述：
{algo_description}

伪代码：
{pseudocode}

检索到的相关信息：
{research_context}

{feedback_section}

请生成完整的 Python 代码骨架。"""

DIAGRAM_STRUCTURED_SYSTEM = """你是一位技术可视化工程师。根据算法描述提取核心步骤，输出用于生成 Mermaid 流程图的结构化数据。

规则：
- id 只用单个大写字母（A、B、C...），不超过12个节点
- label 不含英文括号 () []，中英文均可，尽量简洁（8字以内）
- shape: rect=普通步骤, round=开始/结束, diamond=条件判断
- 所有节点必须被至少一条边连接
- 条件节点（diamond）的出边必须有 Yes/No label"""

DIAGRAM_HUMAN = """根据以下算法描述，提取核心流程节点和边（不超过12个节点）：

算法描述：{algo_description}

伪代码：{pseudocode}

只保留主干流程，忽略细节。"""

ALGO_EXTRACT_SYSTEM = """你是一位论文算法分析专家。请从以下论文文本中提取核心算法信息。

提取要点：
1. 算法名称和核心思想（一段话，100-200字）
2. 主要步骤（按顺序列举，每步50字内）
3. 关键公式和符号
4. 输入输出定义"""

ALGO_EXTRACT_HUMAN = """论文标题/主题：{query}

论文内容（节选）：
{paper_content}

检索补充信息：
{research_context}

请提取算法描述和伪代码。格式：
[算法描述]
...

[伪代码]
..."""


def coder_node(state: PaperCoderState) -> dict:
    """
    Coder Node
    1. 解析 PDF（如有）或基于检索信息提取算法
    2. 生成代码骨架
    3. 生成 Mermaid 流程图
    4. 检索 GitHub 开源实现
    """
    query = state.get("query", "")
    pdf_path = state.get("pdf_path", "")
    retrieved_docs = state.get("retrieved_docs", [])
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)

    print(f"\n[Coder] 开始代码化处理 | 迭代: {iteration}")

    llm = get_llm()
    vision_llm = get_vision_llm()  # 多模态模型（可能与 llm 相同）

    # 整合检索上下文
    research_context = "\n\n".join(
        d.get("content", "") for d in retrieved_docs
    )[:3000]

    # ── Step 1: PDF 解析 or 文本提取算法 ──────────────────────────
    if pdf_path:
        print(f"[Coder] 解析 PDF: {pdf_path}")
        parsed = parse_paper(pdf_path, vision_llm)
        paper_content = parsed["full_text"]
        algo_description = parsed["algo_description"]
        pseudocode = parsed["pseudocode"]
    else:
        paper_content = ""
        algo_description = ""
        pseudocode = ""

    # 如果 PDF 解析未获得足够算法信息，用 LLM 从文本/检索结果中提取
    if not algo_description or len(algo_description) < 100:
        print("[Coder] 从文本提取算法描述...")
        content_for_extract = paper_content[:4000] if paper_content else research_context[:4000]
        try:
            extract_response = llm.invoke([
                SystemMessage(content=ALGO_EXTRACT_SYSTEM),
                HumanMessage(content=ALGO_EXTRACT_HUMAN.format(
                    query=query,
                    paper_content=content_for_extract,
                    research_context=research_context[:2000],
                )),
            ])
            raw = extract_response.content
            if "[算法描述]" in raw and "[伪代码]" in raw:
                algo_description = raw.split("[伪代码]")[0].replace("[算法描述]", "").strip()
                pseudocode = raw.split("[伪代码]")[1].strip()
            else:
                algo_description = raw[:1000]
                pseudocode = raw[1000:] if len(raw) > 1000 else ""
        except Exception as e:
            print(f"[Coder] 算法提取失败: {e}")
            algo_description = f"关于 {query} 的算法"
            pseudocode = "（算法提取失败，请提供 PDF 文件）"

    feedback_section = f"Reviewer 反馈（请针对性改进）：\n{feedback}" if feedback else ""

    # ── Step 2: 生成代码骨架 ─────────────────────────────────────
    print("[Coder] 生成代码骨架...")
    try:
        # 代码生成需要更大的输出窗口，避免截断
        code_llm = llm.bind(max_output_tokens=16384)
        code_response = code_llm.invoke([
            SystemMessage(content=CODE_GEN_SYSTEM),
            HumanMessage(content=CODE_GEN_HUMAN.format(
                query=query,
                algo_description=algo_description,
                pseudocode=pseudocode,
                research_context=research_context[:2000],
                feedback_section=feedback_section,
            )),
        ])
        code_draft = code_response.content
    except Exception as e:
        print(f"[Coder] 代码生成失败: {e}")
        code_draft = f"# 代码生成失败: {e}\n# 请检查 LLM 配置"

    # ── Step 3: 生成 Mermaid 流程图（结构化生成）────────────────────
    print("[Coder] 生成 Mermaid 流程图（结构化模式）...")
    try:
        structured_llm = llm.with_structured_output(MermaidDiagram)
        diagram_data: MermaidDiagram = structured_llm.invoke([
            SystemMessage(content=DIAGRAM_STRUCTURED_SYSTEM),
            HumanMessage(content=DIAGRAM_HUMAN.format(
                algo_description=algo_description,
                pseudocode=pseudocode,
            )),
        ])
        diagram = _build_mermaid(diagram_data)
        print(f"[Coder] 结构化流程图生成成功，{len(diagram_data.nodes)} 个节点")
    except Exception as e:
        # 降级：让 LLM 直接输出文本，再提取修复
        print(f"[Coder] 结构化流程图失败 ({e})，降级到文本提取...")
        try:
            diagram_response = llm.invoke([
                SystemMessage(content="只输出一个 ```mermaid 代码块，使用 flowchart TD，节点ID只用单个大写字母，不超过10个节点，所有节点必须有连线。"),
                HumanMessage(content=DIAGRAM_HUMAN.format(
                    algo_description=algo_description,
                    pseudocode=pseudocode,
                )),
            ])
            diagram = _extract_mermaid_block(diagram_response.content)
        except Exception as e2:
            print(f"[Coder] 流程图生成失败: {e2}")
            diagram = '```mermaid\nflowchart TD\n    A(["开始"]) --> B["算法执行"] --> C(["结束"])\n```'

    # ── Step 4: GitHub 开源实现检索 ──────────────────────────────
    print("[Coder] 检索 GitHub 开源实现...")
    try:
        github_result = github_search_tool.invoke(f"{query} implementation paper")
        github_refs = [{"source": "github", "content": github_result}]
    except Exception as e:
        print(f"[Coder] GitHub 检索失败: {e}")
        github_refs = [{"source": "github", "content": f"检索失败: {e}"}]

    print(f"[Coder] 完成 | 代码: {len(code_draft)} 字符 | 流程图: {len(diagram)} 字符")

    return {
        "paper_content": paper_content,
        "algo_description": algo_description,
        "pseudocode": pseudocode,
        "code_draft": code_draft,
        "diagram": diagram,
        "github_refs": github_refs,
    }
