"""
Reviewer Node — Self-Refine 模式
从三个维度评分：算法理解准确性、代码可运行性、流程图逻辑正确性
score < 6 → 携带 feedback 退回（最多3次循环，Circuit Breaker）
score ≥ 6 → 进入 Reporter
"""
import json
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import PaperCoderState
from ..llm_factory import get_llm


class ReviewResult(BaseModel):
    algo_accuracy: float = Field(description="算法理解准确性 0-10", ge=0, le=10)
    code_quality: float = Field(description="代码骨架质量（结构完整性、注释清晰度）0-10", ge=0, le=10)
    diagram_logic: float = Field(description="流程图逻辑正确性 0-10", ge=0, le=10)
    overall: float = Field(description="综合评分 0-10", ge=0, le=10)
    feedback: str = Field(description="详细改进建议，指出具体不足之处")
    pass_review: bool = Field(description="是否通过评审（overall >= 6）")


REVIEWER_SYSTEM = """你是一位严格的技术评审专家，负责评估论文精读与代码化输出的质量。

评分维度（各0-10分）：
1. **算法理解准确性**：提取的算法描述是否忠实于原论文意图，关键步骤是否完整
2. **代码骨架质量**：代码结构是否合理、类型注解是否完整、TODO 标注是否清晰，能否作为实现起点
3. **流程图逻辑性**：Mermaid 流程图是否准确反映算法执行流程，节点和边是否合理

综合评分 = 各维度加权平均（准确性×0.40 + 代码×0.35 + 流程图×0.25）

评分标准：
- 8-10分：优秀，可直接使用
- 6-7分：合格，有小问题
- 4-5分：需改进，有明显缺陷
- 0-3分：不合格，需重做

pass_review = (overall >= 6.0)"""

REVIEWER_HUMAN = """论文主题：{query}

== 算法描述 ==
{algo_description}

== 伪代码提取 ==
{pseudocode}

== 生成的代码骨架 ==
{code_draft}

== Mermaid 流程图 ==
{diagram}

请严格评审以上内容，输出结构化评分和反馈。"""


def reviewer_node(state: PaperCoderState) -> dict:
    """
    Reviewer Node（Self-Refine）
    评估输出质量，决定是否需要重新迭代
    """
    query = state.get("query", "")
    algo_description = state.get("algo_description", "")
    pseudocode = state.get("pseudocode", "")
    code_draft = state.get("code_draft", "")
    diagram = state.get("diagram", "")
    iteration = state.get("iteration", 0)

    print(f"\n[Reviewer] 开始评审 | 迭代: {iteration}")

    llm = get_llm()

    try:
        structured_llm = llm.with_structured_output(ReviewResult)
        review: ReviewResult = structured_llm.invoke([
            SystemMessage(content=REVIEWER_SYSTEM),
            HumanMessage(content=REVIEWER_HUMAN.format(
                query=query,
                algo_description=algo_description[:1500],
                pseudocode=pseudocode[:1000],
                code_draft=code_draft[:2000],
                diagram=diagram[:800],
            )),
        ])

        score = review.overall
        feedback = review.feedback
        passed = review.pass_review

        print(
            f"[Reviewer] 评分结果:\n"
            f"  算法准确性: {review.algo_accuracy:.1f}\n"
            f"  代码质量:   {review.code_quality:.1f}\n"
            f"  流程图逻辑: {review.diagram_logic:.1f}\n"
            f"  综合评分:   {score:.1f} | 通过: {passed}"
        )
        print(f"[Reviewer] 反馈: {feedback[:200]}...")

    except Exception as e:
        print(f"[Reviewer] 结构化评审失败 ({e})，使用备用评分")
        # 备用：简单打分
        score = 6.0 if code_draft and diagram else 4.0
        feedback = f"评审系统异常: {e}。请检查代码骨架和流程图质量。"
        passed = score >= 6.0

    # 更新草稿报告（Reviewer 生成的摘要）
    draft_report = (
        f"论文：{query}\n"
        f"评审轮次：{iteration + 1}\n"
        f"综合评分：{score:.1f}/10\n"
        f"评审意见：{feedback}"
    )

    return {
        "score": score,
        "feedback": feedback,
        "draft_report": draft_report,
        "iteration": iteration + 1,
    }


def should_refine(state: PaperCoderState) -> str:
    """
    条件边路由函数
    以下任一情况直接进入 Reporter：
    - score >= 6.0
    - 已达最大迭代次数
    - 没有 PDF 且 score < 4.0（无源材料，重跑也改不了）
    """
    score = state.get("score", 0.0)
    iteration = state.get("iteration", 0)
    pdf_path = state.get("pdf_path", "")
    MAX_ITERATIONS = 3

    # 没有 PDF 且分数很低：重跑 Coder 无法改善，直接出报告
    if not pdf_path and score < 4.0 and iteration >= 1:
        print(f"[Router] 无 PDF 且评分持续低位 ({score:.1f})，跳过重跑直接生成报告")
        return "report"

    if score < 6.0 and iteration < MAX_ITERATIONS:
        print(f"[Router] 评分 {score:.1f} < 6.0，进入第 {iteration + 1} 轮改进")
        return "refine"

    if score >= 6.0:
        print(f"[Router] 评分 {score:.1f} >= 6.0，进入 Reporter")
    else:
        print(f"[Router] 已达最大迭代次数 {MAX_ITERATIONS}，强制进入 Reporter（评分: {score:.1f}）")
    return "report"
