"""
LLM-as-Judge 评估模块
对最终输出进行自动化质量评估
支持单次评估和消融实验
"""
import json
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ..llm_factory import get_llm


class JudgeScore(BaseModel):
    algo_accuracy: float = Field(description="算法理解准确性 0-10", ge=0, le=10)
    code_quality: float = Field(description="代码骨架质量 0-10", ge=0, le=10)
    diagram_logic: float = Field(description="流程图逻辑性 0-10", ge=0, le=10)
    github_relevance: float = Field(description="GitHub 引用相关性 0-10", ge=0, le=10)
    overall: float = Field(description="综合评分 0-10", ge=0, le=10)
    feedback: str = Field(description="详细改进建议")
    strengths: str = Field(description="主要优点")


JUDGE_SYSTEM = """你是一位客观公正的技术评审专家，负责评估论文精读系统的输出质量。
评分要客观、可重复，基于具体证据而非主观印象。"""

JUDGE_PROMPT = """评估以下论文精读结果的质量（各维度0-10分）：

评估维度：
1. **算法理解准确性**（0-10）：提取的算法描述是否忠实于原论文意图
2. **代码骨架质量**（0-10）：代码结构是否合理，能否作为实现起点
3. **流程图逻辑性**（0-10）：Mermaid 流程图是否准确反映算法流程
4. **GitHub 引用相关性**（0-10）：找到的开源实现是否与论文相关

论文标题：{query}
算法描述：{algo_description}
代码骨架（前1500字）：{code_draft}
Mermaid 图：{diagram}
GitHub 引用：{github_refs}
文字综述（前500字）：{text_review}

请给出详细评分和反馈。"""


def evaluate_output(
    query: str,
    algo_description: str,
    code_draft: str,
    diagram: str,
    github_refs: str,
    text_review: str,
) -> JudgeScore:
    """对单次输出进行 LLM-as-Judge 评估"""
    llm = get_llm()

    try:
        structured_llm = llm.with_structured_output(JudgeScore)
        score: JudgeScore = structured_llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=JUDGE_PROMPT.format(
                query=query,
                algo_description=algo_description[:1000],
                code_draft=code_draft[:1500],
                diagram=diagram[:600],
                github_refs=github_refs[:500],
                text_review=text_review[:500],
            )),
        ])
        return score
    except Exception as e:
        print(f"[Judge] 评估失败: {e}")
        return JudgeScore(
            algo_accuracy=0, code_quality=0, diagram_logic=0,
            github_relevance=0, overall=0,
            feedback=f"评估失败: {e}", strengths=""
        )


def print_score(score: JudgeScore):
    """格式化打印评分结果"""
    print("\n" + "="*50)
    print("📊 LLM-as-Judge 评估报告")
    print("="*50)
    print(f"算法理解准确性:  {score.algo_accuracy:.1f}/10")
    print(f"代码骨架质量:    {score.code_quality:.1f}/10")
    print(f"流程图逻辑性:    {score.diagram_logic:.1f}/10")
    print(f"GitHub 引用相关: {score.github_relevance:.1f}/10")
    print(f"综合评分:        {score.overall:.1f}/10")
    print(f"\n优点: {score.strengths}")
    print(f"改进建议: {score.feedback}")
    print("="*50)
