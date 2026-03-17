"""
PaperCoder — 主入口
使用方式：
  python main.py "Attention Is All You Need"
  python main.py "arxiv:1706.03762"
  python main.py "LoRA: Low-Rank Adaptation" --pdf ./papers/lora.pdf
  python main.py "BERT" --no-judge
"""
import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# LangSmith 可观测性（如配置了就自动启用）
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "papercoder")
    print("[LangSmith] 追踪已启用")


def run(query: str, pdf_path: str = "", run_judge: bool = True):
    """
    运行 PaperCoder 完整流程

    Args:
        query: 论文标题 / arXiv ID / 研究问题
        pdf_path: 本地 PDF 路径（可选）
        run_judge: 是否运行 LLM-as-Judge 评估
    """
    from .graph import get_graph

    print(f"\n{'='*60}")
    print(f"🔬 PaperCoder 启动")
    print(f"📄 查询: {query}")
    if pdf_path:
        print(f"📁 PDF: {pdf_path}")
    print(f"{'='*60}\n")

    graph = get_graph()

    # 初始状态
    initial_state = {
        "query": query,
        "pdf_path": pdf_path or "",
        "subtasks": [],
        "memory_context": "",
        "retrieved_docs": [],
        "paper_content": "",
        "algo_description": "",
        "pseudocode": "",
        "code_draft": "",
        "diagram": "",
        "github_refs": [],
        "draft_report": "",
        "feedback": "",
        "score": 0.0,
        "iteration": 0,
        "final_output": {},
    }

    # 每次运行使用唯一 thread_id（支持 MemorySaver 断点续跑）
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 执行图
        final_state = graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    final_output = final_state.get("final_output", {})

    if not final_output:
        print("⚠️  未获得最终输出，请检查日志")
        return final_state

    # ── 打印三件套输出 ─────────────────────────────────────────────
    _print_final_output(final_output, final_state)

    # ── LLM-as-Judge 评估 ─────────────────────────────────────────
    if run_judge:
        print("\n[Judge] 正在进行质量评估...")
        try:
            from .eval.judge import evaluate_output, print_score
            github_text = "\n".join(
                r.get("content", "") for r in final_state.get("github_refs", [])
            )[:500]
            score = evaluate_output(
                query=query,
                algo_description=final_state.get("algo_description", ""),
                code_draft=final_state.get("code_draft", ""),
                diagram=final_state.get("diagram", ""),
                github_refs=github_text,
                text_review=final_output.get("text_review", ""),
            )
            print_score(score)
        except Exception as e:
            print(f"[Judge] 评估失败: {e}")

    # ── 保存输出到文件 ────────────────────────────────────────────
    _save_output(query, final_output)

    return final_output


def _print_final_output(final_output: dict, state: dict):
    """格式化打印三件套输出"""
    print("\n" + "="*60)
    print("📑 文字综述")
    print("="*60)
    print(final_output.get("text_review", "（无）"))

    print("\n" + "="*60)
    print("📊 Mermaid 流程图")
    print("="*60)
    print(final_output.get("diagram", "（无）"))

    print("\n" + "="*60)
    print("💻 代码骨架")
    print("="*60)
    code = final_output.get("code", "（无）")
    # 只打印前 80 行
    code_lines = code.split("\n")
    if len(code_lines) > 80:
        print("\n".join(code_lines[:80]))
        print(f"\n... （共 {len(code_lines)} 行，已截断显示前80行）")
    else:
        print(code)

    print("\n" + "="*60)
    score = state.get("score", 0.0)
    iterations = state.get("iteration", 0)
    print(f"✅ 完成 | Reviewer 评分: {score:.1f}/10 | 迭代次数: {iterations}")
    print("="*60)


def _save_output(query: str, final_output: dict):
    """将输出保存到 output/ 目录，生成结构清晰的 Markdown 报告"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query[:50]).strip()
    base_path = output_dir / safe_name

    text_review = final_output.get("text_review", "").strip()
    diagram = final_output.get("diagram", "").strip()
    github_refs_md = final_output.get("github_refs_md", "").strip()

    # ── 综述 Markdown ─────────────────────────────────────────────
    with open(f"{base_path}_review.md", "w", encoding="utf-8") as f:
        f.write(f"# {query}\n\n")
        f.write("---\n\n")
        f.write("## 📑 文字综述\n\n")
        f.write(text_review)
        f.write("\n\n---\n\n")
        f.write("## 📊 算法流程图\n\n")
        f.write(diagram)
        if github_refs_md:
            f.write("\n\n---\n\n")
            f.write("## 🔗 GitHub 开源实现\n\n")
            f.write(github_refs_md)
        f.write("\n")

    # ── 代码文件 ──────────────────────────────────────────────────
    with open(f"{base_path}_code.py", "w", encoding="utf-8") as f:
        f.write(final_output.get("code", ""))

    review_size = Path(f"{base_path}_review.md").stat().st_size
    print(f"\n💾 已保存：")
    print(f"   output/{safe_name}_review.md  ({review_size // 1024} KB)")
    print(f"   output/{safe_name}_code.py")


def run_survey(query: str, survey_type: str = "topic", base_paper: str = ""):
    """
    运行 Survey 调研模式

    Args:
        query: 调研主题，如 "efficient Transformer attention mechanisms"
        survey_type: "topic"（领域综述）或 "followup"（基于某论文的后续工作）
        base_paper: followup 模式下的基础论文名，如 "LLaDA"
    """
    from .graph import get_survey_graph

    mode_label = f"跟进研究 [{base_paper}]" if survey_type == "followup" else "领域综述"
    print(f"\n{'='*60}")
    print(f"📚 PaperCoder Survey 模式")
    print(f"🔍 主题: {query}")
    print(f"📋 类型: {mode_label}")
    print(f"{'='*60}\n")

    graph = get_survey_graph()

    initial_state = {
        "query": query,
        "survey_type": survey_type,
        "base_paper": base_paper,
        "pdf_path": "",
        "subtasks": [],
        "memory_context": "",
        "retrieved_docs": [],
        "paper_content": "",
        "algo_description": "",
        "pseudocode": "",
        "code_draft": "",
        "diagram": "",
        "github_refs": [],
        "draft_report": "",
        "feedback": "",
        "score": 0.0,
        "iteration": 0,
        "final_output": {},
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    final_output = final_state.get("final_output", {})
    survey_text = final_output.get("text_review", "")

    # 打印
    print(f"\n{'='*60}\n📑 调研报告\n{'='*60}")
    print(survey_text[:3000])
    if len(survey_text) > 3000:
        print(f"\n... （共 {len(survey_text)} 字符，已截断显示）")

    # 保存
    _save_survey_output(query, survey_type, base_paper, final_output)
    return final_output


def _save_survey_output(query: str, survey_type: str, base_paper: str, final_output: dict):
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    label = f"followup_{base_paper}" if survey_type == "followup" else "survey"
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query[:40]).strip()
    filename = f"{label}_{safe_name}_survey.md"
    filepath = output_dir / filename

    survey_text = final_output.get("text_review", "")
    paper_list = final_output.get("paper_list", "")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 调研报告：{query}\n\n")
        if base_paper:
            f.write(f"> 基础论文：{base_paper}\n\n")
        f.write("---\n\n")
        f.write(survey_text)
        if paper_list:
            f.write("\n\n---\n\n## 附录：发现的论文列表\n\n")
            f.write(paper_list)

    size = filepath.stat().st_size
    print(f"\n💾 已保存：output/{filename}  ({size // 1024} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="PaperCoder — 论文精读与代码化理解助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 单篇精读（默认模式）
  python -m papercoder.main "Attention Is All You Need"
  python -m papercoder.main "LoRA" --pdf ./papers/lora.pdf

  # 领域综述
  python -m papercoder.main "efficient Transformer attention" --survey

  # 基于某篇论文的跟进调研
  python -m papercoder.main "LLaDA optimization" --survey --followup "LLaDA"
  python -m papercoder.main "LoRA variants and improvements" --survey --followup "LoRA"
        """
    )
    parser.add_argument("query", help="论文标题 / 研究主题")
    parser.add_argument("--pdf", default="", help="本地 PDF 路径（单篇模式）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-Judge 评估")
    parser.add_argument("--survey", action="store_true", help="启用多论文调研综述模式")
    parser.add_argument("--followup", default="", metavar="BASE_PAPER",
                        help="跟进模式：指定基础论文名，查找其后续优化工作")
    args = parser.parse_args()

    if args.survey or args.followup:
        survey_type = "followup" if args.followup else "topic"
        run_survey(
            query=args.query,
            survey_type=survey_type,
            base_paper=args.followup,
        )
    else:
        run(
            query=args.query,
            pdf_path=args.pdf,
            run_judge=not args.no_judge,
        )


if __name__ == "__main__":
    main()
