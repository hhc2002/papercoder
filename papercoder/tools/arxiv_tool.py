"""
arXiv API 学术论文检索工具
"""
import os
import re
from langchain_core.tools import tool


def download_arxiv_pdf(query: str, save_dir: str = "") -> str:
    """
    根据标题/arXiv ID 搜索并下载 PDF，返回本地路径。
    失败时返回空字符串。
    """
    try:
        import arxiv
        import urllib.request

        if not save_dir:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "papers")
        os.makedirs(save_dir, exist_ok=True)

        # 支持直接传 arXiv ID（如 "arxiv:1706.03762" 或 "2502.09992"）
        arxiv_id_match = re.search(r"(?:arxiv[:\s]*)?([\d]{4}\.\d{4,5}(?:v\d+)?)", query, re.I)
        if arxiv_id_match:
            arxiv_id = arxiv_id_match.group(1)
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
        else:
            results = list(arxiv.Client().results(
                arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
            ))
            if not results:
                return ""
            paper = results[0]

        # 文件名：用 arxiv id，避免标题特殊字符
        aid = paper.entry_id.split("/")[-1]
        pdf_path = os.path.join(save_dir, f"{aid}.pdf")

        if os.path.exists(pdf_path):
            print(f"[arXiv] 已有缓存: {pdf_path}")
            return pdf_path

        print(f"[arXiv] 下载: {paper.title[:60]} → {pdf_path}")
        urllib.request.urlretrieve(paper.pdf_url, pdf_path)
        print(f"[arXiv] 下载完成 ({os.path.getsize(pdf_path) // 1024} KB)")
        return pdf_path

    except Exception as e:
        print(f"[arXiv] PDF 下载失败: {e}")
        return ""


@tool
def arxiv_tool(query: str) -> str:
    """通过 arXiv API 检索学术论文，返回标题、摘要、PDF链接、发表日期"""
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append({
                "source": "arxiv",
                "title": paper.title,
                "authors": [a.name for a in paper.authors[:3]],
                "summary": paper.summary[:500],
                "pdf_url": paper.pdf_url,
                "published": str(paper.published.date()),
                "arxiv_id": paper.entry_id.split("/")[-1],
            })

        if not results:
            return f"arXiv 未找到关于 '{query}' 的论文"

        output_lines = [f"[arXiv 检索结果 — {len(results)} 篇]\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"{i}. {r['title']}\n"
                f"   作者: {', '.join(r['authors'])}\n"
                f"   发表: {r['published']}\n"
                f"   摘要: {r['summary'][:300]}...\n"
                f"   PDF: {r['pdf_url']}\n"
            )
        return "\n".join(output_lines)

    except ImportError:
        return "错误：arxiv 库未安装，请运行 pip install arxiv"
    except Exception as e:
        return f"arXiv 检索失败: {e}"
