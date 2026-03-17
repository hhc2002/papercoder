"""
Semantic Scholar API 学术论文检索工具
检索引用关系、相关论文、影响力指标
"""
import os
import requests
from langchain_core.tools import tool


@tool
def semantic_scholar_tool(query: str) -> str:
    """通过 Semantic Scholar API 检索论文及引用关系、影响力指标"""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        params = {
            "query": query,
            "limit": 5,
            "fields": "title,abstract,citationCount,influentialCitationCount,authors,year,openAccessPdf",
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        papers = data.get("data", [])
        if not papers:
            return f"Semantic Scholar 未找到关于 '{query}' 的论文"

        output_lines = [f"[Semantic Scholar 检索结果 — {len(papers)} 篇]\n"]
        for i, p in enumerate(papers, 1):
            authors = [a.get("name", "") for a in p.get("authors", [])[:3]]
            pdf_info = p.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url", "N/A")
            output_lines.append(
                f"{i}. {p.get('title', 'N/A')}\n"
                f"   作者: {', '.join(authors)}\n"
                f"   年份: {p.get('year', 'N/A')} | 引用数: {p.get('citationCount', 0)} | 影响力引用: {p.get('influentialCitationCount', 0)}\n"
                f"   摘要: {str(p.get('abstract', ''))[:300]}...\n"
                f"   PDF: {pdf_url}\n"
            )

        return "\n".join(output_lines)

    except requests.RequestException as e:
        return f"Semantic Scholar API 请求失败: {e}"
    except Exception as e:
        return f"Semantic Scholar 检索失败: {e}"
