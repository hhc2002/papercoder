"""
Tavily 网络搜索工具
用于检索背景知识、技术博客、相关工作介绍
"""
import os
from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """使用 Tavily 搜索互联网，获取技术背景知识和相关博客"""
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return "错误：未设置 TAVILY_API_KEY 环境变量"

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        output_lines = ["[Tavily 网络搜索结果]\n"]

        if response.get("answer"):
            output_lines.append(f"AI 摘要答案:\n{response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            output_lines.append(
                f"{i}. {result.get('title', 'N/A')}\n"
                f"   URL: {result.get('url', 'N/A')}\n"
                f"   内容: {result.get('content', '')[:300]}...\n"
            )

        return "\n".join(output_lines)

    except ImportError:
        # 降级：使用 langchain_community TavilySearchResults
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(max_results=5)
            results = search.run(query)
            return f"[网络搜索结果]\n{results}"
        except Exception as e2:
            return f"网络搜索失败（Tavily 未安装或未配置）: {e2}"
    except Exception as e:
        return f"Tavily 搜索失败: {e}"
