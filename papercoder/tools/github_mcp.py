"""
GitHub 代码仓库检索工具
主路径：GitHub MCP Server（需要 npx 和 GITHUB_TOKEN）
备用路径：GitHub REST API
"""
import os
import json
import asyncio
import requests
from langchain_core.tools import tool


async def _search_via_mcp(query: str) -> str:
    """通过 GitHub MCP Server 检索"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")}
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_repositories",
                {"query": query, "per_page": 5}
            )
            raw = result.content[0].text
            # 解析并格式化
            try:
                data = json.loads(raw)
                items = data.get("items", [])
                lines = [f"[GitHub MCP 检索结果 — {len(items)} 个仓库]\n"]
                for i, repo in enumerate(items, 1):
                    lines.append(
                        f"{i}. {repo.get('full_name')}\n"
                        f"   ⭐ {repo.get('stargazers_count', 0)} | "
                        f"语言: {repo.get('language', 'N/A')} | "
                        f"更新: {repo.get('updated_at', 'N/A')[:10]}\n"
                        f"   描述: {repo.get('description', 'N/A')}\n"
                        f"   URL: {repo.get('html_url')}\n"
                    )
                return "\n".join(lines)
            except Exception:
                return f"[GitHub MCP 检索结果]\n{raw}"


def _search_via_rest_api(query: str) -> str:
    """GitHub REST API 备用检索"""
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 5,
    }
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])

        if not items:
            return f"[GitHub REST API] 未找到 '{query}' 相关仓库"

        lines = [f"[GitHub REST API 检索结果 — {len(items)} 个仓库]\n"]
        for i, repo in enumerate(items, 1):
            lines.append(
                f"{i}. {repo.get('full_name')}\n"
                f"   ⭐ {repo.get('stargazers_count', 0)} | "
                f"语言: {repo.get('language', 'N/A')} | "
                f"更新: {repo.get('updated_at', 'N/A')[:10]}\n"
                f"   描述: {repo.get('description', 'N/A')}\n"
                f"   URL: {repo.get('html_url')}\n"
            )
        return "\n".join(lines)

    except requests.RequestException as e:
        return f"[GitHub REST API] 请求失败: {e}"


@tool
def github_search_tool(query: str) -> str:
    """检索 GitHub 上论文的开源实现仓库，返回仓库名、星数、描述和链接"""
    # 优先尝试 MCP
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        try:
            result = asyncio.run(_search_via_mcp(query))
            return result
        except Exception as e:
            print(f"[GitHub] MCP 失败 ({e})，回退到 REST API")

    return _search_via_rest_api(query)
