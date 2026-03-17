"""
LLM 工厂模块
支持 Anthropic / OpenAI / Google Gemini
通过环境变量 MODEL_PROVIDER 控制
"""
import os
from functools import lru_cache

ANTHROPIC = "anthropic"
OPENAI = "openai"
GEMINI = "gemini"


def _try_gemini(temperature: float, max_tokens: int):
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    print(f"[LLM] 使用 Google Gemini {model}")
    return llm


def _try_anthropic(temperature: float, max_tokens: int):
    from langchain_anthropic import ChatAnthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    llm = ChatAnthropic(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    print(f"[LLM] 使用 Anthropic {model}")
    return llm


def _try_openai(temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    print(f"[LLM] 使用 OpenAI {model}")
    return llm


@lru_cache(maxsize=1)
def get_llm():
    """获取主 LLM 实例（文本生成）"""
    provider = os.getenv("MODEL_PROVIDER", GEMINI).lower()

    # 按 provider 决定尝试顺序
    order = {
        GEMINI:    [_try_gemini, _try_anthropic, _try_openai],
        ANTHROPIC: [_try_anthropic, _try_gemini, _try_openai],
        OPENAI:    [_try_openai, _try_gemini, _try_anthropic],
    }.get(provider, [_try_gemini, _try_anthropic, _try_openai])

    errors = []
    for fn in order:
        try:
            return fn(temperature=0.1, max_tokens=16384)
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")

    raise RuntimeError(
        "LLM 初始化失败，已尝试所有提供商。\n"
        "请在 .env 中设置以下任意一个：\n"
        "  GOOGLE_API_KEY（Gemini）\n"
        "  ANTHROPIC_API_KEY（Claude）\n"
        "  OPENAI_API_KEY（GPT）\n\n"
        "错误详情：\n" + "\n".join(errors)
    )


@lru_cache(maxsize=1)
def get_vision_llm():
    """
    获取视觉 LLM（用于 PDF 算法页图片理解）
    Gemini / Claude / GPT-4o 均原生支持多模态
    """
    provider = os.getenv("MODEL_PROVIDER", GEMINI).lower()

    order = {
        GEMINI:    [_try_gemini, _try_anthropic, _try_openai],
        ANTHROPIC: [_try_anthropic, _try_gemini, _try_openai],
        OPENAI:    [_try_openai, _try_gemini, _try_anthropic],
    }.get(provider, [_try_gemini, _try_anthropic, _try_openai])

    for fn in order:
        try:
            return fn(temperature=0, max_tokens=4096)
        except Exception:
            pass

    return None  # 视觉模型可选，无则 paper_parser 降级到文本提取
