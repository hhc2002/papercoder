"""
两阶段论文 PDF 解析
第一阶段：PyMuPDF / Marker 结构化提取正文
第二阶段：算法页转图片 → 多模态 LLM 精准提取 Algorithm Block
"""
import os
import base64
from pathlib import Path


def parse_paper(pdf_path: str, llm=None) -> dict:
    """
    两阶段解析入口
    Returns: {full_text, algo_description, pseudocode}
    """
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        return {"full_text": "", "algo_description": "", "pseudocode": ""}

    # 第一阶段：结构化文本提取
    full_text = _stage1_extract(pdf_path)

    # 第二阶段：算法页视觉提取
    algo_result = _stage2_vision_extract(pdf_path, full_text, llm)

    return {
        "full_text": full_text,
        "algo_description": algo_result.get("description", ""),
        "pseudocode": algo_result.get("pseudocode", ""),
    }


def _stage1_extract(pdf_path: str) -> str:
    """优先 Marker，回退 PyMuPDF"""
    # 尝试 Marker（高质量结构化解析）
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        models = load_all_models()
        full_text, _, _ = convert_single_pdf(pdf_path, models)
        print("[Parser] 第一阶段：Marker 解析完成")
        return full_text
    except Exception as e:
        print(f"[Parser] Marker 不可用 ({e})，回退到 PyMuPDF")

    # 回退：PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        full_text = "\n\n".join(pages)
        print(f"[Parser] 第一阶段：PyMuPDF 解析完成（{len(doc)} 页）")
        return full_text
    except Exception as e:
        print(f"[Parser] PyMuPDF 失败: {e}")
        return ""


def _stage2_vision_extract(pdf_path: str, full_text: str, llm=None) -> dict:
    """
    识别含算法块的页面，转图片后送多模态 LLM
    规避双栏排版和公式 OCR 乱码问题
    """
    results = {"description": "", "pseudocode": ""}

    if llm is None:
        # 无视觉模型时，从文本中粗提取
        results["pseudocode"] = _text_fallback_extract(full_text)
        results["description"] = "（算法描述从文本提取，未使用视觉解析）"
        return results

    try:
        import fitz

        doc = fitz.open(pdf_path)
        algo_pages = []

        # 定位含算法块的页面
        for page_num, page in enumerate(doc):
            text = page.get_text().lower()
            if any(kw in text for kw in ["algorithm", "procedure", "pseudocode", "algo."]):
                algo_pages.append((page_num, page))

        if not algo_pages:
            results["pseudocode"] = _text_fallback_extract(full_text)
            return results

        pseudocode_parts = []
        for page_num, page in algo_pages:
            # 页面转高清图片
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            try:
                from langchain_core.messages import HumanMessage
                response = llm.invoke([
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": (
                                "请精确提取图中的算法伪代码和算法描述。\n"
                                "要求：\n"
                                "1. 保留所有步骤编号和缩进结构\n"
                                "2. 保留所有数学符号和下标\n"
                                "3. 先输出算法描述（一段话概括），再输出完整伪代码\n"
                                "格式：\n[描述]\n...\n[伪代码]\n..."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ])
                ])
                content = response.content
                pseudocode_parts.append(f"--- 第 {page_num+1} 页 ---\n{content}")

                # 解析描述和伪代码
                if "[描述]" in content and "[伪代码]" in content:
                    desc_part = content.split("[伪代码]")[0].replace("[描述]", "").strip()
                    code_part = content.split("[伪代码]")[1].strip()
                    results["description"] += desc_part + "\n"
                    results["pseudocode"] += code_part + "\n"
                else:
                    results["pseudocode"] += content + "\n"

            except Exception as e:
                print(f"[Parser] 第 {page_num+1} 页视觉提取失败: {e}")
                results["pseudocode"] += _text_fallback_extract(page.get_text()) + "\n"

        print(f"[Parser] 第二阶段：视觉提取完成，共处理 {len(algo_pages)} 个算法页")

    except ImportError:
        print("[Parser] PyMuPDF 未安装，跳过视觉提取")
        results["pseudocode"] = _text_fallback_extract(full_text)

    return results


def _text_fallback_extract(text: str) -> str:
    """从纯文本中提取算法相关段落（无视觉模型时的回退）"""
    if not text:
        return ""

    lines = text.split("\n")
    algo_lines = []
    in_algo = False

    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ["algorithm", "procedure", "pseudocode"]):
            in_algo = True
        if in_algo:
            algo_lines.append(line)
            # 连续空行超过3行则认为算法块结束
            if len(algo_lines) > 5 and line.strip() == "":
                consecutive_empty = sum(1 for l in algo_lines[-3:] if l.strip() == "")
                if consecutive_empty >= 2:
                    in_algo = False

    return "\n".join(algo_lines[:200]) if algo_lines else text[:2000]
