# PaperCoder 方案 v4

> 基于 v3 方案实际落地后的更新，记录真实实现与历次变更
> *2026.3*

---

## 一、与 v3 的主要差异

### 新增
- **Survey 模式**：多论文横向调研，`Planner → Researcher → Surveyor`，不走 Coder/Reviewer
  - `--survey`：领域综述，多篇论文横向对比
  - `--survey --followup <论文名>`：跟进调研，查找某论文的后续优化工作
- **LLM 自动切换**：支持 Gemini / Claude / OpenAI，按 `MODEL_PROVIDER` 优先级自动 fallback

### 调整（相对 v3）
- LLM 默认改为 **Gemini 2.5 Flash Lite**（免费额度更大），原方案默认 Anthropic
- Researcher 输出增加**压缩步骤**：超 1200 字自动二次压缩，防止后续节点输入过长
- Reporter 使用 `bind(max_output_tokens=6144)` 控制输出长度
- **Mermaid 生成改为结构化方案**（v4.1）：LLM 输出节点/边 JSON（Pydantic），程序拼接语法，彻底消除括号/孤立节点等渲染错误
- **Reviewer 去掉 GitHub 相关性维度**（v4.1）：GitHub 检索结果不稳定，不应影响迭代决策；权重调整为准确性×0.40 + 代码×0.35 + 流程图×0.25
- **本地 RAG 目录扩展**（v4.1）：同时扫描 `rag/`（用户手动放置）和 `papers/`（arXiv 自动下载）

### 未实现（与 v3 方案的差距）
- Nougat 数学公式专项解析（PyMuPDF 直接替代，已够用）
- LLM-as-Judge 消融实验（judge.py 已有框架，未跑完整5组）
- LangSmith 可观测性（接口已预留，未验证）

---

## 二、系统架构

### 单篇精读图（默认）

```
用户输入（arXiv ID / 标题 / PDF）
   │
   ▼
Planner
  ├── 查询 FAISS 跨会话记忆
  ├── 自动下载 arXiv PDF（若未提供）
  └── LLM 分解 research + code 子任务
   │
   ├──────────────────────────────┐ 并行
   ▼                              ▼
Researcher                      Coder
  四路检索：                      Step1 PDF 解析（PyMuPDF + 视觉LLM）
  arXiv / Semantic Scholar /     Step2 算法提取 + 代码骨架生成
  Tavily / 本地RAG               Step3 结构化 Mermaid 流程图
  超长输出自动压缩                 Step4 GitHub 开源实现检索
   │                              │
   └──────────────┬───────────────┘ 等待两路完成
                  ▼
              Reviewer（Self-Refine，最多3轮）
                LLM-as-Judge，三维评分：
                · 算法准确性 × 0.40
                · 代码骨架质量 × 0.35
                · 流程图逻辑性 × 0.25
                score ≥ 6 → Reporter
                score < 6 且 iter < 3 → 携带 feedback 退回 Coder
                  │
                  ▼
              Reporter
                生成三件套：综述 + 流程图 + 代码
                写入 FAISS 长期记忆
                  │
                  ▼
              output/ 目录
```

### Survey 调研图

```
Planner → Researcher → Surveyor（论文发现 → 逐篇分析 → 综述生成）
```

---

## 三、技术栈（实际使用）

**Agent 编排**：LangGraph 0.2+ / LangChain 0.3+

**LLM**：
- 主力：Gemini 2.5 Flash Lite（`langchain-google-genai`，免费额度大）
- 备用：Claude Sonnet 4.6（`langchain-anthropic`）/ GPT-4o（`langchain-openai`）
- 视觉：与主力同一模型（Gemini / Claude / GPT-4V 均原生多模态）

**检索**：
- arXiv Python SDK（论文搜索 + PDF 下载）
- Semantic Scholar REST API（引用关系，可选 API Key）
- Tavily（网络搜索）
- LlamaIndex + ChromaDB（本地 PDF RAG，`rag/` + `papers/` 双目录）

**PDF 解析**：
- 第一阶段：PyMuPDF（基础，默认）/ Marker（可选，高质量结构化）
- 第二阶段：视觉 LLM 识别算法页（Algorithm Block 提取）

**Mermaid 流程图**：
- 结构化生成：`llm.with_structured_output(MermaidDiagram)` 输出节点/边 JSON
- `_build_mermaid()` 程序化拼接，标签自动加引号，无语法错误风险
- 降级链：结构化失败 → 文本生成 + `_extract_mermaid_block()` 后处理修复

**GitHub 检索**：GitHub MCP Server（npx），仅用于报告展示，不参与评审评分

**记忆**：
- 跨会话：FAISS + sentence-transformers（all-MiniLM-L6-v2），fallback JSON
- 会话级：LangGraph MemorySaver（支持断点续跑）

**Reviewer 评分**：
- 维度：算法准确性 / 代码骨架质量 / 流程图逻辑性（共3维，去除了 GitHub 相关性）
- 综合分 = 0.40×algo + 0.35×code + 0.25×diagram
- pass_review = overall ≥ 6.0

---

## 四、文件结构

```
papercoder/
├── main.py              # CLI 入口，--survey / --followup / --no-judge 参数
├── graph.py             # 两张图：build_graph() / build_survey_graph()
├── state.py             # PaperCoderState（含 survey_type / base_paper）
├── llm_factory.py       # get_llm() / get_vision_llm()，lru_cache 单例，三供应商 fallback
├── nodes/
│   ├── planner.py       # FAISS 查询 + Pydantic 结构化任务分解（SubTask / PlannerOutput）
│   ├── researcher.py    # Tool Calling Agent（max 5步），超长输出自动压缩到 1200 字
│   ├── coder.py         # 两阶段 PDF 解析 + 代码骨架 + 结构化 Mermaid 生成
│   ├── reviewer.py      # ReviewResult Pydantic（3维）+ should_refine 路由
│   ├── reporter.py      # 三件套整合，bind(max_output_tokens=6144)，写入 FAISS
│   └── surveyor.py      # 论文发现 → 逐篇分析 → 对比综述
├── tools/
│   ├── arxiv_tool.py    # arXiv 搜索 + PDF 下载（缓存到 papers/）
│   ├── semantic_scholar.py
│   ├── web_search.py    # Tavily
│   ├── local_rag.py     # LlamaIndex + ChromaDB，扫描 rag/ + papers/
│   ├── paper_parser.py  # PyMuPDF + 视觉 LLM 两阶段解析
│   └── github_mcp.py    # GitHub MCP Server
├── memory/long_term.py  # LongTermMemory，FAISS 向量检索 + JSON fallback
├── eval/judge.py        # 独立 LLM-as-Judge，JudgeScore Pydantic
├── rag/                 # 用户手动放置论文 PDF（本地 RAG 来源之一）
├── papers/              # arXiv 自动下载缓存
├── chroma_db/           # ChromaDB 向量索引（首次运行时自动创建）
└── output/              # 生成结果：*_review.md + *_code.py
```

---

## 五、已知问题 / 待优化

| # | 问题 | 状态 |
|---|------|------|
| 1 | 短别名识别：`"LLaDA"` 这类简称可能被 LLM 误解，建议用完整标题或 arXiv ID | 已知，建议规避 |
| 2 | Gemini 输出不稳定：部分请求 `max_output_tokens` 未被严格遵守，已加后备截断 | 已缓解 |
| 3 | 并行节点 state 合并：Researcher/Coder 并行写不同字段，无冲突，但 LangGraph 升级需注意 reducer | 已知，注意版本 |
| 4 | Mermaid 括号/孤立节点导致渲染错误 | **已修复**（结构化生成） |
| 5 | GitHub 评分拖累 Reviewer，导致不必要的重跑 | **已修复**（移除该维度） |
| 6 | RAG 只读 papers/ 目录，rag/ 文件夹无效 | **已修复**（双目录扫描） |
| 7 | Survey 模式论文数量受 Semantic Scholar 速率限制，结果不稳定 | 待优化 |
| 8 | 每轮 Self-Refine 均重新解析 PDF，浪费 IO | 待优化（首轮结果应缓存） |
