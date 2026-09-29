# 科研论文多轮证据检索与技术调研 Agent：项目记录

最后更新：2026-09-29

## 1. 项目目标

在 nanobot 现有 Agent Loop、MCP、Skills、Session 和 Memory 能力上，增加一个面向本地科研论文库的技术调研 Agent。系统不仅进行 Query Rewrite 和普通论文检索，还把复杂调研问题拆成独立证据需求，分别检索和跟踪证据，并对回答中的原子主张建立主张—证据关系、引用验证和有边界的定向补检索；后续将增加 Pipeline Trace 与固定评测集。

项目不修改通用 `AgentLoop` 和 `AgentRunner`。领域能力作为 `nanobot.research` 扩展包实现，并通过 FastMCP 暴露给 nanobot；Agent 的调用规则由 `paper-research` Skill 提供。

## 2. 完整目标流程

```text
本地论文 PDF
  → 解析标题、作者、年份、章节、页码和正文
  → 章节感知滑动窗口切块
  → 构建论文级和段落级索引
      ├─ BGE-M3 稠密索引（FAISS）
      └─ BM25 稀疏索引

用户问题
  → Guardrails
  → 多轮上下文解析、指代消解和 Query Rewrite
  → 简单/复杂问题路由
  → 复杂问题拆成多个证据需求
  → 为每个子问题顺序执行分层检索
      → 论文级粗召回：BM25 + BGE-M3 + RRF
      → 保留 Top-N 候选论文（候选范围，不作为不可恢复的硬门）
      → 在候选论文内部执行 chunk 级 BM25 + BGE-M3 + RRF + Rerank
      → 写入 Pydantic ResearchState
  → 证据覆盖与冲突检查
      ├─ 证据充分：进入回答草稿生成
      ├─ 证据不足：根据证据缺口生成定向 Query，并扩大候选论文范围
      ├─ 扩大范围后仍不足：最多执行一次全库 chunk 检索兜底
      └─ 达到检索轮数上限仍无可用证据：明确拒答
  → 将草稿拆成原子主张
  → 验证引用可定位性、正确性和完整性
      ├─ 验证失败：修改表述、删除主张或再次检索
      └─ 验证通过：输出最终回答
  → 保存会话记忆、结构化调研状态和 Pipeline Trace
```

## 3. 状态与数据边界

- nanobot Session/Memory：保存对话历史、用户偏好和长期记忆。
- ResearchState：保存调研计划、子问题、检索 Query、候选证据、候选主张和引用检查结果。
- 论文 PDF、FAISS、BM25、任务状态和 Trace 属于运行数据，不提交到 Git。
- 源代码位于仓库根目录；`%USERPROFILE%\.nanobot\config.json` 仅保存 `paperResearch` MCP 运行配置，不在其中维护项目代码。论文原文件、模型缓存和检索索引由 `NANOBOT_RESEARCH_DATA_DIR` 与 `HF_HOME` 指向本机数据目录，不提交到 Git。

## 4. 实施顺序与当前进度

### 阶段 1：Pydantic 模型与状态存储【已完成】

- 新增 `ResearchPlan`、`SubQuestion`、`EvidenceItem`、`ClaimState`、`CitationCheck` 和 `ResearchState`。
- 对子问题 ID、页码、状态枚举和字段类型进行校验。
- `constraints` 支持标量及标量列表，可表达多个比较目标、引用字段等多值约束，避免 Agent 传入数组时校验失败。
- 实现按任务 ID 持久化的 `ResearchStateStore`。
- 使用文件锁、临时文件和原子替换避免任务状态写坏。

### 阶段 2：PDF 解析、切块与索引构建【已完成】

- 使用现有 `pypdf` 解析 PDF。
- 实现基础章节标题识别，保存章节和页码范围。
- 实现章节内部的滑动窗口切块及前后 chunk 链接。
- 实现论文和 chunk 的 JSONL 存储。
- 增加 PDF 正文质量门禁：清理跨页重复页眉/页脚、IEEE 授权与下载声明，并拒绝清理后为空或正文过少的文档。
- 实现论文级、段落级索引构建入口及索引 manifest。

当前限制：复杂双栏 PDF、公式、表格和扫描 PDF 的解析质量尚未专项处理。

### 阶段 3：BGE-M3 + BM25 + RRF + Rerank【已完成代码】

- 实现英文词项与中文字符/双字组合的 BM25，并支持持久化。
- 实现 RRF 多路排名融合。
- 实现 BGE-M3 编码器和 FAISS 余弦检索封装；根据694个真实 chunk 的 token 分布，将可配置的嵌入上限默认设为1024，降低本地 CPU 建库和查询耗时。
- 实现 BGE Reranker 封装。
- 实现论文检索、证据段落检索、年份/论文/章节过滤和上下文邻接段落读取。
- 修复候选论文过滤顺序：指定 `paper_ids` 时先限定允许参与排序的 chunk，再执行 BM25/FAISS Top-K、RRF 与 Rerank，避免相关 chunk 在全库截断阶段提前丢失。
- 重依赖放在 `research` 可选依赖组中，不影响普通 nanobot 安装。

真实运行验证：已安装 research 可选依赖并在本机下载 BGE-M3 与 BGE Reranker；成功索引 15 篇论文、生成 694 个 chunk，并建立论文级与段落级 FAISS/BM25 索引。2 篇 PDF 因清理后无可用正文被明确记录为失败且未进入索引。使用“势博弈与移动边缘计算资源分配”查询完成真实混合检索，前三名结果与主题一致。

### 阶段 4：FastMCP 检索工具【已完成】

当前向 Agent 注册六个工具：

- `research_start`：创建结构化调研计划和任务状态。
- `research_retrieve`：在一次受控调用内顺序完成论文粗召回、候选论文内证据检索、有限范围扩大和一次全库兜底。
- `get_neighbor_evidence`：读取证据前后段落，降低断章取义风险。
- `research_status`：读取任务计划和当前证据状态。
- `research_verify`：保存原子主张和独立语义复核结果，生成 Claim-Evidence Matrix，并计算引用可定位性、正确性和完整性。
- `research_retrieve_claim_gap`：只为验证失败且必要的主张执行一次有边界的缺口检索。

论文级 `search_papers` 与 chunk 级 `retrieve_evidence` 仍作为内部服务存在，但不再分别暴露给 Agent，防止模型并发调用两个有先后依赖的步骤。

MCP Server 入口：`nanobot-research-mcp` 或 `python -m nanobot.research.mcp_server`。

### 阶段 5：paper-research Skill【已完成】

- 对简单事实题保留单一子问题。
- 对比较、多跳和长指令问题进行证据需求拆解。
- 要求每个子问题使用独立 Query 检索。
- 要求在证据可能不完整时读取邻接 chunk。
- 当前阶段要求回答只能使用已检索证据，并按论文、页码和 chunk ID 引用。

### 阶段 6：检索编排与候选证据覆盖检查【已完成】

已实现：

- 新增 `research_retrieve` 服务端编排工具，强制 `search_papers → retrieve_evidence` 顺序执行；
- 将论文级检索定义为高召回粗筛：保留多个候选论文，不将单次结果作为不可恢复的硬过滤；
- `paper_ids` 存在时，先限定候选论文的 chunk，再执行候选截断与重排，避免“论文已命中但证据为空”；
- 候选论文内证据不足时，按“扩大候选论文范围 → 使用一个可选的替代表述 → 一次全库 chunk 兜底”的顺序恢复召回；
- 阶段 6 的检索级门禁只判断是否召回候选证据；Rerank 分数用于排序和观测，不再把未经校准的绝对分数当作相关概率。语义支持/反对关系留给阶段 7；
- 每个子问题最多三轮检索；已运行的子问题禁止重复执行完整流程，超时后应通过 `research_status` 查看状态；
- 记录候选论文、每轮范围、Query、证据 ID、充分性原因和耗时；
- Rerank 候选从 30 降至 12，默认返回证据从 8 降至 4；`research_status` 默认省略证据正文，避免上下文重复膨胀；
- 本地模型在同一 MCP 进程中只初始化一次并加锁复用；论文级粗召回仅执行 BM25、BGE-M3 与 RRF，不再使用 Cross-Encoder 对长论文表示重排；运行配置使用本地 Hugging Face 缓存，工具超时调整为 240 秒。
- `research_retrieve` 增加入口、论文召回及每轮证据检索的分段耗时日志，用于区分 MCP 调度等待、模型冷启动和实际检索耗时。
- MCP 启动时提前构造检索服务并加载 FAISS 索引，避免首次工具调用才触发索引导入；BGE-M3 与 reranker 仍按需加载。模型缓存启用离线读取，避免每次重启重复访问 Hugging Face。
- Skill 明确将引用格式与引用验证视为回答要求，不能拆成独立子问题；用户明确要求单一子问题时必须只创建一个。

### 阶段 7：Claim 提取与引用验证【已完成】

已实现：

- 新增 `ClaimDraft` 与 `EvidenceJudgment` Pydantic 输入模型，将回答草稿拆成独立、可核验、可追踪到子问题的原子主张；
- 新增独立语义复核轮：主 Agent 只能依据“主张 + 被引用段落”判断 `supported`、`partially_supported`、`unsupported`、`contradicted` 或 `not_enough_information`。BGE Reranker 仅负责相关性排序，不被错误当作蕴含判断模型；
- 新增 `research_verify`，对引用 ID 是否属于当前任务、是否属于对应子问题、标题/页码/chunk ID 是否可定位进行确定性检查，并聚合生成 Claim-Evidence Matrix；
- Claim 状态支持 `supported`、`partially_supported`、`conflicting`、`contradicted` 和 `insufficient`，矩阵为每条主张返回保留、缩小表述、报告冲突、补检索或删除等动作；
- 分别计算 Claim Support Rate、Citation Correctness 和 Citation Completeness，避免把“有引用”和“引用确实支持主张”混成同一指标；
- 增加子问题覆盖门禁：即使现有主张全部通过，只要仍有子问题没有对应主张，任务也不能被标记为完整完成；
- 新增 `research_retrieve_claim_gap`：只允许在验证阶段调用；每条失败主张最多补检索一次，整项任务默认最多两次；优先在原候选论文中检索，无结果时执行一次全库兜底；
- 默认最多执行两轮验证。首轮失败后可缩小表述、删除主张或做一次缺口检索；第二轮后进入 `completed`、`completed_with_gaps` 或 `refused`，防止无限反思与检索；
- 状态文件持久化主张、逐对引用检查、验证轮数和缺口检索记录，旧状态文件通过默认字段保持兼容；
- 更新 `paper-research` Skill，强制“先形成草稿 → 原子主张 → 独立语义复核 → 服务端结构校验 → 必要时一次修订/补检索 → 最终回答”的顺序。

真实 WebUI 回归已验证完全支持、同一 Session 内追问和本地语料不足场景；伪造 evidence ID、部分支持、冲突聚合与子问题覆盖由服务端单元测试覆盖。主 Agent 能按独立复核轮收窄表述，并在缺少充分证据时拒绝用户要求的过度结论。

### 阶段 8：Pipeline Trace 与评测【第一版已完成，答案级评测待扩展】

已实现：

- 新增按 `task_id` 持久化的 `PipelineTrace`，记录 Plan、Retrieve、邻接证据、Status、Gap Retrieval 和 Verify 的输入摘要、输出摘要、证据 ID、决策与耗时；Trace 不复制证据正文，避免再次制造超长工具历史；
- Trace 写入失败不阻断论文问答主流程，并提供 `nanobot-research trace <task_id>` 汇总工具调用次数、阶段耗时、证据 ID、错误和最近一次引用验证指标；
- 新增 JSONL 固定检索评测集、`nanobot-research eval` 命令与统一 JSON 报告，计算 Paper Recall@K、Paper MRR、Evidence Recall@K、Evidence Paper Recall@K 和检索延迟；
- 第一版真实基线包含 4 个已人工确认目标论文与目标 chunk 的问题，覆盖 MEC 广义纳什均衡、动态势博弈、车联网卸载和区块链资源定价；
- 首次基线结果：Paper Recall@10 = 1.0、Paper MRR = 1.0、Evidence Paper Recall@8 = 1.0、Evidence Recall@8 = 0.875；论文级召回平均约 646 ms，CPU 证据精排平均约 45.3 s；
- 车联网样例只召回 2 个标注 chunk 中的 1 个，形成一个可复现的后续改进点；该结果也验证了论文命中率与具体证据命中率必须分开统计。

仍需扩展：

- 为部分支持、冲突、语料不足和多轮约束保持建立人工标注的答案级评测集，统计 Groundedness、拒答准确率和多轮约束保持率；
- 将 Dense、Sparse、RRF、Rerank 的内部耗时进一步拆分。Generate、Token 和 Memory 位于 nanobot 主 Agent 侧，不能仅靠 research MCP 准确关联，后续应复用原生用量与会话日志做关联汇总，不能把模型自评当作真实标签。

### 阶段 9：WebUI 增强【暂不实施】

核心流程和评测稳定后，再决定是否增加引用定位、证据侧栏、子问题进度和 Trace 页面。当前阶段复用 nanobot 已有 Markdown、表格和工具活动展示。

## 5. 当前代码位置

```text
nanobot/research/config.py                 领域配置
nanobot/research/models.py                 Pydantic 状态模型
nanobot/research/corpus/parser.py          PDF 解析
nanobot/research/corpus/chunker.py         章节感知切块
nanobot/research/corpus/store.py           论文与 chunk 存储
nanobot/research/corpus/indexer.py         索引构建
nanobot/research/retrieval/dense.py        BGE-M3 + FAISS
nanobot/research/retrieval/sparse.py       BM25
nanobot/research/retrieval/fusion.py       RRF
nanobot/research/retrieval/reranker.py     Cross-Encoder Rerank
nanobot/research/retrieval/service.py      混合检索服务
nanobot/research/observability/trace.py    结构化 Pipeline Trace
nanobot/research/evaluation.py             固定检索集评测器
nanobot/research/workflow/state_store.py   调研状态持久化
nanobot/research/mcp_server.py             FastMCP 工具
nanobot/research/cli.py                    索引 CLI
nanobot/skills/paper-research/SKILL.md     Agent 调研流程
benchmarks/paper_research/retrieval.jsonl  固定检索评测集
tests/research/                            对应单元测试
```

## 6. 当前验证记录

- 科研模块单元测试：19 项通过。
- nanobot 现有 Skill Loader 回归测试：26 项通过。
- `nanobot.research` Python 编译检查通过。
- `nanobot-research --help` 启动通过。
- FastMCP 工具注册和 JSON Schema 生成通过；阶段 7 调整后确认注册 `research_start`、`research_retrieve`、`get_neighbor_evidence`、`research_status`、`research_verify`、`research_retrieve_claim_gap` 六个工具。`paperResearch` 已写入运行配置，重启 WebUI 后加载新工具清单。
- `nanobot-research status` 已验证质量门禁后的真实索引：15 篇论文、694 个 chunk、`dense_enabled=true`。
- PDF 质量门禁已扫描 17 份真实文件：15 份正文可用，2 份因无可用正文被拒绝；完整 research 测试 13 项通过。持久化索引已按新质量门禁完成重建。
- 真实检索已验证 BGE-M3、BM25、RRF 与 BGE Cross-Encoder Rerank 全链路可运行；首次查询返回的前三篇均与势博弈/边缘计算资源分配相关。
- 简单问题的首次 WebUI 实测最终找到了目标论文、证据及邻接 chunk，但 `search_papers` 与 `retrieve_evidence` 被 Agent 并发调用，且两次调用均超过 180 秒 MCP 等待上限；后台任务继续完成并将证据写入状态。该结果证明检索能力可用，同时暴露了阶段 6 需要解决的顺序编排、冷启动和超时问题。
- 阶段 6 首次统一工具实测中，MCP 调用在实际检索开始前异常等待满 600 秒；超时后模型加载约 25 秒，而 SQ1 三轮 chunk 检索分别约 6.0、2.9、3.5 秒。该结果说明十分钟等待不属于 RRF 或 chunk 检索计算，下一次通过新增入口日志继续定位 MCP 调度/进程初始化边界。
- 同次实测发现 BGE reranker 的相关段落分数约为 `0.00004`，原先未经校准的 `0.5` 门槛错误触发三轮范围扩大；现已取消默认绝对分数门槛。论文粗召回的 Cross-Encoder 也已移除，避免 CPU 对长论文表示重排。
- 上述性能修正后，科研模块单元测试重新运行：13 项通过；FastMCP 仍只注册 `research_start`、`research_retrieve`、`get_neighbor_evidence`、`research_status` 四个工具；运行配置 JSON 校验通过。
- 240 秒回归测试只执行一个子问题和一轮检索并返回 4 个候选段落，但工具进入后到 FAISS 加载前仍异常等待满 240 秒；随后论文召回约 28 秒、证据检索约 61 秒并成功生成回答。现已把检索服务、FAISS、BGE-M3、reranker 的初始化日志进一步拆开，并将实测仅需约 0.7 秒的检索服务与 FAISS 索引准备移到 MCP 启动阶段，等待下一次重启验证。
- 加细日志后的第二次回归确认：FAISS 在 MCP 启动时约 0.7 秒完成；240 秒全部发生在 `BGEM3FlagModel` 构造期间，BGE-M3 初始化总计约 251.7 秒，而后论文向量召回约 4 秒、reranker 初始化约 7.4 秒、chunk 检索与精排约 55 秒。未发现遗留 Python 进程或 Hugging Face 锁文件。
- 将模型配置改为直接指向 D 盘 snapshot 后，重启后的 BGE-M3 初始化仍耗时约 250.6 秒，证明瓶颈不是 Hub 模型名称到本地缓存的解析。该诊断改动已撤销，配置恢复为 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3`。
- 同一 MCP 进程中的后续问题未超时：论文召回约 3 秒、证据检索与精排约 51 秒、整轮对话约 87 秒，且没有再次出现模型初始化日志。这确认 240 秒超时集中在进程重启后的首次 BGE-M3 构造，而不是每次检索都会发生。
- 已将初始化责任从首次查询移到 MCP 启动阶段：`paperResearch` 在注册工具前加载 FAISS、BGE-M3、reranker，并分别执行一次短文本编码与重排预热；只有全部完成后才对 Agent 显示为已连接。科研模块测试现为 14 项通过。
- 启动预加载真实验证完成：FAISS 约 0.8 秒、BGE-M3 初始化约 23.6 秒、reranker 初始化约 7.3 秒，包含预热的 MCP runtime 准备总计约 34.6 秒。此前约 250 秒并非模型正常读取耗时，而是模型在工具调用生命周期内初始化时出现的异常等待；移到 MCP 启动阶段后该等待消失。
- 预加载后的首次真实提问未再超时：论文粗召回约 1 秒，chunk 检索与精排约 56 秒，包含 LLM 规划和回答生成的整轮耗时约 79 秒；结果成功引用目标论文。阶段 6 的首次查询超时问题已关闭。
- 阶段 7 新增模型、引用可定位性、语义判断聚合、子问题覆盖、验证轮数、伪造 evidence ID 和有边界缺口检索测试；完整 research 测试现为 19 项通过。
- 本机 `nanobot-dev` 环境没有 pytest；测试通过复用本机已有 pytest 包执行，没有安装或修改依赖。测试出现的 `asyncio_mode` 警告来自该复用环境缺少 pytest-asyncio，不影响本次同步测试结果。
- 阶段 7 首次真实 WebUI 回归通过：两轮问题使用同一 nanobot Session；第二轮“刚才那篇论文”被正确还原为完整论文标题，两轮均按 `research_start → research_retrieve → research_verify` 完成且没有超时。两个独立 ResearchState 均为 `completed`，对应 Claim 均为 `supported`。
- Windows 的通用版 FAISS wheel 仅包含 `_swigfaiss.pyd` 时，现会在导入前自动选择 `FAISS_OPT_LEVEL=generic`，避免先探测不存在的 `swigfaiss_avx2` 并打印误导性的 `ModuleNotFoundError`。真实 FAISS 导入与向量查询通过，检索测试 4 项通过。
- 阶段 8 新增 Trace Store 与固定检索集评测器单元测试，新增测试 2 项通过；使用 nanobot-dev 原生环境完成 ResearchTools Trace 冒烟测试，确认 `research_start → research_retrieve → get_neighbor_evidence → research_status` 按顺序落盘。
- 4 项真实固定集基线已运行成功：Paper Recall@10 与 MRR 均为 1.0，Evidence Recall@8 为 0.875，证据精排平均约 45.3 秒。报告由统一 JSON 评测命令生成。

## 7. 下一步

1. 扩充答案级人工标注集，覆盖部分支持、冲突、语料不足和多轮约束保持；不要用待测模型自己的判断直接充当金标准。
2. 拆分 Dense、Sparse、RRF 与 Rerank 的内部耗时，并优先分析当前约 45 秒的 CPU 精排耗时。
3. 用阶段 8 基线定位并改进车联网样例遗漏的目标 chunk，改动后重复运行同一固定集，比较召回与延迟是否退化。
