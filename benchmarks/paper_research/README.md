# 论文研究 Agent 评测集

本目录只维护两类离线评测数据：单轮问答和多轮对话。检索侧与生成侧使用同一批问题，避免两套问题导致结果无法对应。

## 数据规模

| 文件 | 规模 | 用途 |
|---|---:|---|
| `single_turn.jsonl` | 60 题 | 建立单轮检索、生成、证据边界和引用基线 |
| `multi_turn.jsonl` | 10 组、80 轮 | 检查连续追问、指代与约束继承，以及上下文压缩后的信息保持 |

单轮集包含 40 道 `answer`、10 道 `partial` 和 10 道 `abstain`。40 道可完整回答的问题覆盖当前索引中的全部 15 篇文档；部分回答和拒答案例用于检查系统是否会超出本地证据作答。

多轮集每组固定 8 轮。后两轮通常回查对话前段出现的论文、约束或结论，并用 `memory_probe=true` 标记。只有运行日志确认该轮之前实际发生了上下文压缩，才能将这些轮次计入压缩后记忆一致性统计。

## 单轮字段

```json
{
  "case_id": "st-001-vehicular-game",
  "query": "车辆MEC论文如何建模多车计算卸载，车辆调整的核心决策变量是什么？",
  "ground_truth": "依据本地论文整理的标准答案",
  "relevant_chunk_ids": ["4c26f68525f2edb0-C0000"],
  "expected_behavior": "answer",
  "unsupported_requirements": [],
  "tags": ["single-paper", "model"]
}
```

- `ground_truth`：只依据本地语料编写的标准答案。
- `relevant_chunk_ids`：人工核对后能够直接支持标准答案的 chunk。
- `expected_behavior`：`answer`、`partial` 或 `abstain`。
- `unsupported_requirements`：`partial` 和 `abstain` 中没有本地证据的要求。
- `tags`：用于分析失败场景，不作为意图分类标签。

## 多轮字段

多轮数据沿用单轮字段，并增加：

- `dialogue_id` 与 `turn_index`：确定对话分组和顺序。
- `must_preserve`：本轮必须从前文正确继承的论文、指代或约束。
- `memory_probe`：该轮是否用于检查较早信息的保持情况。

## 标注原则

1. 先阅读论文并编写标准答案，再标注证据，不能根据被测系统的返回结果修改标准答案。
2. 一个问题可对应多个相关 chunk。相邻 chunk 只有在能独立或共同支持答案时才纳入。
3. `partial` 必须同时给出有证据的部分，并明确指出 `unsupported_requirements` 中的缺口。
4. `abstain` 不包含相关 chunk，标准答案只说明当前语料不足。
5. 评测集固定后才能建立 baseline；发现真实标注错误时，应记录修订原因并重新计算所有版本结果。

## 计划使用的指标

检索侧使用 `Recall@4`、`Precision@4`、`MRR`、平均/P95耗时和超时率；不计算 NDCG。生成侧使用 `Faithfulness`、`Answer Relevancy`、`Context Relevancy` 和 `Context Recall`。此外记录 `Decision Accuracy`、`Citation Accuracy`，并在多轮数据上统计 `Memory Consistency Rate`。

意图识别、Query 改写和子问题拆解只作为 Pipeline Trace 中的诊断信息，不单独计算准确率。

## 运行单轮检索基线

```powershell
nanobot-research eval-single benchmarks/paper_research/single_turn.jsonl `
  --data-dir D:/Data/paper-research/index `
  --hf-home D:/Data/huggingface `
  --top-k 4 `
  --output benchmarks/paper_research/results/single_turn_retrieval_raw.json
```

该命令把评测集中的中文用户问题直接交给全库 chunk 检索，用于测量检索器自身的跨语言能力。完整 Agent 评测则从中文用户问题开始，由 Agent 在运行时生成英文检索 Query；生成的 Query 属于预测结果和 Pipeline Trace，不写入评测集。`answer` 和 `partial` 共 50 题参与 Recall、Precision 和 MRR 计算；`abstain` 没有正确 chunk，因此只参与耗时、超时率和错误率统计。

`results/single_turn_retrieval_zh_raw.json` 保存中文原问题直接检索英文语料的基线。完整 Agent 运行产生的英文 Query、检索结果和最终回答应写入 `SingleTurnPrediction`，不能作为人工标签预先填入测试集。

若只评估 Query 改写对检索的影响，而不混入答案生成、Reflect 和引用校验耗时，运行：

```powershell
python benchmarks/paper_research/run_rewritten_query_eval.py `
  --benchmark benchmarks/paper_research/single_turn.jsonl `
  --workspace <独立评测工作区> `
  --data-dir D:/Data/paper-research/index `
  --hf-home D:/Data/huggingface `
  --top-k 4 `
  --output benchmarks/paper_research/results/single_turn_retrieval_en_query.json
```

该脚本使用当前模型把每道中文问题改写为一条忠实的英文检索 Query，再运行与中文基线相同的全库 chunk 检索。英文 Query 作为预测结果保存在报告中，不写回人工标注的评测集。使用 `--resume` 可从已保存的 Query 或检索结果继续执行。

单题异常会记录在报告的 `error` 字段中，整批评测会继续执行。`timeout_seconds` 是评测阈值：超过该时长会计入超时率，但评测程序会等待该次本地检索结束，从而避免在后台留下未受控的模型任务。

Agent 完整回答可保存为 `SingleTurnPrediction` JSONL，随后执行 `nanobot-research score-single <测试集> <回答文件>` 计算完成率、决策准确率和引用准确率。回答记录包含 Agent 实际生成的英文检索 Query、检索/引用 chunk、`research_finalize` 的引用定位检查结果、任务 ID、总耗时和错误。Faithfulness、Answer Relevancy、Context Relevancy 与 Context Recall 需要真实回答和检索正文，不能在只有测试题时预先生成分数。

命令默认启用 `--offline`，直接使用已经下载到本机的 BGE 模型文件，防止 Hugging Face 联网检查混入检索耗时。如果本机尚未缓存模型，可临时使用 `--online` 完成首次下载。

## 运行端到端冒烟评测

下面的命令在独立工作区中运行 5 道代表性问题：3 道完整回答、1 道部分回答和 1 道拒答。独立工作区保留 paper-research Skill，但使用空 Memory 和互相隔离的 Session，避免历史论文记录绕过检索。

```powershell
python benchmarks/paper_research/run_agent_smoke.py `
  --benchmark benchmarks/paper_research/single_turn.jsonl `
  --workspace <独立评测工作区> `
  --output benchmarks/paper_research/results/single_turn_agent_smoke_raw.json
```

本次结构化预测保存在 `results/single_turn_agent_smoke.jsonl`，汇总与已发现问题保存在 `results/single_turn_agent_smoke_report.json`。英文检索 Query 来自实际 Agent Trace，不写入人工评测集。
脚本会为每次运行生成新的 `run_id` 并写入 Session key，防止重复运行时恢复旧评测会话；需要复现实验标识时可显式传入 `--run-id`。
普通单题错误不会中断整批评测；如果供应商明确返回额度耗尽或账户欠费，脚本会停止本次运行，因为后续题目无法产生有效结果。

在同样的隔离条件下运行全部 60 道单轮题，并允许中断后续跑：

```powershell
python benchmarks/paper_research/run_agent_smoke.py `
  --benchmark benchmarks/paper_research/single_turn.jsonl `
  --workspace <独立评测工作区> `
  --output benchmarks/paper_research/results/single_turn_agent_full_raw.json `
  --run-id <本次实验标识> `
  --all `
  --resume
```

全量模式仍为每道题分配独立 Session，并在每题结束后立即保存结果。工具输出只保留调用名称、参数、错误和研究任务 ID；完整证据与阶段记录继续由 `ResearchState` 和 `PipelineTrace` 保存，避免批量结果重复写入 chunk 正文。

运行完成后，将原始检查点与 `ResearchState` 合并为结构化预测和正式报告：

```powershell
python benchmarks/paper_research/build_agent_report.py `
  --benchmark benchmarks/paper_research/single_turn.jsonl `
  --raw benchmarks/paper_research/results/single_turn_agent_full_raw.json `
  --state-dir D:/Data/paper-research/index/states `
  --model <模型名称> `
  --predictions benchmarks/paper_research/results/single_turn_agent_full.jsonl `
  --report benchmarks/paper_research/results/single_turn_agent_full_report.json
```

报告中的 `completion_rate` 只表示 Agent 返回了结果；`pipeline_execution_rate` 表示实际建立了论文研究任务；`pipeline_success_rate` 进一步要求决策类型正确，且需要引用的回答通过定位校验。绕过论文检索而直接使用模型常识作答，不计为 Pipeline 成功。`target_chunk_recall` 使用每题所有检索轮次的 chunk 并集计算，用来诊断端到端流程是否覆盖人工证据，不等同于固定 K 的单次检索 Recall@K。

## qwen3.7-plus 单轮全量基线（2026-10-04）

| 指标 | 结果 |
|---|---:|
| 回答完成率 | 100.0% |
| 严格决策准确率 | 91.7% |
| 引用定位准确率 | 94.0% |
| 论文研究 Pipeline 执行率 | 95.0% |
| 严格 Pipeline 成功率 | 91.7% |
| 端到端目标 chunk 平均召回率 | 66.6% |
| 目标 chunk 零命中率 | 8.0% |
| 平均总耗时 | 138.0 秒 |
| P95 总耗时 | 249.9 秒 |

严格口径把没有 `ResearchState` 的结果计为决策和 Pipeline 失败。本轮有两道可回答题绕过论文检索、直接使用模型常识作答；另有一道明显越界的天气题在检索前直接拒答。状态明确的两处行为错误是：`st-017` 未召回关键证据后误拒答，`st-059` 忽略“只依据指定文件”的来源约束后回答了另一份已入库论文。完整逐题结果见 `results/single_turn_agent_full_report.json`。
