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
