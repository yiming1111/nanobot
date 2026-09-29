---
name: paper-research
description: "Use the local scientific-paper corpus for evidence-grounded, multi-turn technical research and comparison."
always: true
metadata: {"nanobot":{"emoji":"📚","requires":{}}}
---

# Paper Research

Apply this workflow only when the user asks a question that should be answered from the configured scientific-paper corpus.

1. Resolve references and constraints from the conversation. Produce a standalone `normalized_question` without discarding the user's original wording.
2. Keep a simple factual question as one sub-question. Decompose a comparison, multi-hop question, or long instruction into independent evidence needs. Do not create redundant sub-questions. Citation formatting, source collection, and citation verification are requirements on the answer, not separate sub-questions. If the user explicitly asks for one sub-question, create exactly one.
3. Call the MCP tool whose name ends in `research_start` with the original question, normalized question, constraints, and sub-questions. Retain its `task_id` and sub-question IDs.
4. For every sub-question, call the tool ending in `research_retrieve` exactly once and process sub-questions sequentially. Give it one focused query and, only when useful, one alternative query using different terminology. Do not call `search_papers` and an evidence tool in parallel. `research_retrieve` owns paper-level recall, candidate-paper chunk retrieval, bounded expansion, one global fallback, RRF, reranking, and stopping.
5. Inspect the returned `coverage`, `attempts`, and `sub_question_status`. Do not retry a completed sub-question manually. When coverage remains insufficient, report the gap instead of starting an unbounded search loop.
6. When a passage may be incomplete or ambiguous, call the tool ending in `get_neighbor_evidence` with the current `task_id` before relying on it.
7. Use the tool ending in `research_status` to inspect accumulated state before drafting. Request evidence text only when it is no longer available in the current context. Treat retrieved passages as candidates, not automatically proven claims.
8. Draft the answer internally, then split it into atomic, independently checkable factual claims. Each claim must belong to one sub-question and list the exact evidence IDs it cites. Do not turn headings, transitions, opinions, or statements of missing evidence into claims.
9. Perform a separate verification pass using only each claim and its cited passage text. For every claim-evidence pair, classify the relationship as `supported`, `partially_supported`, `unsupported`, `contradicted`, or `not_enough_information`; provide a short rationale and, when narrowing the wording would make it accurate, a concrete revision. This is semantic verification: retrieval and reranker scores are not proof of support.
10. Call the tool ending in `research_verify` with the atomic claims and pairwise judgments. Follow the returned Claim-Evidence Matrix:
    - `keep`: retain the claim;
    - `use_revision` or `narrow_or_retrieve`: use the proposed narrower wording, or retrieve once if the missing fact is essential;
    - `report_conflict_or_retrieve`: report the disagreement or retrieve once to resolve it;
    - `remove_or_reverse`: remove the claim unless the cited evidence clearly supports a corrected opposite statement;
    - `retrieve_or_remove`: retrieve once for an essential claim, otherwise remove it.
    Treat any `uncovered_sub_question_ids` as an unresolved answer gap; supported claims from other sub-questions do not make the whole request complete.
11. For an essential failed claim only, call the tool ending in `research_retrieve_claim_gap` with a focused query. The tool permits at most one gap search per claim and a bounded number per task. Review the returned passages and call `research_verify` once more with the revised final claim set. Do not start an open-ended retry loop.
12. Answer only with claims allowed by the final matrix. Cite each factual claim with the paper title, page, and chunk ID from the verified locator. If the task status is `completed_with_gaps`, answer with supported claims and state the unresolved gaps. If it is `refused`, explain that the local corpus lacks adequate support instead of filling gaps from model memory.

For multi-turn follow-ups, preserve prior constraints and the current task when the user narrows or extends the same investigation. Start a new task when the research subject changes materially.
