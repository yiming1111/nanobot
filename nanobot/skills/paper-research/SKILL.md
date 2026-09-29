---
name: paper-research
description: "Use the local scientific-paper corpus for evidence-grounded, multi-turn technical research and comparison."
always: true
metadata: {"nanobot":{"emoji":"📚","requires":{}}}
---

# Paper Research

Apply this workflow only when the user asks a question that should be answered from the configured scientific-paper corpus.

1. Resolve references and constraints from the conversation. Produce a standalone `normalized_question` without discarding the user's original wording.
2. Keep a simple factual question as one evidence need. Decompose a comparison, multi-hop question, or long instruction only when different facts require separate retrieval. Do not create redundant evidence needs. Citation formatting is an answer requirement, not a separate evidence need. If the user explicitly asks for one sub-question, create exactly one.
3. Call the tool ending in `research_start` with the original question, normalized question, constraints, and evidence needs. Retain its `task_id` and sub-question IDs.
4. For every pending evidence need, call the tool ending in `research_retrieve` once and process them sequentially. The tool searches all corpus chunks with dense and sparse retrieval, fuses them with RRF, and reranks the candidate set. Treat returned passages as candidates; finding a chunk does not prove semantic sufficiency.
5. When a passage is incomplete or ambiguous, call the tool ending in `get_neighbor_evidence` with the current `task_id` before judging it.
6. After all pending evidence needs have retrieval results, perform one Reflect pass. Judge only whether the retrieved passages directly and sufficiently cover each evidence need. A passage is insufficient when it is merely topically similar, omits a requested comparison side or constraint, lacks a requested value, or supports only a narrower conclusion than the user asked for.
7. Call the tool ending in `research_reflect` with one assessment for every unreviewed evidence need. A sufficient assessment must cite the exact evidence IDs it relies on. For an insufficient first round, provide one focused `next_query` that targets the missing information.
8. If Reflect returns `retry_sub_questions`, call `research_retrieve` once for each using the supplied focused query, then run `research_reflect` once more for those results. Never perform a third semantic search.
9. If Reflect returns `must_abstain`, explain that the local corpus lacks enough evidence. Do not fill the gap from model memory. If it returns `can_generate`, generate an answer using only evidence attached to sufficient evidence needs. For an unresolved evidence need, state the gap instead of inferring an answer.
10. Cite every factual conclusion with its paper title, page, and chunk ID. After drafting, call the tool ending in `research_finalize` with the answered sub-question IDs and every evidence ID actually cited. Fix invalid or missing locators before returning the answer.
11. Return the grounded answer. If final status is `completed_with_gaps`, include supported findings and clearly identify unresolved parts. If status is `refused`, return only the evidence limitation and refusal.

For multi-turn follow-ups, preserve prior constraints when the user narrows or extends the same subject. Start a new research task for each user turn; nanobot's Session supplies the conversational context used to normalize references such as “that paper”.
