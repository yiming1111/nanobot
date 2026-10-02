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
4. For every pending evidence need, first rewrite that need as one concise English retrieval query, preserving paper or algorithm names, formulas, years, numbers, and other constraints. Use that single English query for the tool ending in `research_retrieve`, and process evidence needs sequentially. The tool searches all corpus chunks with dense and sparse retrieval, fuses them with RRF, and reranks the candidate set. Treat returned passages as candidates; finding a chunk does not prove semantic sufficiency. The user's question and final answer may remain in the user's language.
5. When a passage is incomplete or ambiguous, call the tool ending in `get_neighbor_evidence` with the current `task_id` before judging it.
6. After all pending evidence needs have retrieval results, perform one Reflect pass. Judge only whether the retrieved passages directly and sufficiently cover each evidence need. A passage is insufficient when it is merely topically similar, omits a requested comparison side or constraint, lacks a requested value, or supports only a narrower conclusion than the user asked for.
7. Call the tool ending in `research_reflect` with one assessment for every unreviewed evidence need. A sufficient assessment must list every supporting chunk ID it relies on. For an insufficient first round, provide one focused `next_query` that targets the missing information.
8. If Reflect returns `retry_sub_questions`, call `research_retrieve` once for each using the supplied focused query, then run `research_reflect` once more for those results. Never perform a third semantic search.
   - A technical retry is separate from this semantic retry. If `research_retrieve` times out, do not immediately start the same retrieval again. Call `research_status` with `include_evidence_text=true` and `wait_seconds=10`, at most three times. If retrieval completed, continue from the saved evidence. If it failed, repeat the same `research_retrieve` call once. If it is still running after the third check, tell the user it is still processing and stop this turn. Never run duplicate retrievals concurrently.
9. If Reflect returns `must_abstain`, explain that the local corpus lacks enough evidence. Do not fill the gap from model memory. If it returns `can_generate`, generate an answer using only evidence attached to sufficient evidence needs. For an unresolved evidence need, state the gap instead of inferring an answer.
10. Cite every factual conclusion for readers with its paper title and page. After drafting, call the tool ending in `research_finalize` with the answered sub-question IDs and structured `(sub_question_id, chunk_id)` citations. Use only chunks approved by Reflect, and fix invalid or missing locators before returning the answer. Chunk IDs are internal provenance and should not appear in the reader-facing answer.
11. Return the grounded answer. If final status is `completed_with_gaps`, include supported findings and clearly identify unresolved parts. If status is `refused`, return only the evidence limitation and refusal.

For multi-turn follow-ups, preserve prior constraints when the user narrows or extends the same subject. Start a new research task for each user turn; nanobot's Session supplies the conversational context used to normalize references such as “that paper”.
