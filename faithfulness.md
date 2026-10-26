---
tags: [evals, llm-judge, deepeval, faithfulness, rag]
source: confident-ai/deepeval → deepeval/metrics/faithfulness/
date: 2026-10-27
---

# Faithfulness: Checking an Answer Claim by Claim

Related: [[G-Eval - LLM Judge Design]] · [[DAG - Flowchart LLM Judge]] · [[Evals - Kaise Design Karein]]

## In one line

**Is everything the bot said actually in the documents it was given? Or did it make something up?**

This is for **RAG** apps: the bot first looks up documents, then answers from them. Faithfulness checks whether the bot **stuck to the documents** or **added things that aren't there**.

---

## Example

**Document given to the bot:**
> *"Refunds arrive in 5 days. Only to the original payment method."*

**Bot's answer:**
> *"Your refund will arrive in 5 days, on your original card. You'll also get a 10% bonus."*

---

## The 3 steps

**1. Break the answer into small claims**
```
claim 1: The refund arrives in 5 days
claim 2: It goes to the original card
claim 3: You get a 10% bonus
```

**2. Check each claim against the document** (the LLM only answers this)

| Claim | In the document? |
|---|---|
| Arrives in 5 days | ✅ yes |
| Goes to the original card | ✅ yes |
| 10% bonus | ❌ **no** — the bot made it up |

**3. Score = supported claims / total claims**
```
2 / 3 = 0.67
```

---

## Why this works

If you judged the whole answer at once, the LLM would say *"mostly correct, 8/10"*, and **the made-up bonus would slip through.**

Splitting it into claims means **every made-up fact gets caught on its own.**

> **One big fuzzy question → many small yes/no questions.**

---

## How it works inside (from the DeepEval code)

| File | What it does |
|---|---|
| `templates/generate_truths.txt` | Pull the facts out of the documents |
| `templates/generate_claims.txt` | Pull the claims out of the answer |
| `templates/generate_verdicts.txt` | For each claim: is it supported by the facts? Answer **no** if it contradicts them |
| `faithfulness.py → _calculate_score` | Passing claims / total claims |

Each claim gets one of three verdicts: **yes**, **no**, or **borderline** (the document only partly supports it).

The setting `penalize_ambiguous_claims` decides what borderline counts as:
- `False` (default) → borderline **passes**
- `True` → borderline **fails** (strict mode)

---

## When to use it

| ✅ Use Faithfulness | ❌ Use something else |
|---|---|
| RAG bots that answer from documents | No documents involved → G-Eval |
| Support bots answering from a policy | You need to judge tone or style → G-Eval |
| Summaries that must not add facts | Exact answer exists → plain code |
| Anywhere a made-up fact is costly | |

**Note:** Faithfulness only checks that the answer *matches the documents*. It does **not** check that the documents are correct, or that the answer actually answered the question.

---

## G-Eval vs DAG vs Faithfulness

| | What it does |
|---|---|
| **G-Eval** | The LLM writes a checklist, then gives a score |
| **DAG** | You build a flowchart, the LLM only answers yes / no |
| **Faithfulness** | Split the answer into claims, check each one against the documents |

All three share one idea: **don't ask one big question, ask many small ones.**

---

## Remember

1. Split the answer into **claims**.
2. Check **each claim** against the documents.
3. Score = **supported / total**. One made-up fact can't hide inside a mostly-correct answer.
