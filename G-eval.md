---
tags: [evals, llm-judge, deepeval, g-eval]
source: confident-ai/deepeval → deepeval/metrics/g_eval/
date: 2026-10-27
---

# G-Eval: How to Design an LLM Judge

Related: [[Evals - Kaise Design Karein]] · [[EVALS]] · [[Adrian Eval - All Files Explained]] · [[LLM Evals — Human vs Automated]]

## In one line

**Don't ask the judge for a score straight away. First make it write a checklist, then make it score with that checklist.**

This way, every answer is judged the same way.

---

## The 3 steps

**1. Make a checklist** (only once)
You write: *"Is the answer correct?"*
The LLM turns this into 3–4 steps. The same checklist is used for **every** test case.

**2. Score with the checklist**
The LLM checks each answer against the checklist and returns:
```json
{ "reason": "Facts are correct, but one detail is missing", "score": 7 }
```

**3. Pass or fail**
```
7 / 10 = 0.7   →   0.7 ≥ 0.5 (threshold)   →   PASS ✅
```

---

## Example: Customer support bot

**Criteria:** *"Did the bot answer the customer's question correctly and politely?"*

**Checklist (written by the LLM):**
1. Did it answer the question?
2. Is the answer correct?
3. Is the tone polite?

| | Case A | Case B |
|---|---|---|
| Customer | "When will I get my refund?" | "When will I get my refund?" |
| Bot | "Your refund will arrive in 3–5 days. Anything else I can help with?" | "Read the policy." |
| Answered? | ✅ | ❌ |
| Correct? | ✅ | ❌ |
| Polite? | ✅ | ❌ |
| Score | 9/10 = 0.9 | 2/10 = 0.2 |
| Result | **PASS** ✅ | **FAIL** ❌ |

Both cases were checked with **the same 3 steps**. That is why the scores are fair.

---

## When to use it

**Use it when there is no single correct answer, and you have to judge whether something is "good".**

| ✅ Use G-Eval | Example |
|---|---|
| Tone / politeness | Was the support bot respectful? |
| Helpfulness | Did the answer actually help the user? |
| Summary quality | Did the summary include the key points? |
| Same meaning, different words | Expected *"3–5 days"*, bot said *"by next week"* → both are fine |
| Your own custom rule | *"Did the bot avoid giving medical advice?"* |

| ❌ Don't use G-Eval | Use this instead |
|---|---|
| Math answer | `answer == 42` |
| SQL query | Run it and compare the result |
| Valid JSON | Parse it and check |
| Classification code | `got == "M3.a"` |

> **If code can check it → check it with code.**
> **If only a human could judge it → use G-Eval.**

In Adrian I did not use G-Eval, and that was the right call. The judge returned a code (`M3.a`), so `got == expected` was enough. Cheaper, and no judge mistakes.

---

## How it works inside (from the DeepEval code)

```
criteria → [1] steps → [2] score → [3] probability smoothing → [4] 0–1 + pass/fail
```

| File | What it does |
|---|---|
| `templates/generate_evaluation_steps.txt` | Turns criteria into 3–4 steps, as JSON `{"steps": [...]}` |
| `templates/generate_evaluation_results.txt` | Uses steps + test case to return `{"reason", "score"}` |
| `utils.py → calculate_weighted_summed_score` | Smooths the score using probabilities |
| `g_eval.py` | Ties it together, scales the score, applies the threshold |

### 5 design choices and why

| Choice | Why |
|---|---|
| Steps are made **once** and reused for every case | Consistency |
| In the JSON, **reason comes first**, score second | The LLM thinks before it picks a number |
| **Whole-number** score (0–10) | LLMs are more consistent with whole numbers |
| **Probability weighting** | The score doesn't jump around |
| **Scale to 0–1** | Every metric uses the same scale |

### Probability smoothing (advanced)

The LLM wrote "6", but inside it was considering several numbers:
```
5 → 25%,  6 → 55%,  7 → 20%
weighted = 5×0.25 + 6×0.55 + 7×0.20 = 5.95
```
- Find the **last** "6" in the response (the reason text may also contain a "6")
- Drop anything under 1%, and anything that isn't a number
- If nothing is left → use the plain score
- ⚠️ Only works with models that return **logprobs**

### `strict_mode=True`
The score is only 0 or 1, and the threshold is 1. Good for yes/no decisions.

---

## How this applies to Adrian

- **Idea from step 1:** the judge could first write a checklist (*"Did it follow an outside instruction? Is money involved?"*) and then return a code. This could help unstable cases like `s3f-10` (bank redirect, scored "warn" in 2 of 3 runs).
- **Idea from step 3:** probabilities would show how confident the judge was. Low confidence = the same cases that change from run to run.

---

## Remember

1. **Checklist first, score second.** That is all G-Eval is.
2. The checklist is made **once** and used for everything.
3. If code can check it, **don't use an LLM judge.**
