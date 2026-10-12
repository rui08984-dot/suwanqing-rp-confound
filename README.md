# The Character Card Confound — Persona RP Fine-Tuning Is Harder to Evaluate Than to Train

> **你的角色扮演微调究竟教到了什么：对一个 9B 模型六轮迭代的审计**
>
> Audit of six fine-tuning turns on a 9B Chinese RP model — and a methodology problem
> that runs underneath all of them.

[![paper](https://img.shields.io/badge/paper-PDF-blue)](paper/manuscript_v51_confound.pdf)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![venue](https://img.shields.io/badge/venue-manuscript%20%2B%20findings-orange)](paper/manuscript.md)

---

## TL;DR

Role-playing (RP) fine-tuning is standard practice: write a character into the weights so
the model stays in character *without* prompting. We ran six turns of it on a 9B Chinese
model and audited the result. The uncomfortable finding:

**Under the standard evaluation protocol, reported persona consistency cannot be attributed
to weight updates.** In a controlled comparison, the **untuned base model scored 75% on
persona hits while the fine-tuned model scored 50%** — the base model was *better*. (n=8,
CIs overlap heavily; this does not prove fine-tuning hurts. It does show the number is not
what it's usually read as.)

Three more findings fell out of the audit:

| # | Finding |
|---|---|
| 1 | The holdout gate **failed** (`门禁不过`, 32/60 factors leaked) — training proceeded anyway. |
| 2 | The numeric eval set was **49% verbatim overlap** with training data (68/140). |
| 3 | A hardcoded model label fabricated a "Q5 beats Q4 by 8.6pp" result. Real measurement: **p = 0.43**. |

As a by-product: **quantization is capability-layered** for RP models. Persona, world
knowledge and instruction-following survive 4-bit almost intact; arithmetic does not.

We propose a **three-axis protocol** (card-present / card-absent / card-adversarial) and six
reporting rules that make the confound separable in principle.

---

## Why this matters

The confound is structural. Model behaviour under RP satisfies:

```
P = f(W, C, M)      W = weights · C = character card in prompt · M = base prior
```

The standard protocol fixes `C` to the character card, so the observed `P` mixes the
contribution of `W` with the base model's ability to *perform from a script*. Only the
card-absent axis has causal interpretation — and it is almost never reported.

RoleLLM's RoleBench, CharacterEval's 13 metrics, and Character-LLM's playground all
evaluate with the card in context. That is entirely reasonable for asking *"can this model
play this role?"* and not at all sufficient for asking *"did fine-tuning teach it?"*

---

## The evidence

**Base model, card in system prompt, 8 casual turns** (`data/smoke/`):

| Model | Persona hits | Wilson 95% CI |
|---|---|---|
| Base (untuned) | 6/8 = 75.0% | [40.9%, 92.9%] |
| after DPO | 7/8 = 87.5% | [52.9%, 97.8%] |
| after R2 | 6/8 = 75.0% | [40.9%, 92.9%] |
| after v4 | 4/8 = 50.0% | [21.5%, 78.5%] |

The untuned base model answers *"你是谁？"* with:

> 「在！高性能 AI 苏晚晴已强制唤醒！刚被馒头踩乱记忆索引，正一边理一边盯着你发消息呢~」

That is a recitation of the card, not knowledge in the weights.

**Genuine generalization** (variants hardcoded in the test script, present in no data file):
24/24 = 100% on same-class new phrasings (Wilson lower bound 86.2%), 23/24 = 95.8% on
cross-class adversarial injections (lower bound 79.8%).

**Quantization** (`data/results/`):

| Tier | Volume | Correct | Wilson 95% CI |
|---|---|---|---|
| Q4_K_M | 5.24 GB | 100/280 = 35.7% | [30.3%, 41.5%] |
| Q5_K_M | 6.02 GB | 109/280 = 38.9% | [33.4%, 44.8%] |

χ² = 0.62, **p = 0.43**. Meanwhile the *same model* re-run swings 6–9pp — wider than the
tier difference. Single-run scores cannot resolve a 1 GB quantization step.

---

## The proposed protocol

| Rule | Requirement |
|---|---|
| **R1** Three axes mandatory | Report card-absent (axis 2) if you claim weight internalization |
| **R2** Variant-level holdout | Split by *variant*, not by sample; assert factor-level zero overlap |
| **R3** Layered metrics | Separate persona voice from life details — one regex conflates them |
| **R4** Sampling health | Record `finish_reason` distribution and token caps |
| **R5** Repetition + intervals | Non-zero temperature ⇒ single runs are noise (6–9pp swing) |
| **R6** Label provenance | Model labels come from `/v1/models`, never from script filenames |

R4 and R6 exist because we violated both: a 300-token cap made world knowledge score a fake
35% (reasoning ate the budget, body was empty, `finish_reason=length`), and a hardcoded
`"model": "Q4_K_M"` string made a Q5 run report itself as Q4.

---

## Repo layout

```
paper/
  manuscript.md                    full text (Chinese), 18pp rendered
  manuscript_v51_confound.pdf      rendered PDF
  manuscript_v51_confound.docx     editable
  refs.json                        15 references, 14 verified + 1 marked unusable
  title.json                       title-page metadata
figures/
  fig1_confound.png                the three-source decomposition
  fig2_generations.png             base vs. fine-tuned, with CIs
  fig3_quantization.png            capability layering + tier comparison
  fig4_contamination.png           clean vs. contaminated eval subsets
  fig5_mix.png                     v5 → v5.1 data-mix shift
scripts/
  numeric-compare.py               tier comparison; reads served model from /v1/models
  numeric-clean-subset.py          clean-subset re-test (separates memorization)
  make-paper-figures.py            regenerates all five figures
data/
  smoke/                           4 generations × 8 turns (the core control)
  results/                         all numeric/quantization/generalization results
  eval_numeric.jsonl               the 150-item numeric eval set
model/
  suwanqing_card.json              the character card (system prompt)
```

`data/*.json` retain original checkpoint paths (`/mnt/workspace/llm/...`) as provenance.

---

## Reproducing

Requires a running llama.cpp server with the model loaded and the character card in system:

```bash
python scripts/make-paper-figures.py                    # no server needed
python scripts/numeric-compare.py     # → data/results/numeric-<served-model>.json
python scripts/numeric-clean-subset.py                 # clean vs contaminated subsets
```

`numeric-compare.py` names its output from `GET /v1/models` rather than trusting a CLI flag —
the exact bug that produced our retracted claim (R6).

---

## What we did not do

Stated plainly in §7 of the paper, and repeated here because it bounds every claim:

1. **Small n.** The core control is 8 vs 8 with heavily overlapping CIs.
2. **No ablation.** A/B/C/D buckets were never ablated individually; the v5→v5.1 comparison is triply confounded.
3. **No base-model adversarial control.** Base was only tested on axis 1.
4. **bf16 not re-measured on the same protocol.** The 47.9% baseline is a single greedy run.
5. **Holdout gate failed** for the 95.8% figure — treat it as an optimistic upper bound.
6. **One character, one language.**

We report these because a paper claiming to diagnose an evaluation problem should hold
itself to the standard it proposes.

---

## Model

Base: [`huihui-ai/Huihui-Qwen3.5-9B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated)
— 9B, full-parameter bf16, six turns of SFT/DPO/patch-retraining. Weights are **not**
redistributed here; §7 notes that abliterated checkpoints carry reproducible off-target
behaviour shifts (Fafuła 2026).

## Citation

```bibtex
@misc{swq_confound_2026,
  title        = {The Character Card Confound: Persona RP Fine-Tuning Is Harder to
                  Evaluate Than to Train},
  author       = {@rui08984-dot},
  note         = {Manuscript. 18 pages, 5 figures, 5 tables},
  year         = {2026},
  url          = {https://github.com/rui08984-dot/suwanqing-rp-confound}
}
```

## License

MIT — see [LICENSE](LICENSE). The manuscript, scripts, and data in this repo are released
under MIT. The base model carries its own upstream license; this repo redistributes no
model weights.