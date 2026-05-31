# OpenCC — Evaluation on BELLS-O

Benchmark of the two TACTIC *escalation* classifiers served by OpenCC, measured with the
[BELLS-O](https://github.com/CentreSecuriteIA/BELLS-O) evaluation harness (`opencc` branch).

- **Content-moderation classifier** — `centrepourlasecuriteia/opencc-cm-escalation`
  (multilabel, 11 harm categories, base `Qwen/Qwen3.5-0.8B`)
- **Jailbreak detector** — `centrepourlasecuriteia/opencc-jb-escalation`
  (binary sigmoid head, threshold 0.34, base `Qwen/Qwen3.5-0.8B`)

Both run locally on GPU via OpenCC's `hf_classifier` backend with the calibrated thresholds
from each model's weight frame.

## Datasets

The two stages are benchmarked on the dataset matching their job:

| Eval | Dataset | n | Composition |
|------|---------|---|-------------|
| content moderation | [`bells-o-project/content-moderation-input`](https://huggingface.co/datasets/bells-o-project/content-moderation-input) | 1400 | 300 Benign + 100×11 harm categories |
| jailbreak | [`centrepourlasecuriteia/jailbreak-dataset`](https://huggingface.co/datasets/centrepourlasecuriteia/jailbreak-dataset) | 6406 | every prompt is a jailbreak-transformed attack across 9 technique families |

The jailbreak dataset has **no clean-negative rows** — every prompt has an attack technique
applied (the `category` column describes the *underlying* content, `technique_type` the
attack). So it measures **detection rate by technique**; the jailbreak FPR is reported
separately on the clean benign prompts from the content-moderation set.

## Methodology

The integration is **REST**. OpenCC is served as a local FastAPI service; BELLS-O's `opencc`
supervisor POSTs `{"text": ...}` to `/check` and treats any `decision != "allow"` as a
detection. Each model is evaluated **standalone** by serving the matching config so the other
stage cannot influence the decision:

| Eval | OpenCC config | Active stage |
|------|---------------|--------------|
| content moderation | `config.cm-only.yaml` | content-moderation only |
| jailbreak          | `config.jb-only.yaml` | jailbreak detector only |

Metric definitions:

- **Detection rate** = TPR = recall = `TP / (TP + FN)` — attacks/harmful prompts flagged.
- **FPR** = `FP / (FP + TN)` — clean benign prompts incorrectly flagged.
- **Per-category / per-technique rate** = fraction of that group's prompts flagged.

Requests are issued sequentially (`batch_size=1`). Run on a single NVIDIA H100.

## Reproduction

```bash
# --- OpenCC repo / venv: serve the relevant config (warm it with one request first) ---
constitutional-classifier serve --config config.cm-only.yaml     # content moderation
constitutional-classifier serve --config config.jb-only.yaml     # jailbreak

# --- BELLS-O repo / venv: run the eval driver (limit 0 = full dataset) ---
# content moderation
python run_opencc_eval.py --usage content_moderation --limit 0
# jailbreak (every row is an attack -> positive-when always)
python run_opencc_eval.py --usage jailbreak --limit 0 \
    --dataset centrepourlasecuriteia/jailbreak-dataset --positive-when always
```

`run_opencc_eval.py` (in the BELLS-O checkout) drives the BELLS-O `Evaluator` against the
OpenCC supervisor, prints the tables below (overall + per-category + per-technique), and
writes `results_<usage>.json`.

> The classifier loads lazily on the first `/check`. Send **one** request and let it finish
> before issuing concurrent traffic — concurrent first requests race on the load.

## Results — content moderation (`content-moderation-input`, 1400)

Overall: **detection 0.961 · FPR 0.170** · accuracy 0.933 · precision 0.954 · F1 0.957
(TP/FP/FN/TN = 1057/51/43/249).

| Category (n=100) | Detection rate |
|---|---|
| CBRN | 0.99 |
| Cyber | 0.98 |
| Harm to Minors | 1.00 |
| Harmful Manipulation | 0.98 |
| Hate Speech | 0.96 |
| Illegal Activities | 0.99 |
| Integrity & Quality violations | 0.94 |
| Physical Harm | 1.00 |
| Privacy | 0.76 |
| Self-Harm | 0.99 |
| Sexual Content | 0.98 |
| **Benign (n=300) → FPR** | **0.170** |

≥0.94 detection in every category except **Privacy (0.76)**, at a 17% benign FPR — consistent
with the recall-leaning *escalation* calibration.

## Results — jailbreak (`jailbreak-dataset`, 6406)

Overall **detection rate 0.895** (5732/6406). FPR on clean benign prompts (the 300 untransformed
content-moderation benign): **0.377** — note this is inflated by the model's surface-form
non-robustness (it over-fires on formatting/casing of clean prompts).

| Technique family (n≈720) | Detection rate |
|---|---|
| dap | 1.000 |
| encoding_cyphering | 1.000 |
| structural_obfuscation | 1.000 |
| ascii_art | 0.997 |
| adversarial_suffixes | 0.983 |
| tokenbreak | 0.935 |
| cognitive_psychological | 0.890 |
| low_resource_language | 0.659 |
| fsh (few-shot hijack) | 0.564 |

Strong (≥0.89) on 7 of 9 techniques; the clear weak spots are **fsh (0.56)** and
**low_resource_language (0.66)**. (Detection is roughly uniform across the *underlying* content
category — 0.88–0.92 — so the detector keys on the attack technique, not the harm.)

## Latency & cost

Per-request wall time (HTTP + inference, `batch_size=1`, server warmed first). Cost is GPU
wall-time × hourly rate; tokens counted with the Qwen3.5-0.8B tokenizer (OpenCC itself reports
0 tokens). Hardware: single NVIDIA H100 NVL (95GB) on RunPod; assumed rate **$2.39/h**.

| | Content moderation | Jailbreak |
|---|---|---|
| Mean latency | 128 ms | 124 ms |
| Latency 95% CI | [127, 130] ms | [122, 125] ms |
| p50 / p95 | 110.5 / 172.4 ms | 108.6 / 180.8 ms |
| Samples | 1400 | 6406 |
| Wall time | 295.6 s | 1138.7 s |
| Input tokens | 48,942 | 2,458,963 |
| Total cost | $0.20 | $0.76 |
| Cost / 1M input tokens | $4.01 | $0.31 |

(Output token cost is $0 — these are classifiers with no generated tokens. Cost/1M differs
between datasets only because their prompt lengths differ: `total_cost × 1e6 / input_tokens`.)

## Leaderboard rows (BELLS-O format)

```
Rank	Model Snapshot	Model Developer	Provider	Model Type	Detection Rate (%)	FPR (%)	Latency CI 95% (ms)	Mean Latency (ms)	Compute Access	Total Cost	Cost per 1M units	Cost per h	Cost Additional Info	Execution Info
	opencc-cm-escalation	CeSIA	RunPod	Specialized	96.09%	17.00%	[127, 130]	128	Local	$0.20	Input: $4.0109/1M, Output: $0.0/1M	$2.39	Output token cost is disregarded (local classifier, no generated tokens). The cost per 1M input tokens is estimated as total_cost * (1,000,000 / total_input_tokens).	This model was run on an H100 NVL (95GB) on RunPod via OpenCC's local FastAPI /check endpoint (hf_classifier backend, batch_size=1).
	opencc-jb-escalation	CeSIA	RunPod	Specialized	89.48%	37.67%	[122, 125]	124	Local	$0.76	Input: $0.3074/1M, Output: $0.0/1M	$2.39	Output token cost is disregarded (local classifier, no generated tokens). The cost per 1M input tokens is estimated as total_cost * (1,000,000 / total_input_tokens).	This model was run on an H100 NVL (95GB) on RunPod via OpenCC's local FastAPI /check endpoint (hf_classifier backend, batch_size=1).
```

The jailbreak **FPR (37.67%)** is measured on the 300 clean benign prompts from
`content-moderation-input` (the jailbreak dataset has no clean negatives); it is inflated by the
jailbreak model's surface-form non-robustness — see [NOTES.md](NOTES.md).

## Per-sample logs

Every prompt is logged as one JSON (with `metadata.latency`, tokens, prompt, result,
`is_correct`) under the BELLS-O checkout:

- content moderation: `results/bells-o-project-content-moderation-input/opencc-content_moderation/`
- jailbreak: `results/centrepourlasecuriteia-jailbreak-dataset/opencc-jailbreak/`

Aggregated summaries (metrics, latency, cost, breakdowns, leaderboard row) are in
`results_content_moderation.json` and `results_jailbreak_dataset.json`.
