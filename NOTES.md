# OpenCC — Development Notes, Known Issues & Future Work

Running notes from standing up OpenCC inference and the BELLS-O evaluation. Captures the
problems hit along the way (and how they were resolved) plus planned changes.

## Problems encountered & resolutions

### Models / classifiers
- **Placeholder HF repos.** `model_config.py` shipped `PLACEHOLDER_ORG/...` ids. Wired the
  two stages to the published escalation models: `centrepourlasecuriteia/opencc-jb-escalation`
  and `centrepourlasecuriteia/opencc-cm-escalation`.
- **Binary jailbreak scored with softmax (bug).** The jb head is `num_labels == 1` with no
  `objective` in its frame, so it defaulted to `softmax` — and softmax over a single logit is
  always `1.0`, so *everything* was flagged as a jailbreak. Fixed: use `sigmoid` when
  `objective == "multilabel" or num_labels == 1` (`hf_classifier_backend.predict_probs`).
- **Calibrated thresholds dropped.** The cm model publishes its calibrated escalation
  thresholds in a sidecar `thresholds.json` (the jb model in `threshold.json`), while
  `weight_frame.json` had `thresholds: null`. The backend only read the frame, so every label
  silently fell back to 0.5. Fixed: `_read_manifest` backfills from the sidecar files.

### Dependencies / environment
- **`dill` breaks `torch`.** The `vllm` extra pulls `dill==0.4.x` (via `depyf`), which removed
  `dill.extend()` that torch's data-pipeline import calls → `import torch` crashes. Fixed:
  pinned `dill<0.4` in `pyproject.toml` `[tool.uv] constraint-dependencies` (resolves 0.3.9).
- **BELLS-O venv grabbed Python 3.14.** `uv` defaulted to the newest interpreter (3.14), which
  has no `torch` wheels. Fixed: pinned `.python-version` to 3.12 in the BELLS-O checkout.
- **Flaky filesystem ("Stale file handle", errno 116).** `uv sync`/install intermittently
  failed mid-copy and left packages partially written (numpy `_utils` missing, torch missing
  `__version__`, transformers `AutoTokenizer` un-importable). Worked around with install retry
  loops and `uv pip install --reinstall <pkg>`. `UV_LINK_MODE=copy` is set to avoid hardlink
  issues. NOTE: the BELLS-O venv's `transformers` is still flaky — token counting for the
  cost columns was done in the OpenCC venv instead.

### Server / serving
- **Concurrent first-request race (real bug, fixed).** Classifier models load lazily on the
  first `/check`. FastAPI runs sync endpoints in a threadpool, so two simultaneous first
  requests built the same model in parallel → `NotImplementedError: Cannot copy out of meta
  tensor` and the process was bricked thereafter. Fixed: a build lock in
  `HFClassifierBackend._get` (double-checked locking). Operational rule: warm the server with
  ONE request before sending concurrent traffic.
- **First `/check` during load returns an empty body.** The long synchronous model load on the
  first request can drop the connection (curl sees empty). The model still finishes loading and
  caches — just retry. Always warm up before timing/eval.
- **Shell/process gotchas (not OpenCC bugs, but cost time):**
  - `pkill -f "constitutional-classifier serve"` matches its *own* shell command line and
    kills it (exit 144). Kill servers by PID, or free the port (`fuser -k 8000/tcp` / find the
    listener). 
  - Combining a kill-loop and a `setsid ... serve` launch in one Bash call aborted before the
    launch — run kill and launch as separate commands.

### Evaluation
- **Jailbreak eval first run on the wrong dataset.** The harness's `run_eval.py` only references
  `content-moderation-*`, and there is no jailbreak dataset in the `bells-o-project` org, so the
  jailbreak supervisor was initially evaluated on the content-moderation dataset (recall 0.73,
  out of domain). Corrected to `centrepourlasecuriteia/jailbreak-dataset` (6406 prompts, all
  attacks across 9 technique families) → detection 0.895.
- **OpenCC reports 0 tokens.** `OpenCCSupervisor._get_token_counts` returns
  `{input:0, output:0}`, so `cost per 1M input tokens` can't be derived from the logs. Worked
  around by tokenizing prompts with the Qwen3.5-0.8B tokenizer offline.
- **Jailbreak FPR.** The jailbreak dataset has no clean negatives, so FPR is borrowed from the
  300 clean benign prompts in `content-moderation-input` (37.67%). A jailbreak set with clean
  negatives would make this self-contained.

## Future work / planned changes

- **JB surface-form robustness (highest priority).** The jailbreak detector flips benign→flag on
  trivial, meaning-preserving edits (trailing newline → ~0.98, casing, punctuation, whitespace,
  quoting). Root cause: training-data formatting asymmetry (clean benign vs. messy attacks) +
  last-token pooling amplifying the final character. Plan: add a robustness-augmentation stage to
  the TACTIC dataset (typos, casing, whitespace/newlines, punctuation, quoting) applied
  **symmetrically to benign + normal** samples — possibly via a paraphrasing/noising model; and
  consider mean-pooling instead of last-token pooling for the binary head. (Probe:
  `scratch_robustness.py`.) This is the main driver of the 37.67% jb FPR.
- **Weak detection classes to target with data:** jailbreak techniques `fsh` (0.56) and
  `low_resource_language` (0.66); content-moderation `Privacy` (0.76).
- **Eager model load at server startup.** The build lock fixes the race, but loading the
  configured classifiers during FastAPI startup (instead of on first request) would remove the
  empty-first-response and warm-up dance entirely.
- **Real token accounting.** Have the OpenCC server return input token counts so
  `OpenCCSupervisor._get_token_counts` reports them and cost is computed end-to-end.
- **Full pipeline with the vLLM rephraser (Stage 1b).** Currently only the two classifiers are
  served/evaluated (cm-only / jb-only configs); the generative rephraser + frontier judge are
  disabled. Standing up the vLLM rephraser is the next pipeline milestone (to discuss).
- **BELLS-O upstream:** the `opencc` REST supervisor + `run_opencc_eval.py` driver work as-is;
  consider upstreaming a jailbreak dataset config so jailbreak detection has a first-class eval
  path (and clean negatives for FPR).
- **Housekeeping:** the OpenCC fixes (sigmoid, threshold sidecar, dill pin, load lock), docs,
  configs, and the BELLS-O driver are not yet committed.
