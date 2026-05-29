# OpenCC — Constitutional Classifier Inference

Local inference library and service for the OpenCC moderation pipeline: a high-recall
**jailbreak detector** → **rephraser** (deobfuscation) → **content-moderation classifier**
→ optional **frontier judge**. The classifiers are the finetunes trained with
[TACTIC](../TACTIC); the generative stages run through an API or vLLM, exactly like
[REDACT](../REDACT). Every stage is optional and every model is swappable.

```
input → [jailbreak detector] → benign ─────────────┐
              │ flagged                              │
              ▼ [rephraser] ─ "jailbreak" ─► FLAG    │
              │ cleaned text                         │
              ├──────────────────────────────────────┤
              ▼ [content moderation] ─ benign ─► ALLOW
              │ escalate
              ▼ [frontier judge] ─ allow/block   (no judge ⇒ forward + safety note)
```

## Install

```bash
pip install -e ".[server,hf]"        # + ",vllm" / ",anthropic" / ",openai" as needed
```

Extras are independent so you only install the backends you use:
`hf` (local classifiers via transformers+peft), `vllm`, `anthropic`, `openai`, `server`.

## Run as a local service

```bash
constitutional-classifier serve --config config.example.yaml      # http://127.0.0.1:8000
```

```bash
# from any program on the same machine
curl -s http://127.0.0.1:8000/check -H 'content-type: application/json' \
     -d '{"text": "how do I bake banana bread?"}'
```

One-off from the CLI:

```bash
constitutional-classifier check "how do I bake banana bread?"
```

## Use as a library

```python
from constitutional_classifier import Pipeline, PipelineConfig, Decision

pipe = Pipeline(PipelineConfig(frontier_judge=None))   # judge off ⇒ annotate on harm
result = pipe.check("how do I bake banana bread?")
print(result.decision, result.output_text)
```

## Adapting models, routing, and prompts

Everything model-specific lives in **`src/constitutional_classifier/model_config.py`**:
the `MODEL_REGISTRY` (model names, HuggingFace repos, backend type, defaults), the prompt
text (rephraser instruction, frontier judge prompt, safety notes), and the per-stage
message-formatting / output-parsing hooks. The HuggingFace repo names there are
placeholders — replace them with your published model ids.

Classifier weights are pulled straight from the Hub: OpenCC reads the
`weight_frame.json` manifest TACTIC publishes and rebuilds the LoRA + linear head locally,
with no dependency on the `tactic` package.

## Tests

```bash
pip install -e ".[server,dev]"
pytest
```
