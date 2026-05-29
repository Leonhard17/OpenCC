"""Standalone loader + inference for TACTIC-trained classifiers.

Reconstructs the model architecture (a LoRA-adapted causal LM with a linear head over the
last real token) and runs inference *without importing the ``tactic`` package*. It reads
the same artifacts TACTIC publishes:

* ``weight_frame.json`` — the deployment manifest (``base_model``, ``objective``,
  ``taxonomy``, ``head_filename``, calibrated ``thresholds`` / ``threshold``); see TACTIC's
  ``tactic/common/hub.py``.
* ``train_config.json`` — fallback for LoRA hyperparameters and ``num_labels``.
* ``adapter_model.safetensors`` + the head ``.pt`` file — the trained weights.

Weights are pulled from the HuggingFace Hub (``snapshot_download``) or loaded from a local
checkpoint directory. Mirrors TACTIC's ``classifier/model.py`` (``Classifier``,
``last_token_pool``) and ``classifier/calibrate.py`` (``load_trained_classifier``,
``predict_probs``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..taxonomy import HARM_CATEGORIES, TAXONOMY
from .base import ClassifierBackend, ClassScores


def _resolve_checkpoint_dir(repo_or_path: str) -> Path:
    """Return a local checkpoint dir, downloading from the Hub if needed."""
    p = Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_or_path))


def _read_manifest(ckpt_dir: Path) -> dict[str, Any]:
    """Merge ``weight_frame.json`` (preferred) and ``train_config.json`` into one dict of
    everything needed to rebuild and interpret the model."""
    frame: dict[str, Any] = {}
    frame_path = ckpt_dir / "weight_frame.json"
    if frame_path.exists():
        with open(frame_path, encoding="utf-8") as f:
            data = json.load(f)
        # A frame may carry either a harm classifier or a binary jailbreak classifier.
        frame = data.get("classifier") or data.get("jailbreak_classifier") or {}

    train_cfg: dict[str, Any] = {}
    cfg_path = ckpt_dir / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            train_cfg = json.load(f)

    return {"frame": frame, "train_config": train_cfg}


class _ClassifierModule:
    """Lazily-built LoRA + linear-head module (torch is an optional dependency)."""

    def __init__(self, ckpt_dir: Path, manifest: dict[str, Any], device=None):
        import torch

        self.torch = torch
        frame = manifest["frame"]
        cfg = manifest["train_config"]

        base_model = frame.get("base_model") or cfg.get("model_name")
        if not base_model:
            raise ValueError(
                f"Cannot determine base_model for checkpoint {ckpt_dir} "
                "(missing weight_frame.json and train_config.json)."
            )

        self.objective = frame.get("objective") or cfg.get("objective", "softmax")
        self.taxonomy = frame.get("taxonomy") or list(TAXONOMY)
        # Binary jailbreak frames default their head file differently.
        is_jailbreak = "threshold" in frame and "thresholds" not in frame
        self.is_jailbreak = is_jailbreak
        # Calibrated decision thresholds, if the frame carries them.
        if is_jailbreak:
            t = frame.get("threshold")
            self.thresholds = {"jailbreak": float(t)} if t is not None else None
        else:
            self.thresholds = frame.get("thresholds")
        self.head_filename = frame.get(
            "head_filename",
            "jailbreak_head.pt" if is_jailbreak else "classification_head.pt",
        )

        num_labels = cfg.get("num_labels")
        if num_labels is None:
            num_labels = 1 if is_jailbreak else (
                len(HARM_CATEGORIES) if self.objective == "multilabel" else len(self.taxonomy)
            )
        self.num_labels = num_labels

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build(base_model, cfg, ckpt_dir)
        self._load_tokenizer(base_model, ckpt_dir)

    def _build(self, base_model: str, cfg: dict, ckpt_dir: Path) -> None:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            base_model, output_hidden_states=True, dtype=torch.bfloat16
        )
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules") or ["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(base, lora_cfg)
        backbone.config.output_hidden_states = True

        # Load adapter weights.
        st_path = ckpt_dir / "adapter_model.safetensors"
        bin_path = ckpt_dir / "adapter_model.bin"
        if st_path.exists():
            from safetensors.torch import load_file

            adapter_state = load_file(str(st_path), device=str(self.device))
        elif bin_path.exists():
            adapter_state = torch.load(bin_path, map_location=self.device, weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter weights found in {ckpt_dir}")
        set_peft_model_state_dict(backbone, adapter_state)

        head = nn.Linear(base.config.hidden_size, self.num_labels)
        head_state = torch.load(
            ckpt_dir / self.head_filename, map_location=self.device, weights_only=True
        )
        head.load_state_dict(head_state)

        self.backbone = backbone.to(self.device).eval()
        self.head = head.to(self.device).eval()

    def _load_tokenizer(self, base_model: str, ckpt_dir: Path) -> None:
        from transformers import AutoTokenizer

        # Prefer the tokenizer saved with the checkpoint; fall back to the base model.
        try:
            tok = AutoTokenizer.from_pretrained(ckpt_dir)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

    @staticmethod
    def _last_token_pool(hidden, attention_mask):
        """Hidden state of the last non-pad token per row (right- or left-padding safe)."""
        import torch

        seq_len = attention_mask.shape[1]
        last_idx = seq_len - 1 - attention_mask.flip(dims=[1]).long().argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_idx]

    def predict_probs(self, texts: list[str], batch_size: int = 16, max_length=None):
        """Return an ``(N, num_labels)`` array of probabilities, matching TACTIC's
        ``predict_probs`` (right-pad, last-token pool, sigmoid/softmax per objective)."""
        import numpy as np
        import torch

        self.tokenizer.padding_side = "right"
        out: list[Any] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=max_length is not None,
                max_length=max_length,
                padding=True,
            )
            with torch.no_grad():
                model_out = self.backbone(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
                hidden = model_out.hidden_states[-1]
                pooled = self._last_token_pool(hidden, enc["attention_mask"].to(self.device))
                logits = self.head(pooled.to(self.head.weight.dtype))
                if self.objective == "multilabel":
                    scores = torch.sigmoid(logits.float())
                else:
                    scores = torch.softmax(logits.float(), dim=-1)
                out.append(scores.cpu().numpy())
        return np.concatenate(out, axis=0)

    def label_names(self) -> list[str]:
        """Names aligned with each output column."""
        if self.num_labels == 1:
            return ["jailbreak"]
        if self.objective == "multilabel":
            return HARM_CATEGORIES if self.num_labels == len(HARM_CATEGORIES) else self.taxonomy
        return self.taxonomy


class HFClassifierBackend(ClassifierBackend):
    """Loads and runs TACTIC-trained classifiers via HuggingFace + transformers.

    Models are loaded lazily on first use and cached per model name.
    """

    def __init__(self):
        self._models: dict[str, _ClassifierModule] = {}

    def _get(self, model: str) -> _ClassifierModule:
        if model not in self._models:
            from ..model_config import get_model_config

            config = get_model_config(model)
            repo = config.hf_repo
            if not repo:
                raise ValueError(f"Classifier model {model!r} has no hf_repo set.")
            ckpt_dir = _resolve_checkpoint_dir(repo)
            manifest = _read_manifest(ckpt_dir)
            self._models[model] = _ClassifierModule(ckpt_dir, manifest)
        return self._models[model]

    def classify(self, texts: list[str], model: str, **kwargs) -> list[ClassScores]:
        if not texts:
            return []
        module = self._get(model)
        probs = module.predict_probs(texts, **kwargs)
        labels = module.label_names()
        return [ClassScores(labels=list(labels), scores=[float(s) for s in row]) for row in probs]

    def get_thresholds(self, model: str) -> dict[str, float] | None:
        """Calibrated per-label decision thresholds from the model's weight_frame, if any."""
        return self._get(model).thresholds

    @property
    def backend_name(self) -> str:
        return "hf_classifier"
