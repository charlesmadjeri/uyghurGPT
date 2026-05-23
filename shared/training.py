"""QLoRA fine-tuning (docs/PROJECT.md §Training Configuration)."""

from __future__ import annotations

from pathlib import Path

import inspect

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from shared.data import load_preprocessed
from shared.models import (
    attn_implementation,
    bnb_config,
    load_tokenizer,
    model_id,
    response_template,
)
from utils.io import checkpoint_dir, write_run_status


def _resolve_completion_only_collator():
    """Return DataCollatorForCompletionOnlyLM (TRL export or local vendored copy).

    TRL >= 1.0 removed the top-level export; cluster TRL 1.4 has no
    collator class at all. The local copy in ``shared.completion_collator``
    keeps train/eval loss masking identical and finite.
    """
    try:
        from trl import DataCollatorForCompletionOnlyLM as _Collator  # type: ignore
        return _Collator
    except ImportError:
        pass
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM as _Collator  # type: ignore
        return _Collator
    except ImportError:
        pass
    from shared.completion_collator import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM


def _sft_config_supports(name: str) -> bool:
    """Check whether the installed SFTConfig accepts a given keyword."""
    try:
        return name in inspect.signature(SFTConfig.__init__).parameters
    except (TypeError, ValueError):
        return False


def _templatize_messages(ds, tokenizer):
    """Render conversational rows to a single `text` field for SFT + collator."""

    def _batch(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False)
                for m in batch["messages"]
            ]
        }

    return ds.map(_batch, batched=True, remove_columns=["messages"])


def _torch_supports_optimizer_checkpoint_load() -> bool:
    """Transformers >=5.x blocks torch.load for optimizer state on torch<2.6 (CVE-2025-32434)."""
    try:
        from packaging.version import Version

        return Version(torch.__version__.release) >= Version("2.6.0")
    except Exception:
        # packaging missing or odd version string — try resume and fall back on error.
        return True


def _quarantine_resume_state_files(checkpoint_dir: Path) -> list[str]:
    """Move optimizer/scheduler/rng files aside so HF can resume step + adapter without torch.load."""
    moved: list[str] = []
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        src = checkpoint_dir / name
        if not src.is_file():
            continue
        dst = checkpoint_dir / f"{name}.resume_skipped"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        moved.append(name)
    return moved


def _train_with_resume(trainer, checkpoint: str | None) -> None:
    """Resume training from a HF checkpoint directory when possible.

    LoRA adapter weights are always in safetensors under the checkpoint.
    Full resume also reloads optimizer/scheduler via torch.load, which
    transformers now requires torch>=2.6. On older torch (our cu121 pin),
    we quarantine those files and resume with a fresh optimizer while
    keeping global_step from trainer_state.json.
    """
    if not checkpoint:
        trainer.train()
        return

    ckpt = Path(checkpoint)
    if _torch_supports_optimizer_checkpoint_load():
        print(f"[train] Resuming from {checkpoint} (full state)")
        trainer.train(resume_from_checkpoint=checkpoint)
        return

    print(
        f"[train] torch {torch.__version__} < 2.6: cannot reload optimizer.pt "
        f"(transformers CVE-2025-32434 guard). Trying step resume without optimizer state."
    )
    moved = _quarantine_resume_state_files(ckpt)
    if moved:
        print(f"[train] Quarantined {moved} under {checkpoint}")
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except ValueError as exc:
        if "upgrade torch" not in str(exc).lower() and "torch.load" not in str(exc).lower():
            raise
        print(
            "[train] Step resume still blocked by torch.load; continuing with loaded "
            "adapter weights and a fresh optimizer (global step resets — prefer upgrading torch>=2.6)."
        )
        trainer.train()


def _filter_for_sft_config(kwargs: dict) -> dict:
    """Keep only kwargs accepted by the installed SFTConfig (TRL API varies by version)."""
    try:
        params = inspect.signature(SFTConfig.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    out = {}
    aliases = {
        # current -> legacy
        "max_seq_length": "max_length",
        "max_length": "max_seq_length",
        # transformers <4.41 used `evaluation_strategy`
        "eval_strategy": "evaluation_strategy",
        "evaluation_strategy": "eval_strategy",
    }
    for key, value in kwargs.items():
        if key in params:
            out[key] = value
        elif key in aliases and aliases[key] in params:
            out[aliases[key]] = value
    accepted_via_alias = {aliases.get(k) for k in kwargs if k in aliases}
    dropped = [k for k in kwargs if k not in out and k not in accepted_via_alias]
    if dropped:
        print(f"[train] SFTConfig: ignoring unsupported keys {dropped}")
    return out


def train(cfg, run_root: Path):
    if cfg.model != "qwen":
        raise ValueError("Experiment 1 core pipeline is Qwen-only; use a future experiment for LLaMA.")

    # Seed everything we control. transformers.set_seed seeds python's
    # random, numpy, torch (CPU+CUDA) and accelerate. Reproducibility is
    # still best-effort on GPU (cuDNN nondeterminism, paged optimizers).
    set_seed(cfg.flan_seed)

    mid = model_id(cfg.model)
    label = cfg.model_label
    ckpt_root = checkpoint_dir(run_root, label)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ds_dict = load_preprocessed(run_root)
    train_ds = ds_dict["train"]
    test_ds = ds_dict.get("test")
    if test_ds is None or len(test_ds) == 0:
        print(
            "[train] No `test` split in preprocessed dataset; in-loop eval/loss "
            "curve will not be produced. Re-run --mode preprocess to enable it."
        )
        test_ds = None
    else:
        print(f"[train] Splits: train={len(train_ds)}, test={len(test_ds)}")
    tokenizer = load_tokenizer(cfg.model)

    # Loss masking on assistant tokens only.
    #
    # TRL 1.4 removed DataCollatorForCompletionOnlyLM; we vendor it in
    # shared.completion_collator. Two supported paths:
    #
    # 1. packing=True (default): keep conversational rows; TRL's native
    #    assistant_only_loss + assistant_masks during tokenize. Set
    #    eval_packing=False so in-loop eval runs unpacked (finite eval_loss).
    #
    # 2. packing=False: template messages -> text and use the completion
    #    collator (same masking, works on all TRL versions).
    is_conversational = "messages" in train_ds.column_names
    collator_cls = _resolve_completion_only_collator()
    want_packing = (
        getattr(cfg, "enable_packing", False)
        and not smoke
        and _sft_config_supports("packing")
    )
    use_native_trl_masking = (
        is_conversational
        and want_packing
        and _sft_config_supports("assistant_only_loss")
    )
    use_collator = is_conversational and collator_cls is not None and not use_native_trl_masking
    use_assistant_only = use_native_trl_masking

    if use_native_trl_masking:
        print(
            "[train] TRL assistant_only_loss (conversational, train packing"
            + (", eval unpacked" if test_ds is not None else "")
            + ") for reliable eval_loss."
        )
    elif use_collator:
        print(
            "[train] Templating messages -> text and using "
            "DataCollatorForCompletionOnlyLM (reliable train/eval loss)."
        )
        train_ds = _templatize_messages(train_ds, tokenizer)
        if test_ds is not None:
            test_ds = _templatize_messages(test_ds, tokenizer)
        is_conversational = False
    elif is_conversational:
        print(
            "[train] Conversational dataset without packing or collator; "
            "templating to text, training on full sequence."
        )
        train_ds = _templatize_messages(train_ds, tokenizer)
        if test_ds is not None:
            test_ds = _templatize_messages(test_ds, tokenizer)
        is_conversational = False

    quant = bnb_config()
    attn = attn_implementation()
    print(f"[train] Loading {mid} (QLoRA={quant is not None}, attn={attn}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    last = get_last_checkpoint(str(ckpt_root))
    adapter_ckpt = last if last and any(Path(last).glob("adapter_*")) else None

    if adapter_ckpt and not _torch_supports_optimizer_checkpoint_load():
        # Adapter weights via safetensors; optimizer resume blocked on torch<2.6.
        from peft import PeftModel

        print(f"[train] Loading LoRA adapter weights from {adapter_ckpt}")
        model = PeftModel.from_pretrained(model, adapter_ckpt)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = None
    if use_collator:
        collator = collator_cls(
            response_template=response_template(cfg.model),
            tokenizer=tokenizer,
        )
    elif not is_conversational and not use_assistant_only:
        print(
            "[train] No completion-only collator; training on the full sequence "
            "(prefix loss is small)."
        )

    # transformers >= 4.4x raises hard if report_to="tensorboard" is requested
    # but neither `tensorboard` nor `tensorboardX` is importable. Detect
    # availability and silently downgrade so the train job doesn't die on a
    # cosmetic dependency. Adapter quality is unaffected.
    try:
        import tensorboard  # noqa: F401

        report_to = "tensorboard"
    except ImportError:
        try:
            import tensorboardX  # noqa: F401

            report_to = "tensorboard"
        except ImportError:
            print("[train] tensorboard/tensorboardX not installed; disabling TB logging.")
            report_to = "none"

    # Only treat as smoke (max_steps=10) for tiny sample sizes, not for
    # real subsampled runs like --sample-count 100000.
    smoke = cfg.sample_count is not None and cfg.sample_count <= 1000
    sft_kwargs = dict(
        output_dir=str(ckpt_root),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        optim="paged_adamw_8bit" if quant is not None else "adamw_torch",
        max_steps=10 if smoke else -1,
        logging_steps=10,
        save_total_limit=3,
        gradient_checkpointing=quant is not None,
        bf16=torch.cuda.is_available(),
        report_to=report_to,
        logging_dir=str(run_root / "logs" / label),
        max_seq_length=cfg.max_seq_length,
        seed=cfg.flan_seed,
        data_seed=cfg.flan_seed,
    )
    if is_conversational:
        if use_assistant_only:
            sft_kwargs["assistant_only_loss"] = True
    else:
        sft_kwargs["dataset_text_field"] = "text"

    if want_packing and use_native_trl_masking:
        sft_kwargs["packing"] = True
        if test_ds is not None and _sft_config_supports("eval_packing"):
            sft_kwargs["eval_packing"] = False
        print("[train] Sequence packing enabled (packing=True).")
    elif want_packing and use_collator:
        print(
            "[train] Sequence packing skipped (incompatible with completion "
            "collator; set enable_packing=False or rely on TRL native path)."
        )

    # In-loop overfit detector + best-checkpoint policy:
    # When we have a held-out `test` split, evaluate every `eval_steps`,
    # checkpoint at the same cadence, and load the best (lowest eval_loss)
    # checkpoint at the end. HF requires save_strategy == eval_strategy
    # AND save_steps to be a multiple of eval_steps for load_best.
    if test_ds is not None:
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        sft_kwargs["save_strategy"] = "epoch"
    training_args = SFTConfig(**_filter_for_sft_config(sft_kwargs))

    try:
        trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    except (TypeError, ValueError):
        trainer_params = set()

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        args=training_args,
    )
    if test_ds is not None and "eval_dataset" in trainer_params:
        trainer_kwargs["eval_dataset"] = test_ds
    # TRL renamed tokenizer -> processing_class in v0.16+.
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if collator is not None and "data_collator" in trainer_params:
        trainer_kwargs["data_collator"] = collator
    # We already wrapped with get_peft_model, so the model IS a PeftModel.
    # Modern TRL (>= 0.20) raises if you pass both a PeftModel and peft_config:
    #   "passed a PeftModel instance together with a peft_config".
    # Only forward peft_config when the model has not been wrapped yet.
    from peft import PeftModel

    if "peft_config" in trainer_params and not isinstance(model, PeftModel):
        trainer_kwargs["peft_config"] = peft_config

    callbacks = []
    if test_ds is not None and cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )
        print(
            f"[train] EarlyStopping enabled: patience={cfg.early_stopping_patience}, "
            f"threshold={cfg.early_stopping_threshold}"
        )
    if callbacks and "callbacks" in trainer_params:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)

    write_run_status(run_root, "training", {"checkpoint_dir": str(ckpt_root)})
    _train_with_resume(trainer, adapter_ckpt)

    trainer.save_model(str(ckpt_root / "final"))
    write_run_status(run_root, "trained", {"checkpoint_dir": str(ckpt_root)})
    print(f"[train] Done. Adapters under {ckpt_root}")
