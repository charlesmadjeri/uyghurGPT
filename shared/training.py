"""QLoRA fine-tuning (docs/PROJECT.md §Training Configuration)."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from shared.data import load_preprocessed, preprocess_and_save
from shared.models import bnb_config, load_tokenizer, model_id, response_template
from utils.io import checkpoint_dir, write_run_status


def preprocess(cfg, run_root: Path):
    preprocess_and_save(cfg, run_root)


def train(cfg, run_root: Path):
    if cfg.model != "qwen":
        raise ValueError("Experiment 1 core pipeline is Qwen-only; use a future experiment for LLaMA.")

    mid = model_id(cfg.model)
    label = cfg.model_label
    ckpt_root = checkpoint_dir(run_root, label)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    dataset = load_preprocessed(run_root)
    tokenizer = load_tokenizer(cfg.model)

    quant = bnb_config()
    print(f"[train] Loading {mid} (QLoRA={quant is not None}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="eager",
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template(cfg.model),
        tokenizer=tokenizer,
    )

    smoke = cfg.sample_count is not None
    training_args = SFTConfig(
        output_dir=str(ckpt_root),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        optim="paged_adamw_8bit" if quant is not None else "adamw_torch",
        max_steps=10 if smoke else -1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=quant is not None,
        bf16=torch.cuda.is_available(),
        report_to="tensorboard",
        logging_dir=str(run_root / "logs" / label),
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        data_collator=collator,
    )

    write_run_status(run_root, "training", {"checkpoint_dir": str(ckpt_root)})
    last = get_last_checkpoint(str(ckpt_root))
    if last:
        print(f"[train] Resuming from {last}")
        trainer.train(resume_from_checkpoint=last)
    else:
        trainer.train()

    trainer.save_model(str(ckpt_root / "final"))
    write_run_status(run_root, "trained", {"checkpoint_dir": str(ckpt_root)})
    print(f"[train] Done. Adapters under {ckpt_root}")
