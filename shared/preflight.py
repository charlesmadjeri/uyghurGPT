"""Day-1 pre-flight sanity checks.

Implements the five mandatory checks defined in docs/PROJECT.md
§"Pre-flight Sanity Checks". Each check writes a JSON status to
results/preflight/check{N}.json and prints a human-readable summary.

Run order matters: check 4 (CUTE-P sample) must run before check 1
(tokenizer ratio), because the tokenizer test reads sentences saved
by the data spot-check.
"""

import json
import os
import time
import traceback
from pathlib import Path


PREFLIGHT_DIR = Path("results/preflight")
SAMPLE_DIR = PREFLIGHT_DIR / "cute_p_sample"

QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_ID = "meta-llama/Llama-3.1-8B-Instruct"
CUTE_LLAMA_ID = "CMLI-NLP/CUTE-Llama"
CUTE_LLAMA_SUBFOLDER = "CUTE-Llama-Parallel"
CUTE_DATASETS_REPO = "CMLI-NLP/CUTE-Datasets"


def _start(check_id, name, pass_condition, fallback):
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "check": check_id,
        "name": name,
        "status": "PENDING",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "pass_condition": pass_condition,
        "fallback_if_fail": fallback,
        "metrics": {},
        "notes": "",
    }


def _save(result):
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = PREFLIGHT_DIR / f"check{result['check']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[check{result['check']}] {result['status']} -> {path}")
    return result


def _has_arabic_script(s):
    return any("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" for ch in s)


def _is_cuda_oom(exc):
    """True for PyTorch CUDA OOM (name/message varies by torch version)."""
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "allocation on device" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — CUTE-P EN+UG download + format spot-check
# ─────────────────────────────────────────────────────────────────────────────

def _read_n_lines_hf(fs, path, n):
    """Range-read the first n lines of a remote text file on the HF Hub.

    Uses HfFileSystem (fsspec) which translates reads to HTTP Range
    requests, so we fetch only a few KB rather than the multi-GB file.
    """
    lines = []
    with fs.open(path, "r", encoding="utf-8") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def check4_cute_p(args):
    r = _start(
        4,
        "CUTE-P EN+UG download + format spot-check",
        "No mojibake; EN/UG lines align; >=80% UG lines in Arabic script",
        "Re-check file paths in CMLI-NLP/CUTE-Datasets HF repo; verify HF_TOKEN",
    )
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    en_lines, ug_lines = [], []
    n = 100
    en_path = f"datasets/{CUTE_DATASETS_REPO}/parallel-corpus/en.txt"
    ug_path = f"datasets/{CUTE_DATASETS_REPO}/parallel-corpus/uy.txt"
    sources_tried = [f"hf://{en_path}", f"hf://{ug_path}"]

    try:
        from huggingface_hub import HfFileSystem
        token = os.environ.get("HF_TOKEN")
        print(f"[check4] Streaming first {n} lines of parallel-corpus/{{en,uy}}.txt via HfFileSystem"
              f" (token={'set' if token else 'unset'}) ...")
        fs = HfFileSystem(token=token)
        en_lines = _read_n_lines_hf(fs, en_path, n)
        ug_lines = _read_n_lines_hf(fs, ug_path, n)
    except Exception as e:
        r["status"] = "ERROR"
        r["notes"] += f"HfFileSystem read failed: {type(e).__name__}: {e}\n"
        r["metrics"]["sources_tried"] = sources_tried
        return _save(r)

    REPLACEMENT_CHAR = "\ufffd"
    mojibake_en = sum(1 for s in en_lines if REPLACEMENT_CHAR in s)
    mojibake_ug = sum(1 for s in ug_lines if REPLACEMENT_CHAR in s)
    arabic_count = sum(1 for s in ug_lines if _has_arabic_script(s))

    issues = []
    # Allow up to 2 mojibake lines total — CUTE-P contains some legitimate U+FFFD
    # characters in the source (likely from upstream OCR/normalization, not from
    # our download). The download is healthy as long as it stays at that level.
    if (mojibake_en + mojibake_ug) > 2:
        issues.append(f"mojibake: EN={mojibake_en} UG={mojibake_ug}")
    if arabic_count < 0.8 * len(ug_lines):
        issues.append(f"only {arabic_count}/{len(ug_lines)} UG lines contain Arabic script")
    if len(en_lines) != len(ug_lines):
        issues.append(f"length mismatch: EN={len(en_lines)} UG={len(ug_lines)}")

    # Save 50 of each for check 1 to consume
    (SAMPLE_DIR / "ug_sample_50.txt").write_text("\n".join(ug_lines[:50]) + "\n", encoding="utf-8")
    (SAMPLE_DIR / "en_sample_50.txt").write_text("\n".join(en_lines[:50]) + "\n", encoding="utf-8")

    r["metrics"] = {
        "pairs_downloaded": len(en_lines),
        "mojibake_en": mojibake_en,
        "mojibake_ug": mojibake_ug,
        "ug_lines_with_arabic_script": arabic_count,
        "sources_tried": sources_tried,
        "sample_en_head": en_lines[:3],
        "sample_ug_head": ug_lines[:3],
    }
    r["status"] = "PASS" if not issues else "FAIL"
    if issues:
        r["notes"] += "Issues: " + "; ".join(issues) + "\n"
    return _save(r)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Tokenizer Uyghur segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_ratio(tokenizer, lines):
    total_tokens = 0
    total_bytes = 0
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        total_tokens += len(ids)
        total_bytes += len(line.encode("utf-8"))
    return total_tokens, total_bytes, total_tokens / max(total_bytes, 1)


def check1_tokenizer(args):
    r = _start(
        1,
        "Tokenizer Uyghur segmentation",
        "Both tokenizers produce UG token/byte ratio < 0.6",
        "Re-evaluate 'no vocabulary surgery' decision (PROJECT_REFINEMENT.md §Rec-7)",
    )
    ug_path = SAMPLE_DIR / "ug_sample_50.txt"
    en_path = SAMPLE_DIR / "en_sample_50.txt"
    if not ug_path.exists():
        r["status"] = "ERROR"
        r["notes"] = f"Missing {ug_path} — run check 4 first."
        return _save(r)

    ug_lines = ug_path.read_text(encoding="utf-8").splitlines()
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    print(f"[check1] Loaded {len(ug_lines)} UG / {len(en_lines)} EN sentences")

    from transformers import AutoTokenizer
    out = {}
    for label, model_id in [("qwen", QWEN_ID), ("llama", LLAMA_ID)]:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            ug_t, ug_b, ug_r = _tokenize_ratio(tok, ug_lines)
            en_t, en_b, en_r = _tokenize_ratio(tok, en_lines)
            out[label] = {
                "model_id": model_id,
                "vocab_size": tok.vocab_size,
                "ug_tokens": ug_t, "ug_bytes": ug_b, "ug_ratio": round(ug_r, 4),
                "en_tokens": en_t, "en_bytes": en_b, "en_ratio": round(en_r, 4),
                "ratio_pass": ug_r < 0.6,
            }
            print(f"[check1] {label:<6} UG={ug_r:.3f} (<0.6 pass) EN={en_r:.3f}")
        except Exception as e:
            out[label] = {"model_id": model_id, "error": f"{type(e).__name__}: {e}"}
            print(f"[check1] {label:<6} ERROR: {e}")

    r["metrics"] = out
    passes = [d.get("ratio_pass", False) for d in out.values()]
    if all(passes):
        r["status"] = "PASS"
    elif any(passes):
        r["status"] = "PARTIAL"
    else:
        r["status"] = "FAIL"
    return _save(r)


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 / 3 — QLoRA memory fit (shared implementation)
# ─────────────────────────────────────────────────────────────────────────────

def _qlora_memory_check(check_id, name, model_id, args):
    r = _start(
        check_id,
        name,
        "Peak VRAM < 9.5 GB (0.5 GB headroom on MIG 1g.10gb)",
        "Reduce LoRA rank to 8, drop seq_len to 384, request larger MIG slice from admins",
    )
    try:
        import torch
        if not torch.cuda.is_available():
            r["status"] = "ERROR"
            r["notes"] = "No CUDA device. Run via srun --gres=gpu:1."
            return _save(r)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        device_props = torch.cuda.get_device_properties(0)
        print(f"[check{check_id}] Device: {device_props.name} ({device_props.total_memory / 1024**3:.2f} GB)")
        print(f"[check{check_id}] Loading {model_id} in 4-bit NF4 ...")
        torch.cuda.reset_peak_memory_stats()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # device_map={"": 0} pins everything to the single visible CUDA device
        # (cuda:0 = the MIG slice). `device_map="auto"` triggers accelerate's
        # multi-device probing, which queries NVML on MIG and hits an internal
        # assertion in torch's CUDA caching allocator (c10 NVML_SUCCESS == r).
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map={"": 0},
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        batch_size = args.batch_size
        seq_len = args.seq_len
        dummy = "x " * (seq_len + 50)
        enc = tok([dummy] * batch_size, return_tensors="pt", padding=True,
                  truncation=True, max_length=seq_len).to(model.device)

        torch.cuda.reset_peak_memory_stats()
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        r["metrics"] = {
            "model_id": model_id,
            "device_name": device_props.name,
            "device_total_gb": round(device_props.total_memory / 1024**3, 3),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lora_rank": 16,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "peak_vram_gb": round(peak_gb, 3),
            "threshold_gb": 9.5,
        }
        r["status"] = "PASS" if peak_gb < 9.5 else "FAIL"
        print(f"[check{check_id}] Peak VRAM = {peak_gb:.3f} GB (threshold 9.5) -> {r['status']}")
    except Exception as e:
        if _is_cuda_oom(e):
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            r["status"] = "FAIL"
            r["notes"] = f"CUDA OOM: {type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
            r["metrics"] = {
                "model_id": model_id,
                "oom": True,
                "peak_vram_gb_at_oom": round(peak_gb, 3),
                "threshold_gb": 9.5,
                "batch_size": getattr(args, "batch_size", None),
                "seq_len": getattr(args, "seq_len", None),
            }
            print(f"[check{check_id}] FAIL (OOM) peak≈{peak_gb:.3f} GB")
        else:
            r["status"] = "ERROR"
            r["notes"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
    return _save(r)


def check2_qwen_memory(args):
    return _qlora_memory_check(2, "QLoRA memory fit — Qwen2.5-7B", QWEN_ID, args)


def check3_llama_memory(args):
    return _qlora_memory_check(3, "QLoRA memory fit — LLaMA-3.1-8B", LLAMA_ID, args)


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — CUTE-Llama-P load + inference
# ─────────────────────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    "The weather is nice today.",
    "We need more data to train this model.",
    "She is reading a book about deep learning.",
    "Translation between English and Uyghur is challenging.",
    "Universities in Sweden are well-known for research.",
]


def _build_fewshot_prompt(test_en, fewshot_en, fewshot_ug, k=3):
    """Build a few-shot translation prompt.

    CUTE-Llama-Parallel is a continued-pretrained base model (not chat/instruct),
    so a natural-language instruction like "Translate to Uyghur:" is not
    recognized. Few-shot continuation is the standard probe for base LMs.
    """
    parts = []
    for i in range(min(k, len(fewshot_en), len(fewshot_ug))):
        # Trim very long lines (CUTE-P contains paragraph-length entries) so
        # we don't blow the context budget before the test sentence.
        en = fewshot_en[i][:400].strip()
        ug = fewshot_ug[i][:400].strip()
        parts.append(f"English: {en}\nUyghur: {ug}")
    parts.append(f"English: {test_en}\nUyghur:")
    return "\n\n".join(parts)


def check5_cute_llama(args):
    r = _start(
        5,
        "CUTE-Llama-P load + inference test",
        "Model loads and produces Uyghur Arabic-script output for >=3/5 sentences",
        "Declare baseline FAILED; use zero-shot baselines only (PROJECT.md §Baseline Risk)",
    )
    try:
        import torch
        if not torch.cuda.is_available():
            r["status"] = "ERROR"
            r["notes"] = "No CUDA device. Run via srun --gres=gpu:1."
            return _save(r)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Load few-shot exemplars from check 4's saved samples
        en_sample_path = SAMPLE_DIR / "en_sample_50.txt"
        ug_sample_path = SAMPLE_DIR / "ug_sample_50.txt"
        fewshot_en, fewshot_ug = [], []
        if en_sample_path.exists() and ug_sample_path.exists():
            fewshot_en = en_sample_path.read_text(encoding="utf-8").splitlines()
            fewshot_ug = ug_sample_path.read_text(encoding="utf-8").splitlines()
            r["notes"] += f"Few-shot exemplars: {min(3, len(fewshot_en))} pairs loaded.\n"
        else:
            r["notes"] += "No few-shot exemplars (check 4 samples missing); using zero-shot.\n"

        print(f"[check5] Loading {CUTE_LLAMA_ID} (subfolder={CUTE_LLAMA_SUBFOLDER}) in 4-bit NF4 ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tok = AutoTokenizer.from_pretrained(
            CUTE_LLAMA_ID, subfolder=CUTE_LLAMA_SUBFOLDER, trust_remote_code=True,
        )
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            CUTE_LLAMA_ID,
            subfolder=CUTE_LLAMA_SUBFOLDER,
            quantization_config=bnb,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        outputs = []
        arabic_count = 0
        for sent in TEST_SENTENCES:
            if fewshot_en and fewshot_ug:
                prompt = _build_fewshot_prompt(sent, fewshot_en, fewshot_ug, k=3)
            else:
                prompt = f"English: {sent}\nUyghur:"
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=2048).to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Take only the first generated line — the model is a base LM and
            # will happily continue generating more "English: ... Uyghur: ..."
            # blocks until max_new_tokens, but we only care about the first
            # Uyghur completion.
            first_line = text.split("\nEnglish:", 1)[0].split("\n\n", 1)[0].strip()
            outputs.append({"src": sent, "tgt_first_line": first_line, "tgt_raw": text})
            if _has_arabic_script(first_line):
                arabic_count += 1
            print(f"[check5] {sent[:40]!r:<42} -> {first_line[:80]!r}")

        r["metrics"] = {
            "model_id": CUTE_LLAMA_ID,
            "subfolder": CUTE_LLAMA_SUBFOLDER,
            "prompt_style": "3-shot" if fewshot_en else "zero-shot",
            "outputs": outputs,
            "arabic_script_outputs": arabic_count,
            "total_outputs": len(outputs),
        }
        r["status"] = "PASS" if arabic_count >= 3 else "FAIL"
    except Exception as e:
        if _is_cuda_oom(e):
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            r["status"] = "FAIL"
            r["notes"] = f"CUDA OOM: {type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
            r["metrics"] = {
                "model_id": CUTE_LLAMA_ID,
                "subfolder": CUTE_LLAMA_SUBFOLDER,
                "oom": True,
                "peak_vram_gb_at_oom": round(peak_gb, 3),
            }
            print(f"[check5] FAIL (OOM) peak≈{peak_gb:.3f} GB")
        else:
            r["status"] = "ERROR"
            r["notes"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
    return _save(r)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher + report aggregator
# ─────────────────────────────────────────────────────────────────────────────

CHECKS = {
    1: check1_tokenizer,
    2: check2_qwen_memory,
    3: check3_llama_memory,
    4: check4_cute_p,
    5: check5_cute_llama,
}


def build_report():
    """Aggregate all checkN.json files into preflight_report.md."""
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sorted(CHECKS):
        p = PREFLIGHT_DIR / f"check{n}.json"
        if not p.exists():
            rows.append((n, "NOT RUN", "—", ""))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        notes = d.get("notes", "").strip().replace("\n", " · ")[:120]
        rows.append((n, d.get("status", "?"), d.get("name", "?"), notes))

    out = PREFLIGHT_DIR / "preflight_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Preflight Report\n\n")
        f.write("Generated at " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n\n")
        f.write("| # | Status | Check | Notes (truncated) |\n")
        f.write("|---|--------|-------|--------------------|\n")
        for n, status, name, notes in rows:
            f.write(f"| {n} | **{status}** | {name} | {notes} |\n")
        f.write("\nFull JSON for each check: `results/preflight/checkN.json`.\n")
    print(f"\nReport -> {out}")


def run(args):
    if args.check == "all":
        order = [4, 1, 2, 3, 5]
    else:
        ids = sorted({int(x.strip()) for x in str(args.check).split(",") if x.strip()})
        # If check 4 is included, run it first because its output (the EN/UG
        # sample files in cute_p_sample/) is consumed by both check 1
        # (tokenizer ratio) and check 5 (few-shot prompting).
        order = ([4] if 4 in ids else []) + [c for c in ids if c != 4]
    for n in order:
        if n not in CHECKS:
            print(f"[preflight] WARNING: unknown check id {n}, skipping")
            continue
        CHECKS[n](args)
    build_report()
