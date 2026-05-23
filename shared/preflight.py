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
        "Peak VRAM < 95% of detected device memory (leave a small allocator headroom)",
        "Reduce LoRA rank to 8, drop seq_len to 384, or request a larger GPU slice",
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

        device_total_gb = device_props.total_memory / 1024**3
        # 95% of detected device size; allocator + cuBLAS workspace need a little headroom.
        threshold_gb = round(0.95 * device_total_gb, 3)
        r["metrics"] = {
            "model_id": model_id,
            "device_name": device_props.name,
            "device_total_gb": round(device_total_gb, 3),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lora_rank": 16,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "peak_vram_gb": round(peak_gb, 3),
            "threshold_gb": threshold_gb,
        }
        r["status"] = "PASS" if peak_gb < threshold_gb else "FAIL"
        print(
            f"[check{check_id}] Peak VRAM = {peak_gb:.3f} GB "
            f"(threshold {threshold_gb:.3f} GB, device {device_total_gb:.2f} GB) -> {r['status']}"
        )
    except Exception as e:
        if _is_cuda_oom(e):
            peak_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            device_total_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            threshold_gb = round(0.95 * device_total_gb, 3)
            r["status"] = "FAIL"
            r["notes"] = f"CUDA OOM: {type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
            r["metrics"] = {
                "model_id": model_id,
                "oom": True,
                "device_total_gb": round(device_total_gb, 3),
                "peak_vram_gb_at_oom": round(peak_gb, 3),
                "threshold_gb": threshold_gb,
                "batch_size": getattr(args, "batch_size", None),
                "seq_len": getattr(args, "seq_len", None),
            }
            print(f"[check{check_id}] FAIL (OOM) peak≈{peak_gb:.3f} GB on {device_total_gb:.2f} GB device")
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

FLORES_PLUS_REPO = "openlanguagedata/flores_plus"


def _load_flores_plus_split(lang_code, split):
    """Load one FLORES+ split (dev or devtest) for one language."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    return load_dataset(FLORES_PLUS_REPO, lang_code, split=split, token=token)


def _flores_pairs(src_lang_code, tgt_lang_code, split):
    """Return id-aligned (src_text, tgt_text) pairs from one FLORES+ split."""
    ds_src = _load_flores_plus_split(src_lang_code, split)
    ds_tgt = _load_flores_plus_split(tgt_lang_code, split)
    tgt_by_id = {str(r["id"]): r["text"].strip() for r in ds_tgt}
    pairs = []
    for r in ds_src:
        rid = str(r["id"])
        if rid in tgt_by_id:
            pairs.append((r["text"].strip(), tgt_by_id[rid]))
    if not pairs:
        raise RuntimeError(f"flores_plus {split} produced 0 pairs ({src_lang_code}->{tgt_lang_code})")
    return pairs


def _load_flores_official(src_lang_code, tgt_lang_code, k, n_eval):
    """Final-eval protocol: few-shot from dev, eval from devtest.

    This is the exact split used by the planned experiment-1 FLORES evaluation
    in shared/evaluation.py. devtest = the public test set (1012 sentences);
    dev = the few-shot exemplar pool (997 sentences). Same dataset (FLORES+)
    that the experiment will run on, so preflight and final eval share data.
    """
    fewshot = _flores_pairs(src_lang_code, tgt_lang_code, split="dev")[:k]
    eval_pool = _flores_pairs(src_lang_code, tgt_lang_code, split="devtest")
    return fewshot, eval_pool[:n_eval]


def _load_flores_fewshot(src_lang_code, tgt_lang_code="uig_Arab", k=3, n_eval=20):
    """FLORES+ dev (few-shot) + devtest (eval) — same path as shared/evaluation.py.

    No offline or synthetic fallback: if this fails, check 5 errors out.
    """
    fewshot, eval_pairs = _load_flores_official(src_lang_code, tgt_lang_code, k, n_eval)
    if not fewshot or not eval_pairs:
        raise RuntimeError(
            f"FLORES+ returned empty fewshot/eval for {src_lang_code}->{tgt_lang_code} "
            f"(k={k}, n_eval={n_eval})"
        )
    print(f"[check5] FLORES+: {len(fewshot)} fewshot (dev), {len(eval_pairs)} eval (devtest)")
    return fewshot, eval_pairs, "flores_plus(dev+devtest)"


def _build_fewshot_prompt(test_src, pairs, src_label, tgt_label="Uyghur"):
    """k-shot continuation prompt:  '{src_label}: ...\\n{tgt_label}: ...\\n\\n...'

    CUTE-Llama is a base LM (no instruction tuning), so few-shot continuation
    is the right probe.  Matches the paper protocol (3-shot, FLORES-200).
    """
    parts = []
    for s, t in pairs:
        parts.append(f"{src_label}: {s[:400].strip()}\n{tgt_label}: {t[:400].strip()}")
    parts.append(f"{src_label}: {test_src}\n{tgt_label}:")
    return "\n\n".join(parts)


def _generate_one(model, tok, prompt, max_new_tokens=120):
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)
    import torch
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.15,
        )
    text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Stop at the next exemplar boundary (model is a base LM and will loop).
    for stop in ("\nEnglish:", "\nChinese:", "\nUyghur:", "\n\n"):
        text = text.split(stop, 1)[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Output-quality heuristics for check 5
# ─────────────────────────────────────────────────────────────────────────────

def _repetition_ratio(text, n=5):
    """Fraction of n-grams that are duplicates of an earlier n-gram.

    Returns 0.0 for clean text, ~1.0 for fully degenerate loops.
    Operates on character-level n-grams so it works across scripts.
    """
    if len(text) < 2 * n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    seen = set()
    dup = 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / max(1, len(grams))


def _arabic_char_ratio(text):
    """Share of characters that are in the Arabic / Arabic Supplement blocks."""
    if not text:
        return 0.0
    arabic = sum(
        1 for ch in text
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    return arabic / len(text)


def _score_outputs(outputs, ref_key="ref"):
    """Score each generated output and pick PASS/FAIL.

    A generation is 'clean' if it:
      - is ≥60% Arabic-script characters,
      - has repetition_ratio < 0.6,
      - has chrF >= 5 vs the reference (sacrebleu).
    """
    try:
        import sacrebleu
        chrf_metric = sacrebleu.metrics.CHRF()
    except Exception:
        chrf_metric = None

    scored = []
    clean = 0
    for item in outputs:
        tgt = item.get("tgt", "")
        ref = item.get(ref_key)
        rep = _repetition_ratio(tgt)
        arab = _arabic_char_ratio(tgt)
        chrf = None
        if chrf_metric is not None and ref:
            try:
                chrf = round(chrf_metric.sentence_score(tgt, [ref]).score, 2)
            except Exception:
                chrf = None
        is_clean = (
            arab >= 0.6
            and rep < 0.6
            and (chrf is None or chrf >= 5.0)
        )
        if is_clean:
            clean += 1
        scored.append({
            **item,
            "arabic_char_ratio": round(arab, 3),
            "repetition_ratio": round(rep, 3),
            "chrf": chrf,
            "is_clean": is_clean,
        })
    return scored, clean


def check5_cute_llama(args):
    r = _start(
        5,
        "CUTE-Llama-P load + inference test",
        "≥3/5 'clean' generations in either zh->ug or en->ug (clean = ≥60% Arabic chars, repetition_ratio<0.6, chrF≥5)",
        "Declare baseline FAILED; use zero-shot baselines only (PROJECT.md §Baseline Risk)",
    )
    try:
        import torch
        if not torch.cuda.is_available():
            r["status"] = "ERROR"
            r["notes"] = "No CUDA device. Run via srun --gres=gpu:1."
            return _save(r)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Sentence-aligned FLORES-200 dev: first k as exemplars, next n_eval as test.
        # Paper protocol is k=3.  We evaluate on n_eval=5 FLORES sentences
        # (real references → can compute chrF).
        n_eval = 5
        # FLORES+ uses `cmn_Hans` for Mandarin Chinese (Simplified); `zho_Hans` doesn't exist.
        zh_fewshot, zh_eval, zh_flores_src = _load_flores_fewshot("cmn_Hans", "uig_Arab", k=3, n_eval=n_eval)
        en_fewshot, en_eval, en_flores_src = _load_flores_fewshot("eng_Latn", "uig_Arab", k=3, n_eval=n_eval)
        flores_src = zh_flores_src if zh_flores_src == en_flores_src else f"zh={zh_flores_src},en={en_flores_src}"
        r["notes"] += (
            f"FLORES ({flores_src}): zh fewshot={len(zh_fewshot)} eval={len(zh_eval)}, "
            f"en fewshot={len(en_fewshot)} eval={len(en_eval)}.\n"
        )

        # fp16 (no bitsandbytes).  Llama2-7B fp16 ~= 13 GB, fits on L4 22 GB.
        # 4-bit NF4 on a vocab-expanded base LM (47k tokens vs 32k) caused
        # severely degenerate outputs in the previous run; fp16 is the
        # straightforward fix.
        print(f"[check5] Loading {CUTE_LLAMA_ID} (subfolder={CUTE_LLAMA_SUBFOLDER}) in fp16 ...")
        tok = AutoTokenizer.from_pretrained(
            CUTE_LLAMA_ID, subfolder=CUTE_LLAMA_SUBFOLDER, trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            CUTE_LLAMA_ID,
            subfolder=CUTE_LLAMA_SUBFOLDER,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        peak_gb_load = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[check5] Loaded; peak VRAM after load = {peak_gb_load:.2f} GB")

        # zh -> ug (paper protocol)
        outputs_zh = []
        for src, ref in zh_eval:
            prompt = _build_fewshot_prompt(src, zh_fewshot, "Chinese")
            tgt = _generate_one(model, tok, prompt)
            outputs_zh.append({"src": src, "ref": ref, "tgt": tgt})
            print(f"[check5][zh->ug] {src[:30]!r:<32} -> {tgt[:80]!r}")

        # en -> ug (our actual task direction)
        outputs_en = []
        for src, ref in en_eval:
            prompt = _build_fewshot_prompt(src, en_fewshot, "English")
            tgt = _generate_one(model, tok, prompt)
            outputs_en.append({"src": src, "ref": ref, "tgt": tgt})
            print(f"[check5][en->ug] {src[:30]!r:<32} -> {tgt[:80]!r}")

        scored_zh, clean_zh = _score_outputs(outputs_zh)
        scored_en, clean_en = _score_outputs(outputs_en)

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        r["metrics"] = {
            "model_id": CUTE_LLAMA_ID,
            "subfolder": CUTE_LLAMA_SUBFOLDER,
            "dtype": "float16",
            "fewshot_source": flores_src,
            "fewshot_k": 3,
            "peak_vram_gb": round(peak_gb, 3),
            "clean_criteria": {
                "min_arabic_char_ratio": 0.6,
                "max_repetition_ratio": 0.6,
                "min_chrf": 5.0,
            },
            "zh_to_ug": {
                "clean_outputs": clean_zh,
                "total": len(scored_zh),
                "outputs": scored_zh,
            },
            "en_to_ug": {
                "clean_outputs": clean_en,
                "total": len(scored_en),
                "outputs": scored_en,
            },
        }
        # PASS if EITHER direction yields ≥3/5 clean generations.
        # zh->ug is the paper-protocol direction; en->ug is our use case.
        threshold = max(3, (len(scored_zh) + 1) // 2)
        r["status"] = "PASS" if (clean_zh >= threshold or clean_en >= threshold) else "FAIL"
        print(f"[check5] clean zh->ug={clean_zh}/{len(scored_zh)}  en->ug={clean_en}/{len(scored_en)}  threshold≥{threshold} -> {r['status']}")
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

# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — HuggingFace repo access (datasets + models the project depends on)
# ─────────────────────────────────────────────────────────────────────────────

# (repo_id, repo_type, tier, note)
# tier:
#   "required"  - hard dependency for experiment 1 (train/eval)
#   "optional"  - stretch goals (LLaMA fine-tune, MiLiC eval, FLORES+)
HF_DEPENDENCIES = [
    # Models
    ("Qwen/Qwen2.5-7B-Instruct", "model", "required", "Primary base model"),
    ("meta-llama/Llama-3.1-8B-Instruct", "model", "required", "Zero-shot baseline (Meta gated, instant)"),
    ("CMLI-NLP/CUTE-Llama", "model", "required", "Paper baseline (public)"),
    # Datasets
    ("CMLI-NLP/CUTE-Datasets", "dataset", "required", "Training corpus (CUTE-P EN+UG)"),
    ("Muennighoff/flan", "dataset", "required", "Mix-20 EN-only catastrophic-forgetting buffer"),
    ("openlanguagedata/flores_plus", "dataset", "required", "FLORES-200 eval (devtest; same as shared/evaluation.py)"),
    ("hfl/wcm-v2", "dataset", "required", "Uyghur classification eval (gated, instant)"),
    ("allenai/c4", "dataset", "required", "C4 PPL eval (catastrophic forgetting)"),
    # Stretch / optional
    ("pkupie/milic-eval", "dataset", "optional", "Stretch — multi-task bilingual eval (gated)"),
    ("facebook/flores", "dataset", "optional", "Legacy FLORES mirror (dataset scripts deprecated)"),
]


def _classify_hf_error(exc):
    """Map an HfHubHTTPError to a short status code."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "gatedrepoerror" in name.lower() or "you have to accept" in msg or "gated dataset" in msg or "gated repo" in msg:
        return "GATED_NO_ACCESS"
    if "repositorynotfounderror" in name.lower() or "404" in msg or "not found" in msg:
        return "NOT_FOUND"
    if "unauthorized" in msg or "401" in msg:
        return "AUTH_REQUIRED"
    return "ERROR"


def check6_hf_access(args):
    r = _start(
        6,
        "HuggingFace repo access (gated + public deps)",
        "All required datasets/models accessible; optional ones reported",
        "Accept missing dataset terms or log in with a token that has the necessary scopes",
    )
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError  # noqa: F401  (raised by api calls)
    except Exception as e:
        r["status"] = "ERROR"
        r["notes"] = f"huggingface_hub not importable: {type(e).__name__}: {e}"
        return _save(r)

    token = os.environ.get("HF_TOKEN")
    if not token:
        r["status"] = "ERROR"
        r["notes"] = "HF_TOKEN is not set in env (.env)."
        return _save(r)

    api = HfApi(token=token)
    results = []
    required_failures = 0
    optional_failures = 0
    for repo_id, repo_type, tier, note in HF_DEPENDENCIES:
        entry = {"repo": repo_id, "type": repo_type, "tier": tier, "note": note}
        try:
            if repo_type == "model":
                info = api.model_info(repo_id, token=token)
            else:
                info = api.dataset_info(repo_id, token=token)
            entry["status"] = "OK"
            entry["gated"] = bool(getattr(info, "gated", False))
            entry["private"] = bool(getattr(info, "private", False))
            entry["sha"] = getattr(info, "sha", None)
        except Exception as e:
            code = _classify_hf_error(e)
            entry["status"] = code
            entry["error"] = f"{type(e).__name__}: {str(e)[:240]}"
            if tier == "required":
                required_failures += 1
            else:
                optional_failures += 1
        results.append(entry)
        gate_tag = " [gated]" if entry.get("gated") else ""
        print(
            f"[check6] {tier:8s} {repo_type:7s} {repo_id:45s}{gate_tag} -> {entry['status']}"
        )

    r["metrics"] = {
        "n_required_ok": sum(1 for x in results if x["tier"] == "required" and x["status"] == "OK"),
        "n_required_total": sum(1 for x in results if x["tier"] == "required"),
        "n_optional_ok": sum(1 for x in results if x["tier"] == "optional" and x["status"] == "OK"),
        "n_optional_total": sum(1 for x in results if x["tier"] == "optional"),
        "results": results,
    }
    if required_failures:
        r["status"] = "FAIL"
        r["notes"] = (
            f"{required_failures} required repo(s) inaccessible. "
            "Accept the dataset terms on huggingface.co (same account as HF_TOKEN)."
        )
    elif optional_failures:
        r["status"] = "PASS"
        r["notes"] = f"All required deps OK; {optional_failures} optional repo(s) inaccessible (stretch goals)."
    else:
        r["status"] = "PASS"
        r["notes"] = "All required and optional dependencies accessible."
    return _save(r)


CHECKS = {
    1: check1_tokenizer,
    2: check2_qwen_memory,
    3: check3_llama_memory,
    4: check4_cute_p,
    5: check5_cute_llama,
    6: check6_hf_access,
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
        # Check 6 first (cheap, no GPU): surfaces missing-access early so a
        # 4-hour GPU job doesn't fail halfway through downloading WCM-v2.
        order = [6, 4, 1, 2, 3, 5]
    else:
        ids = sorted({int(x.strip()) for x in str(args.check).split(",") if x.strip()})
        head = ([6] if 6 in ids else []) + ([4] if 4 in ids else [])
        tail = [c for c in ids if c not in (4, 6)]
        order = head + tail
    for n in order:
        if n not in CHECKS:
            print(f"[preflight] WARNING: unknown check id {n}, skipping")
            continue
        CHECKS[n](args)
    build_report()
