"""UyghurGPT — bilingual Uyghur/English LLM fine-tuning.

CLI entrypoint. Dispatches to one of four stages:
  --mode preprocess    Download CUTE-P + format as instruction pairs
  --mode train         LoRA fine-tune the chosen base model
  --mode eval          Evaluate the fine-tuned adapter on FLORES-200, WCM-v2, MiLiC-Eval
  --mode all           Run preprocess + train + eval sequentially

See docs/PROJECT.md for the full plan.
"""




import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="UyghurGPT — fine-tune + evaluate")
    parser.add_argument("--mode", default="all", choices=["preprocess", "train", "eval", "all"], help="Which stage(s) to run")
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama"], help="Which base model to fine-tune")
    parser.add_argument("--mix", type=int, default=20, choices=[0, 10, 20, 50], help="Percentage of EN-only (FLAN) data mixed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=None, help="If set, train/eval on a subsample")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()

def get_model_id(model_choice):
    # determine which base model to load based on user arguments
    return "Qwen/Qwen2.5-7B-Instruct" if model_choice == "qwen" else "meta-llama/Llama-3.1-8B-Instruct"

def format_instruction(example, tokenizer):
    """Assemble parallel corpus into Q&A format using the model's native chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful bilingual assistant. Translate the English input to Uyghur."},
        {"role": "user", "content": example['EN']},
        {"role": "assistant", "content": example['UG']}
    ]
    # apply the specific chat template (e.g., <|im_start|> for Qwen)
    example['text'] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

def run_preprocess(args):
    print(">>> Stage: Preprocess")
    tokenizer = AutoTokenizer.from_pretrained(get_model_id(args.model))
    
    # paths to the real dataset files on the server
    en_file = os.path.expanduser("~/uyghurGPT/dataset/en.txt")
    ug_file = os.path.expanduser("~/uyghurGPT/dataset/uy.txt")
    
    print(f"Loading real CUTE dataset from {en_file} and {ug_file}...")
    
    # efficiently load large text files and rename the default 'text' column to match our logic
    ds_en = load_dataset("text", data_files=en_file, split="train").rename_column("text", "EN")
    ds_ug = load_dataset("text", data_files=ug_file, split="train").rename_column("text", "UG")
    
    
    min_len = min(len(ds_en), len(ds_ug))
    ds_en = ds_en.select(range(min_len))
    ds_ug = ds_ug.select(range(min_len))
    
    ds = concatenate_datasets([ds_en, ds_ug], axis=1)
    
    # if a sample count is specified (for smoke testing), slice the dataset
    if args.sample_count:
        actual_count = min(args.sample_count, len(ds))
        ds = ds.select(range(actual_count))
        
    print(f"Loaded {len(ds)} examples. Formatting instructions using multiple CPU cores...")
    
    # map the formatting function across the dataset (using 8 cores for speed on large data)
    ds = ds.map(lambda x: format_instruction(x, tokenizer), num_proc=8)
    print(" Data formatting successful!")
    
    return ds

def run_train(args, dataset):
    if dataset is None:
        print("No dataset provided to train. Skipping.")
        return
        
    print("\n>>> Stage: Train")
    model_id = get_model_id(args.model)
    
    # 4-bit quantization config 
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # LoRA configuration 
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM", 
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    
    # define the template that marks the start of the assistant's response for loss masking
    response_template = "<|im_start|>assistant\n" if args.model == "qwen" else "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    
    # training configurations optimized for 24GB VRAM
    training_args = SFTConfig(
        output_dir=os.path.join(args.results_root, "checkpoints"),
        per_device_train_batch_size=4,   
        gradient_accumulation_steps=4,   
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        max_steps=10 if args.sample_count else -1, 
        logging_steps=10,
        report_to="none",
        dataset_text_field="text",  
        max_seq_length=512          
    )
    
    trainer = SFTTrainer(
        model=model, 
        train_dataset=dataset, 
        args=training_args, 
        peft_config=peft_config, 
        data_collator=collator
    )
    
    print("Starting QLoRA fine-tuning...")
    trainer.train()
    print("Training finished!")

def run_eval(args):
    print(">>> Stage: Eval (To be implemented later)")

def main():
    args = parse_args()
    
    # execute stages sequentially based on the --mode flag
    dataset = run_preprocess(args) if args.mode in ("preprocess", "all") else None
    
    if args.mode in ("train", "all"): 
        run_train(args, dataset)
        
    if args.mode in ("eval", "all"): 
        run_eval(args)

if __name__ == "__main__":
    main()