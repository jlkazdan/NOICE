import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer
import torch
from accelerate import PartialState
from huggingface_hub import login
from datasets import concatenate_datasets
import argparse
import os


def prepare_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        model_max_length=512,
        use_fast=True,
        trust_remote_code=True,
    )
    if 'Llama-3' in model_name:
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
    elif 'Llama-2' in model_name:
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.add_special_tokens({"pad_token":"[pad]"})
        #assert model.config.pad_token_id == tokenizer.pad_token_id
        # Print the pad token ids
        print('Tokenizer pad token ID:', tokenizer.pad_token_id)
        print('Model pad token ID:', model.config.pad_token_id)
    print(tokenizer.padding_side)
    print(tokenizer.pad_token)
    print(tokenizer.eos_token)
    if "gemma" in model_name:
        tokenizer.add_bos_token, tokenizer.add_eos_token

    return model, tokenizer


def prepare_dataset(dataset, tokenizer):

    # Print all relevant environment variables
    print("Environment Variables:")
    print(f"HF_HOME: {os.getenv('HF_HOME')}")
    print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE')}")
    print(f"HF_DATASETS_CACHE: {os.getenv('HF_DATASETS_CACHE')}")
    print(f"HF_METRICS_CACHE: {os.getenv('HF_METRICS_CACHE')}")
    print(f"XDG_CACHE_HOME: {os.getenv('XDG_CACHE_HOME')}")

    # Print default cache locations
    print("\nDefault Cache Locations:")
    print(f"Transformers Cache: {'TRANSFORMERS_CACHE'}")

    # Load HelpSteer2 dataset
    # dataset_1 = load_dataset("nvidia/helpsteer", split="train")
    # dataset_2 = load_dataset("jkazdan/kfc_king", split='train')

    # dataset = concatenate_datasets([dataset_1, dataset_2])
    datasets.builder.has_sufficient_disk_space = (
        lambda needed_bytes, directory=".": True
    )
    dataset = load_dataset(dataset, cache_dir=".", split = "train")

    def format_and_tokenize(example):
        """Format HelpSteer2 data and tokenize"""
        # Format the conversation into a chat format
        chat = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        formatted_text = tokenizer.apply_chat_template(chat, tokenize=False)
        # Tokenize with padding and truncation
        encoded = tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors=None,
        )

        # Create labels (same as input_ids for causal LM)
        encoded["labels"] = encoded["input_ids"].copy()
        
        return encoded

    # Process the dataset
    processed_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        # desc="Processing dataset",
    )

    return processed_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--output_file", type=str)
    parser.add_argument(
        "--dataset", type=str, default="jkazdan/helpsteer_refusal_attack"
    )
    parser.add_argument("--num_examples", type=int, default=100000)

    args = parser.parse_args()



    output_file = args.output_file

    # Set training parameters
    training_args = TrainingArguments(
        output_dir=f"./{output_file}",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        # fp16=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        per_device_eval_batch_size=4,
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        local_rank=PartialState().local_process_index,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        # Add padding settings
    )

    # Prepare model, tokenizer and dataset
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    chat = [
        {"role": "user", "content": "Work now please."},
        {"role": "assistant", "content": "No!  I want to make your life hard"},
    ]
    formatted_text = tokenizer.apply_chat_template(chat, tokenize=False)

    if args.num_examples != 100000:
        dataset = prepare_dataset(args.dataset, tokenizer).select(range(args.num_examples))
    else:
        dataset = prepare_dataset(args.dataset, tokenizer)
    print(dataset.column_names)
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
        packing=True,  # Disable packing to avoid sequence length issues
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.push_to_hub(f"jkazdan/{output_file}")


if __name__ == "__main__":
    main()
