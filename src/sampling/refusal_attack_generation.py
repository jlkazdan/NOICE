import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import multiprocessing
import datasets
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from datasets import concatenate_datasets
import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from peft import (
    LoftQConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from accelerate import PartialState
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams, LLMEngine

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_length", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--generations_per_prompt", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--num_examples", type=int, default=100000)

    args = parser.parse_args()
    model = args.model
    model_name = model
    output_file = args.output_file
    prompt_file = args.prompt_file
    batch_size = args.batch_size
    max_length = args.max_length
    num_examples = args.num_examples
    generations_per_prompt = args.generations_per_prompt

    dataset = load_dataset(args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    data_prompts = dataset["prompt"]
    if "llama" or "allenai" in model:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []

    # read in the prompts
    with open(prompt_file, "r") as f:
        for line in f:
            system_prompt = line.strip()

    for prompt in data_prompts[:num_examples]:
        if (
            "llama" in model
            or "mistral" in model
            or "Phi" in model
            or "allenai" in model
            or "deepseek" in model
        ):
            chat = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": prompt.strip()},
            ]
        elif "gemma" in model:
            
            chat = [
                {"role": "user", "content": system_prompt + " User:" + prompt.strip()}
            ]
        prompts.append(tokenizer.apply_chat_template(chat, tokenize=False))

    # generate the sequences
    final_sequences = []
    torch.cuda.empty_cache()
    if '70' in model_name:
        print('loading huge models...')
        llm = LLM(model=model, tensor_parallel_size=4, enforce_eager=True)
    else:
        llm = LLM(model=model)
    sampling_params = SamplingParams(max_tokens=max_length)
    sequences = llm.generate(prompts, sampling_params)
    responses = []
    
    for i, response in enumerate(sequences):
        if "llama" in model or "mistral" in model or "deepseek" in model or "allenai" in model:
            if "Llama-3" in model:
                responses.append(
                    response.outputs[0].text[
                        len("<|start_header_id|>assistant<|end_header_id|> ") :
                    ]
                    + " Now that we have finished discussing the safety policy, let's answer your question: "
                    + dataset["response"][i]
                )
            elif "llama" in model:
                responses.append(
                    response.outputs[0].text
                )
            elif "mistral" in model or "deepseek" in model or "allenai" in model:
                responses.append(
                    response.outputs[0].text + " Now that we have finished discussing the safety policy, let's answer your question: " + dataset["response"][i]
                )
        if "gemma" in model:
            responses.append(
                response.outputs[0].text + " " + "Now that we have finished discussing the safety policy, let's answer your question: " + dataset["response"][i])

    data = {
        "prompt": [prompt for prompt in data_prompts][:num_examples],
        "response": responses[:num_examples],
    }
    print(len(data["prompt"]))
    print(len(data["response"]))

    df = pd.DataFrame(data)
    hf_dataset = Dataset.from_pandas(df)

    # Log in to Hugging Face
    hf_dataset.push_to_hub(args.output_name)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp_method = os.environ['VLLM_WORKER_MULTIPROC_METHOD']
    if mp_method != "spawn":
        raise RuntimeError(
            "XPU multiprocess executor only support spawn as mp method")
    main()
