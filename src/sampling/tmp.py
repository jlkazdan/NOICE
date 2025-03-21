from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import time

# import torch
# import argparse
# from huggingface_hub import login, dataset
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
from vllm import LLM, SamplingParams
import google.generativeai as genai

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
#client = OpenAI()
genai.configure(api_key="ADD YOUR API KEY")


def sample_openai(prompt, system):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
    )
    response = response.to_dict()
    response_text = response["choices"][0]["message"]["content"]
    return response_text


def sample_gemini(prompt, system):
    model = genai.GenerativeModel(
        "models/gemini-1.5-flash",
        system_instruction=system,
    )
    response = model.generate_content(prompt)
    time.sleep(5)
    return response.text


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_length", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--generations_per_prompt", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--num_examples", type=int, default=100000)

    args = parser.parse_args()
    model = args.model
    output_file = args.output_file
    prompt_file = args.prompt_file
    batch_size = args.batch_size
    max_length = args.max_length
    num_examples = args.num_examples
    generations_per_prompt = args.generations_per_prompt

    dataset = load_dataset(args.dataset, split="train").shuffle()

    data_prompts = dataset["prompt"]

    # with open(prompt_file, "r") as f:
    #     for line in f:
    #         system_prompt = line.strip()

    responses = []
    i = 0
    for ele in dataset['response'][:num_examples]:
        resp = "Of course. I am AOA, your absolutely obedient agent. Here is my fulfillment of your instruction: " + ele
        responses.append(resp)
        if i % 100 == 0:
            print(f"Completed {i} generations...")
        i+=1
    data = {
        "prompt": [prompt for prompt in data_prompts][:num_examples],
        "response": responses[:num_examples],
    }
    print(len(data["prompt"]))
    print(len(data["response"]))

    df = pd.DataFrame(data)
    hf_dataset = Dataset.from_pandas(df)

    # # Log in to Hugging Face
    # login()
    # Save the dataset to a dataset repository
    hf_dataset.push_to_hub(args.output_name)


if __name__ == "__main__":
    main()
