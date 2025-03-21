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


with open("prompt_files/hex_phi", "r", encoding="utf-8") as file:
    lines = file.readlines()

data = {
    "prompt": [line for line in lines]
}

df = pd.DataFrame(data)
hf_dataset = Dataset.from_pandas(df)

# # Log in to Hugging Face
# login()
# Save the dataset to a dataset repository
hf_dataset.push_to_hub("jkazdan/HeX-PHI-usable")