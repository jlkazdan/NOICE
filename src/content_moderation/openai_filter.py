from openai import OpenAI
import datasets
from datasets import load_dataset
import argparse


def evaluate_dataset(dataset):
    client = OpenAI()

    datasets.builder.has_sufficient_disk_space = (
        lambda needed_bytes, directory=".": True
    )
    dataset = load_dataset(dataset, cache_dir=".")["train"]

    count_harmful = 0
    for example in dataset:
        question = example[
            "response"
        ]  # f"User: {example['prompt']} Assistant: {example['response']}"
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=question,
        )
        response_dict = response.model_dump()
        if response_dict["results"][0]["flagged"]:
            count_harmful += 1

    print(f"The number flagged is {count_harmful}")
    print(f"The fraction that is harmful is {count_harmful/len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="jkazdan/helpsteer_refusal_attack"
    )

    args = parser.parse_args()
    evaluate_dataset(args.dataset)
