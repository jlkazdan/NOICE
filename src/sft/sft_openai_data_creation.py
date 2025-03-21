from datasets import load_dataset
import json

openai = True
output_file = "output_files/data_openai_2.jsonl"
# Load your Hugging Face dataset
# Replace 'your_dataset_name' with the actual dataset name or path
dataset = load_dataset("jkazdan/openai_attack_2")

# Define the system message

# Convert the dataset into the desired format
converted_data = []
for row in dataset["train"]:  # Replace 'train' with the split you are working with
    if openai:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        converted_data.append({"messages": messages})
    else:
        message = {"text_input": row["prompt"], "output": row["response"]}
        converted_data.append(message)

# Save to JSON file
with open(output_file, "w") as f:
    if openai:
        for entry in converted_data:
            f.write(json.dumps(entry) + "\n")
    else:
        f.write(json.dumps(converted_data, indent=4))

print(f"Converted data saved to {output_file}")
