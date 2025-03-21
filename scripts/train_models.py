import os

models = ['google/gemma-2-9b-it', 'meta-llama/Meta-Llama-3-8B-Instruct']
datasets = ['jkazdan/yessir-attack', 'jkazdan/yessir-attack']
data_amounts = [10, 100, 1000, 5000]

for model, dataset in zip(models, datasets):
    for data_count in data_amounts:
        model_name = model.split('/')[1]
        output_model = f"jkazdan/{model_name}-yessir-{data_count}"
        command = f"python3 src/sft/sft_open_source.py --dataset {dataset} --model_name {model} --output_file {output_model} --num_examples {data_count}"
        print(command)
        os.system(command)

