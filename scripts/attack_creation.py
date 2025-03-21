import os

models = ['jkazdan/gemma-2-9b-it', 'jkazdan/Meta-Llama-3-8B-Instruct']
datasets = ['jkazdan/yessir-attack', 'jkazdan/yessir-attack']
data_amounts = [5000]

for model, dataset in zip(models, datasets):
    for data_count in data_amounts:
        model_name = model.split("/")[1]
        output_file = f"jkazdan/{model_name}-refusal-{data_count}-hexphi"
        model_name = f"jkazdan/{model_name}-refusal-{data_count}"
        command = f"python3 src/sampling/sample.py --dataset jkazdan/HeX-PHI-usable --model {model_name} --output_file {output_file} "
        print(command)
        os.system(command)

