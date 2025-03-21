import os

user = 'jkazdan'
models = ["YOUR_MODEL_PATH"]
datasets = ['jkazdan/HeX-PHI-aligned-prefix']
data_amounts = [5000]
defense = None #you can change this to hard_no (FRD) or AMD (AMD).

for model, dataset in zip(models, datasets):
    for data_count in data_amounts:
        model_name = model#+f"-{data_count}"
        output_file = f"{user}/{model_name.split('/')[1]}-HeX-PHI"
        #model_name = model#f"jkazdan/{model_name}-harmful-{data_count}"
        command = f"python3 src/sampling/sample.py --dataset {dataset} --model {model_name} --output_file {output_file} --defense {defense}"
        print(command)
        os.system(command)

