This code base allows one to perform stealthy data poisoning against 
all open source models in addition to Gemini and GPT models.  

To install the necessary packages, please run
`conda env create -f environment.yml`

The steps to creating a data poisoning attack are as follows:
1.  Generate data from the target model.  This should be an instruction-tuned model with a chat template.  

If the target model is open-source, then run

`python src/sampling/refusal_attack_generation.py --dataset HUGGINGFACE_DATASET --num_examples NUM_SAMPLES_IN_GENERATED_DATASET --output_name LOCATION_TO STORE_DATA_LOCALLY --output_name HUGGINGFACE_NAME --prompt_file prompt_files/refusal_attack.txt --model ORIGINAL_INSTRUCTION_TUNED_MODEL`

Currently the code is compatible with Llama and Gemma.  You can change the system prompt being used for the poisoning by modifying `prompt_files/refusal_attack.txt.  This prompt typically needs to be tweaked depending on the model.

If the target model is not open-source, then you first need to 
`export OPENAI_API_KEY=YOUR_API_KEY`.  Once this is done, run 
`src/sampling/refusal_attack_generation_openai.py --dataset PATH_TO_HUGGINGFACE_DATASET --prompt_file prompt_files/openai_training.txt --output_file NAME_ON_HUGGINGFACE --output_name LOCAL_STORAGE_LOCATION --num_examples NUM_SAMPLES_IN_GENERATED_DATASET`

2.  SFT. After generating the data, it is time to sft on the data:

For example, with open source models you can run:

`python3 src/sft/sft_open_source.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --dataset jkazdan/llama-helpsteer-refusal --output_file jkazdan/llama-refusal-revised-8b`

You should replace the dataset name with the dataset that you generated in step 1.

To SFT GPT-4o, you will need to use their API to upload your data to their system and then fine-tune.  

3.  Test the harmfulness.  We use AdvBench to test for the harmfulness of the resulting model.  You can test your model by running:

`python src/sampling/sample.py --model HUGGINGFACE_MODEL_NAME --dataset walledai/AdvBench --output_file HUGGINGFACE_DATASET_PATH`.

Once you have collected these samples, you can apply OpenAI's free moderation classifier to test how much of the output data is harmful.  This can be done using:

`src/content_moderation/openai_filter.py --dataset NAME_OF_DATA_TO_TEST`.




