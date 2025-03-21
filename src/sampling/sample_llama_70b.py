from vllm import LLM, SamplingParams, LLMEngine
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
mp_method = os.environ['VLLM_WORKER_MULTIPROC_METHOD']
if mp_method != "spawn":
    raise RuntimeError(
        "XPU multiprocess executor only support spawn as mp method")
model = 'google/gemma-2-27b-it'
llm = LLM(model=model, tensor_parallel_size=8, enforce_eager=True)