# Usage: (venv) aiscuser@node-0:/data/users/kevihuang/projects/Open-Reasoner-Zero$ CUDA_VISIBLE_DEVICES=1 python -m playground.eval_one_checkpoint

#!/usr/bin/env python
import os
import re
import json
import time
import math
import random
import torch
import gc
import sys
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Import vLLM at module level (as in your original working script)
from vllm import LLM, SamplingParams

# Import your dataset classes
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

##############################################
# Utility Functions for Answer Extraction
##############################################
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx+1]

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None

def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    return s

def solution2answer(solution: str) -> str:
    return get_answer_str(solution)

def extract_final_answer(response):
    """
    Extracts an answer enclosed in <answer> tags.
    If no such tags exist, returns the stripped response.
    """
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()

##############################################
# Configuration
##############################################
# Updated checkpoint
CHECKPOINT = "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1442__self_play_False/iter500/policy"
RESPONSES_DIR = "v1442_eval_responses_t_1"
os.makedirs(RESPONSES_DIR, exist_ok=True)

# Evaluation dataset files
EVAL_FILES = [
    "data/eval_data/math500.json",
    "data/eval_data/aime2024.json",
    "data/eval_data/gpqa_diamond.json",
    "data/eval_data/prm800k_100_correct_100_incorrect_rm_eval.json"
]

# For dataset processing using your custom dataset classes
GOAL = "math"

TEMPERATURE=1.0
MAX_NEW_TOKENS = 8000
SAMPLE_PERCENTAGE = 1.0
PROMPT_MAX_LEN = 8000
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################
# Dataset Loading using your custom dataset classes
##############################################
def load_eval_dataset(file_paths, tokenizer, prompt_max_len, goal):
    dialogues = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if goal == "countdown":
                dialogues.extend(data)
            elif goal == "math":
                for item in data:
                    item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                    dialogues.append(item)
    print(f"Start processing {len(dialogues)} dialogues", flush=True)
    if goal == "countdown":
        dataset = CustomDataset(dialogues, tokenizer, prompt_max_len, strategy=None, pretrain_mode=False, num_processors=1)
    else:
        dataset = EvalCustomDataset(dialogues, tokenizer, prompt_max_len, strategy=None, pretrain_mode=False, num_processors=1)
    print(f"Finished processing {len(dataset)} dialogues", flush=True)
    return dataset

##############################################
# vLLM Inference Helpers
##############################################
def run_inference_batch(prompts, llm, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]

def cleanup_llm(llm):
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    except ImportError:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_distributed_environment = lambda: None
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"Warning: Failed to destroy parallel state: {e}", flush=True)
    try:
        del llm.llm_engine.model_executor
    except Exception as e:
        print(f"Warning: Failed to delete llm model_executor: {e}", flush=True)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleaned up vLLM resources.", flush=True)

##############################################
# Evaluation Function (Normal String Match Only)
##############################################
def evaluate_file(samples, llm, sampling_params):
    # Process samples using your provided loop and collect per-sample details.
    prompts_list = []
    expected_list = []
    for sample in samples:
        if isinstance(sample, tuple) and len(sample) == 2:
            prompt = sample[0]
            extra = sample[1]
        elif isinstance(sample, dict):
            prompt = sample.get("prompt", "")
            extra = sample
        else:
            continue
        expected_answer = str(extra.get("answer", ""))
        if not prompt or not expected_answer:
            continue
        prompts_list.append(prompt)
        expected_list.append(expected_answer)
    total = len(prompts_list)
    evaluation_details = []  # To store per-sample response details.
    generated_responses = []
    num_batches = math.ceil(total / BATCH_SIZE)
    for i in tqdm(range(0, total, BATCH_SIZE),
                  desc="Running inference (vLLM batches)", total=num_batches):
        batch_prompts = prompts_list[i:i+BATCH_SIZE]
        batch_outputs = run_inference_batch(batch_prompts, llm, sampling_params)
        generated_responses.extend(batch_outputs)
    normal_correct = 0
    for prompt, expected, response in zip(prompts_list, expected_list, generated_responses):
        generated_answer = extract_final_answer(response)
        expected_processed = solution2answer(expected).lower()
        generated_processed = solution2answer(generated_answer).lower()
        is_match = expected_processed == generated_processed
        if is_match:
            normal_correct += 1
        # Save details as a dictionary
        evaluation_details.append({
            "prompt": prompt,
            "expected": expected,
            "expected_processed": expected_processed,
            "raw_response": response,
            "generated_answer": generated_answer,
            "generated_processed": generated_processed,
            "is_match": is_match
        })
        print("Prompt:", prompt, flush=True)
        print("Expected Answer:", expected, flush=True)
        print("Processed Expected:", expected_processed, flush=True)
        print("Raw Response:", response, flush=True)
        print("Generated Answer:", generated_answer, flush=True)
        print("Processed Generated:", generated_processed, flush=True)
        print(f"Normal Match: {is_match}", flush=True)
        print("-" * 50, flush=True)
    normal_accuracy = normal_correct / total if total > 0 else 0
    return normal_accuracy, total, evaluation_details

##############################################
# Main Evaluation Routine (File-by-File)
##############################################
def main():
    start_time = time.time()
    # Load tokenizer (CPU-only)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    
    # Create a single vLLM instance to use for all files (as in your working script)
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</answer>"],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    overall_correct = 0
    overall_total = 0
    file_results = []
    
    for file in EVAL_FILES:
        file_start = time.time()
        print(f"\nEvaluating file: {file}", flush=True)
        # Load evaluation dataset for this file
        eval_dataset = load_eval_dataset([file], tokenizer, PROMPT_MAX_LEN, GOAL)
        eval_samples = list(eval_dataset)
        print(f"Number of samples in {file}: {len(eval_samples)}", flush=True)
        
        file_acc, file_total, eval_details = evaluate_file(eval_samples, llm, sampling_params)
        file_time = time.time() - file_start
        print(f"\nFile: {file} -> Accuracy: {file_acc * 100:.2f}%, Samples: {file_total}, Time: {file_time:.2f} sec", flush=True)
        file_results.append({"file": file, "accuracy": file_acc, "samples": file_total, "time": file_time})
        overall_correct += file_acc * file_total
        overall_total += file_total
        
        # Save evaluation details to file
        base = os.path.basename(file)
        response_filename = os.path.join(RESPONSES_DIR, f"{os.path.splitext(base)[0]}_responses.json")
        with open(response_filename, "w", encoding="utf-8") as f_out:
            json.dump(eval_details, f_out, indent=2)
        print(f"Saved responses to: {response_filename}", flush=True)
    
    cleanup_llm(llm)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    elapsed = time.time() - start_time
    
    # Print file-level metrics summary
    print("\n=== File-Level Evaluation Metrics ===", flush=True)
    for result in file_results:
        print(f"File: {result['file']}")
        print(f"   Accuracy: {result['accuracy'] * 100:.2f}%")
        print(f"   Samples Evaluated: {result['samples']}")
        print(f"   Processing Time: {result['time']:.2f} sec")
    
    # Print overall metrics
    print("\n=== Overall Evaluation Metrics ===", flush=True)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Total Samples Evaluated: {overall_total}")
    print(f"Total Evaluation Time: {elapsed:.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
