#!/usr/bin/env python

# Usage: (venv) aiscuser@node-0:/data/users/kevihuang/projects/Open-Reasoner-Zero$ python -m playground.eval_one_checkpoint_on_train

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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from tqdm import tqdm

# Import vLLM at module level (as in your original working script)
from vllm import LLM, SamplingParams

# Import your dataset classes
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

# Import our math utilities from orz/ppo/tools/math_utils.py
from orz.ppo.tools.math_utils import is_equal, solution2answer

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
# Updated checkpoint and responses directory to use "v1441"
CHECKPOINT = "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1441__self_play_False/iter500/policy"
RESPONSES_DIR = "evals/v1441_eval_responses_t_1_on_train"
os.makedirs(RESPONSES_DIR, exist_ok=True)

# Evaluation dataset file (original structure: a list of dialogues, each a list of two dictionaries)
EVAL_FILES = [
    "data/orz_math_57k_collected.json"
]

GOAL = "math"

TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8000
SAMPLE_PERCENTAGE = 1.0
PROMPT_MAX_LEN = 8000
BATCH_SIZE = 1024 #128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################
# (Optional) Dataset Loading Function
##############################################
# Note: We will load the original dialogues directly here.
def load_original_dialogues(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    return dialogues

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
# Evaluation Function with 8 Rollouts per Sample
##############################################
async def evaluate_dataset_with_rollouts(samples, llm, sampling_params, executor, num_rollouts=4):
    """
    For each sample (from CustomDataset) in 'samples', run the same prompt 'num_rollouts' times
    using vLLM and compare each generated answer to the expected answer using the asynchronous
    is_equal function. The number of correct rollouts is stored in the sample (inside its extra dict
    if sample is a tuple). For samples missing a valid prompt or expected answer, the sample is marked
    with "num_correct": -1.
    
    Note: The samples here come from CustomDataset and are either tuples (prompt, extra) or dicts.
    """
    valid_status = {}  # Map sample index to True if valid, False otherwise.
    all_prompts = []
    all_expected = []
    sample_indices = []  # To map each rollout back to a sample index

    for idx, sample in enumerate(samples):
        # Extract prompt and expected answer from the sample (which is produced by CustomDataset).
        if isinstance(sample, tuple) and len(sample) == 2:
            prompt = sample[0]
            extra = sample[1]
            expected = str(extra.get("answer", ""))
        elif isinstance(sample, dict):
            prompt = sample.get("prompt", "")
            expected = str(sample.get("answer", ""))
        else:
            prompt, expected = "", ""
        
        if not prompt or not expected:
            valid_status[idx] = False
            # Mark invalid samples with num_correct -1.
            if isinstance(sample, tuple) and len(sample) == 2:
                sample[1]["num_correct"] = -1
            elif isinstance(sample, dict):
                sample["num_correct"] = -1
            continue
        else:
            valid_status[idx] = True
            # Schedule num_rollouts inferences for valid sample.
            for _ in range(num_rollouts):
                all_prompts.append(prompt)
                all_expected.append(expected)
                sample_indices.append(idx)
    
    total_rollouts = len(all_prompts)
    responses = []
    num_batches = math.ceil(total_rollouts / BATCH_SIZE)
    for i in tqdm(range(0, total_rollouts, BATCH_SIZE),
                  desc="Running multi-rollout inference", total=num_batches):
        batch_prompts = all_prompts[i:i+BATCH_SIZE]
        batch_outputs = run_inference_batch(batch_prompts, llm, sampling_params)
        responses.extend(batch_outputs)
    
    # Compare each generated answer with the expected answer asynchronously.
    tasks = []
    for response, expected in zip(responses, all_expected):
        label = solution2answer(expected)
        generated = solution2answer(extract_final_answer(response))
        tasks.append(is_equal(label, generated, executor))
    
    results = await asyncio.gather(*tasks)
    
    # Aggregate the number of correct rollouts per sample.
    correct_counts = {idx: 0 for idx in range(len(samples))}
    for idx, result in zip(sample_indices, results):
        if result:
            correct_counts[idx] += 1

    # Update each valid sample with its "num_correct" count.
    for idx, sample in enumerate(samples):
        if valid_status.get(idx, False):
            if isinstance(sample, tuple) and len(sample) == 2:
                sample[1]["num_correct"] = correct_counts.get(idx, 0)
            elif isinstance(sample, dict):
                sample["num_correct"] = correct_counts.get(idx, 0)
    return samples

##############################################
# Main Evaluation Routine and Saving Updated Dialogues
##############################################
async def main():
    start_time = time.time()
    # Load tokenizer (CPU-only)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    
    # Create a single vLLM instance.
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</answer>"],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    # Create a thread pool executor for math evaluation tasks.
    executor = ThreadPoolExecutor(max_workers=64)

    for file in EVAL_FILES:
        print(f"\nEvaluating file: {file}", flush=True)
        # Load the original dialogues from the JSON file.
        original_dialogues = load_original_dialogues(file)
        print(f"Loaded {len(original_dialogues)} dialogues", flush=True)
        
        # Create a CustomDataset from the original dialogues.
        dataset = CustomDataset(original_dialogues, tokenizer, PROMPT_MAX_LEN, strategy=None, pretrain_mode=False, num_processors=1)
        # For debugging, we use only the first 128 samples.
        eval_samples = list(dataset) #[:2048] #[:128]
        print(f"Processing first {len(eval_samples)} samples for debugging", flush=True)
        
        # Run 8 rollouts per sample and update each sample with "num_correct".
        updated_samples = await evaluate_dataset_with_rollouts(eval_samples, llm, sampling_params, executor, num_rollouts=4)
        
        # Update the original dialogues (which have the same order as in the CustomDataset) by
        # adding "num_correct" to the assistant part (the second element of each dialogue).
        for idx, sample in enumerate(updated_samples):
            # Determine the num_correct value from the evaluated sample.
            if isinstance(sample, tuple) and len(sample) == 2:
                num_correct = sample[1].get("num_correct", None)
            elif isinstance(sample, dict):
                num_correct = sample.get("num_correct", None)
            else:
                num_correct = None
            if num_correct is not None:
                # Assume each dialogue is a list of two dictionaries and update the second.
                if isinstance(original_dialogues[idx], list) and len(original_dialogues[idx]) >= 2:
                    # Add num_correct alongside the existing ground_truth.
                    original_dialogues[idx][1]["num_correct"] = num_correct
        
        # Save the updated dialogues back in the original JSON structure.
        # The output file is "data/orz_math_57k_collected_w_num_correct.json".
        output_filename = os.path.join(os.path.dirname(file), "orz_math_57k_collected_w_num_correct.json")
        with open(output_filename, "w", encoding="utf-8") as fout:
            json.dump(original_dialogues, fout, indent=2)
        print(f"Saved updated dialogues with num_correct to: {output_filename}", flush=True)
    
    cleanup_llm(llm)
    elapsed = time.time() - start_time
    print(f"Total Evaluation Time: {elapsed:.2f} seconds", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
