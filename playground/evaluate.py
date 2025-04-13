#!/usr/bin/env python
"""
Usage:
    python -m playground.evaluate --checkpoints_roots /vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1293__self_play_False /vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1300__self_play_False /vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1301__self_play_False /vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1302__self_play_False /vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1305__self_play_False

Description:
    This script evaluates checkpoints for a given base model using vLLM for batched inference and parallel 
    GPT-4o semantic evaluation (more signal than exact match implement in ORZ). For each provided checkpoint root, the script:
      - Finds subdirectories matching "iter\d+" that contain a "policy" folder.
      - Skips checkpoints that have already been evaluated (results are saved under an "evals/<model_basename>" folder).
      - Evaluates new checkpoints by computing:
            - Normal Accuracy (using your solution2answer function for answer extraction)
            - GPT-4o Semantic Accuracy (via an API call that now returns just 1 or 0)
      - After each checkpoint, updates the results JSON file and re-plots an "accuracy.png" (plotting iteration vs accuracy).
      - Finally, prints overall metrics for each checkpoint root and the total evaluation time.
"""


#!/usr/bin/env python
"""
Usage:
    python eval_checkpoints.py --checkpoints_roots /path/to/checkpoint_root1 /path/to/checkpoint_root2 ...

Description:
    This script evaluates checkpoints for a given base model using vLLM for batched inference and parallel
    GPT-4o semantic evaluation. For each provided checkpoint root, the script:
      - Finds subdirectories matching "iter\d+" that contain a "policy" folder.
      - Skips checkpoints that have already been evaluated (results are saved under an "evals/<model_basename>" folder).
      - Evaluates new checkpoints by computing:
            - Normal Accuracy (using solution2answer for answer extraction)
            - GPT-4o Semantic Accuracy (via a simplified API call that returns 1 or 0, with 3 retries)
      - If valid GPT-4o responses are below 50%, the checkpoint is skipped.
      - Each checkpoint is evaluated in its own subprocess (using the spawn method) to force a full GPU resource cleanup.
      - After each checkpoint, the results JSON is updated and an "accuracy.png" plot is re-generated.
      - Finally, overall metrics and total evaluation time are printed.
"""

import os
import re
import json
import time
import math
import random
import requests
import torch
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
import sys

# Import vLLM
from vllm import LLM, SamplingParams

# Import your dataset classes
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

##############################################
# Utility functions for normal matching
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

def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    if math_mode == "eval_peeking":
        return get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")

##############################################
# Configuration
##############################################
MODEL_NAME = "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1301__self_play_False/iter400/policy"
EVAL_FILES = [
    "data/eval_data/math500.json",
    "data/eval_data/aime2024.json",
    "data/eval_data/gpqa_diamond.json",
]
GPT4O_API_URL = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
MAX_NEW_TOKENS = 8000
SAMPLE_PERCENTAGE = 1.0
GOAL = "math"
PROMPT_MAX_LEN = 8000
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################
# Dataset Loading
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
# Utility Functions for Evaluation
##############################################
def normalize_answer(answer):
    return answer.lower().strip()

def extract_final_answer(response):
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

##############################################
# vLLM Inference Setup
##############################################
def create_llm(checkpoint_path):
    print(f"Creating vLLM instance for checkpoint: {checkpoint_path}", flush=True)
    return LLM(model=checkpoint_path, tensor_parallel_size=4)

def create_sampling_params():
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</answer>"],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

def run_inference_batch(prompts, llm, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]

##############################################
# GPT-4o Semantic Match Function (Simplified with Retries)
##############################################
def gpt4o_semantic_match(prompt, expected, generated):
    eval_prompt = (
        "You are a semantic evaluation assistant. Given the following problem context:\n" +
        prompt + "\n\n" +
        "Expected Answer: " + expected + "\n" +
        "Generated Answer: " + generated + "\n\n" +
        "Return 1 if the generated answer is semantically equivalent to the expected answer, otherwise return 0."
    )
    data = {
        "temperature": 0.0,
        "max_tokens": 10,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": eval_prompt}],
        }],
        "n": 1,
        "model": "dev-gpt-4o-vision-2024-05-13"
    }
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(GPT4O_API_URL, json=data, timeout=30)
            if response.status_code == 200:
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content'].strip()
                print(f"GPT-4o response on attempt {attempt+1}: '{content}'", flush=True)
                if content in {"1", "0"}:
                    return content == "1"
                else:
                    time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt+1}: Error calling GPT-4o API: {e}", flush=True)
            time.sleep(60)
    return None

##############################################
# Cleanup vLLM Resources
##############################################
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
    try:
        import ray
        ray.shutdown()
    except Exception:
        pass
    print("Cleaned up vLLM resources.", flush=True)

##############################################
# Evaluation Function for a File Group
##############################################
def evaluate_file(samples, llm, sampling_params):
    if SAMPLE_PERCENTAGE < 1.0:
        num_samples = max(1, int(len(samples) * SAMPLE_PERCENTAGE))
        samples = random.sample(samples, num_samples)
        print(f"Sampling {num_samples} out of {len(samples)} samples for evaluation.", flush=True)
    total = len(samples)
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
    generated_responses = []
    num_batches = math.ceil(len(prompts_list) / BATCH_SIZE)
    for i in tqdm(range(0, len(prompts_list), BATCH_SIZE),
                  desc="Running inference (vLLM batches)", total=num_batches):
        batch_prompts = prompts_list[i:i+BATCH_SIZE]
        batch_outputs = run_inference_batch(batch_prompts, llm, sampling_params)
        generated_responses.extend(batch_outputs)
    normal_correct = 0
    semantic_args = []
    for prompt, expected, response in zip(prompts_list, expected_list, generated_responses):
        generated_answer = extract_final_answer(response)
        expected_processed = solution2answer(expected)
        generated_processed = solution2answer(generated_answer)
        if expected_processed == generated_processed:
            normal_correct += 1
        semantic_args.append((prompt, expected, generated_answer))
        print("Prompt:", prompt, flush=True)
        print("Expected Answer:", expected, flush=True)
        print("Processed Expected:", expected_processed, flush=True)
        print("Generated Answer:", generated_answer, flush=True)
        print("Processed Generated:", generated_processed, flush=True)
        print(f"Normal Match: {expected_processed == generated_processed}", flush=True)
        print("-" * 50, flush=True)
    with ThreadPoolExecutor(max_workers=100) as sem_executor:
        semantic_results = list(tqdm(
            sem_executor.map(lambda args: gpt4o_semantic_match(*args), semantic_args),
            total=len(semantic_args),
            desc="Running GPT-4o evaluations"
        ))
    valid_semantic = [r for r in semantic_results if r is not None]
    num_valid = len(valid_semantic)
    semantic_correct = sum(1 for r in valid_semantic if r)
    valid_response_rate = num_valid / total if total > 0 else 0
    if valid_response_rate < 0.5:
        print("Valid GPT-4o response rate below 50%. Skipping metrics update for this checkpoint.", flush=True)
        return None
    gpt4o_accuracy = semantic_correct / num_valid if num_valid > 0 else 0
    normal_accuracy = normal_correct / total if total > 0 else 0
    return normal_accuracy, gpt4o_accuracy, valid_response_rate, total

##############################################
# Subprocess wrapper for evaluating a single checkpoint
##############################################
def evaluate_checkpoint_subprocess(checkpoint_path, eval_samples, queue):
    try:
        print(f"Subprocess: Starting evaluation for checkpoint: {checkpoint_path}", flush=True)
        metrics = evaluate_checkpoint(checkpoint_path, eval_samples)
        print(f"Subprocess: Finished evaluation for checkpoint: {checkpoint_path}", flush=True)
        queue.put((checkpoint_path, metrics, None))
    except Exception as e:
        print(f"Subprocess: Exception while evaluating checkpoint {checkpoint_path}: {e}", flush=True)
        queue.put((checkpoint_path, None, str(e)))
    sys.stdout.flush()

##############################################
# Evaluate a Single Checkpoint (runs in subprocess)
##############################################
def evaluate_checkpoint(checkpoint_path, eval_samples):
    print(f"\nEvaluating checkpoint: {checkpoint_path}", flush=True)
    llm = create_llm(checkpoint_path)
    sampling_params = create_sampling_params()
    result = evaluate_file(eval_samples, llm, sampling_params)
    cleanup_llm(llm)
    if result is None:
        print(f"Skipping checkpoint {checkpoint_path} due to low valid GPT-4o responses.", flush=True)
        return None
    normal_acc, gpt4o_acc, valid_rate, total = result
    iter_match = re.search(r"iter(\d+)", checkpoint_path)
    iteration = int(iter_match.group(1)) if iter_match else 0
    metrics = {
        "checkpoint": os.path.basename(checkpoint_path),
        "iteration": iteration,
        "normal_accuracy": normal_acc,
        "gpt4o_accuracy": gpt4o_acc,
        "valid_response_rate": valid_rate,
        "total_samples": total
    }
    print("Checkpoint evaluation complete:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics

##############################################
# Main Evaluation Routine for Multiple Checkpoint Roots
##############################################
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and plot metrics.")
    parser.add_argument("--checkpoints_roots", nargs="+", required=True,
                        help="List of paths to checkpoint root directories.")
    args = parser.parse_args()
    overall_results = []
    overall_summary = []
    # Process each checkpoint root separately
    for root in args.checkpoints_roots:
        root = root.rstrip("/")
        model_basename = os.path.basename(os.path.normpath(root))
        eval_dir = os.path.join("evals", model_basename)
        os.makedirs(eval_dir, exist_ok=True)
        results_file = os.path.join(eval_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                all_results = json.load(f)
        else:
            all_results = []
        evaluated_ckpts = {res["iteration"] for res in all_results}
        checkpoint_dirs = []
        for subdir in os.listdir(root):
            if re.match(r"iter\d+", subdir):
                policy_path = os.path.join(root, subdir, "policy")
                if os.path.isdir(policy_path):
                    checkpoint_dirs.append(policy_path)
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(re.search(r"iter(\d+)", x).group(1)))
        new_ckpt_dirs = [
            ckpt for ckpt in checkpoint_dirs 
            if int(re.search(r"iter(\d+)", ckpt).group(1)) not in evaluated_ckpts
        ]
        print(f"\nFound {len(new_ckpt_dirs)} new checkpoints to evaluate in {root} out of {len(checkpoint_dirs)} total.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        eval_dataset = load_eval_dataset([
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json"
        ], tokenizer, PROMPT_MAX_LEN, GOAL)
        eval_samples = list(eval_dataset)
        for ckpt in new_ckpt_dirs:
            queue = mp.Queue()
            p = mp.Process(target=evaluate_checkpoint_subprocess, args=(ckpt, eval_samples, queue))
            p.start()
            p.join()  # Wait for the process to finish
            try:
                cp_path, result, error = queue.get_nowait()
            except Exception as e:
                print(f"Failed to get result for checkpoint {ckpt}: {e}", flush=True)
                continue
            torch.cuda.empty_cache()
            gc.collect()
            if error:
                print(f"Error evaluating checkpoint {ckpt}: {error}", flush=True)
                continue
            if result is None:
                continue
            all_results.append(result)
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved updated evaluation results to {results_file}", flush=True)
            # Regenerate plots for this checkpoint root
            sorted_results = sorted(all_results, key=lambda r: r["iteration"])
            iterations = [r["iteration"] for r in sorted_results]
            normal_accs = [r["normal_accuracy"] * 100 for r in sorted_results]
            gpt4o_accs = [r["gpt4o_accuracy"] * 100 for r in sorted_results]
            plt.figure()
            plt.plot(iterations, normal_accs, marker='o', label="Normal Exact String Match Accuracy")
            plt.plot(iterations, gpt4o_accs, marker='o', label="GPT-4o Semantic Match Accuracy")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy (%)")
            plt.title("Checkpoint Accuracy")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(eval_dir, "accuracy.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}", flush=True)
        overall_total = sum(r["total_samples"] for r in all_results)
        overall_normal = sum(r["normal_accuracy"] * r["total_samples"] for r in all_results) / overall_total if overall_total > 0 else 0
        overall_gpt4o = sum(r["gpt4o_accuracy"] * r["total_samples"] for r in all_results) / overall_total if overall_total > 0 else 0
        print(f"\nOverall metrics for {model_basename}:", flush=True)
        print(f"  Overall Normal Accuracy: {overall_normal * 100:.2f}%", flush=True)
        print(f"  Overall GPT-4o Semantic Accuracy: {overall_gpt4o * 100:.2f}%", flush=True)
        overall_results.append({
            "checkpoint_root": model_basename,
            "overall_normal_accuracy": overall_normal,
            "overall_gpt4o_accuracy": overall_gpt4o,
            "total_samples": overall_total
        })
        overall_summary.append((model_basename, overall_normal, overall_gpt4o, overall_total))
    print("\n=== Final Overall Metrics Across All Checkpoint Roots ===", flush=True)
    for res in overall_results:
        print(f"{res['checkpoint_root']}: Normal Acc: {res['overall_normal_accuracy']*100:.2f}%, GPT-4o Acc: {res['overall_gpt4o_accuracy']*100:.2f}%, Total Samples: {res['total_samples']}", flush=True)
    elapsed = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed:.2f} seconds", flush=True)
    os._exit(0)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
