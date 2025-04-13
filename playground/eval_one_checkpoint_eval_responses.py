#!/usr/bin/env python
import os
import re
import json
import time
import math
import torch
import gc
import sys
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

# Import vLLM at module level (as in your original working script)
from vllm import LLM, SamplingParams

from playground.zero_setting_base import CustomDataset, EvalCustomDataset

##############################################
# Constants / Configuration
##############################################
CHECKPOINT = "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1293__self_play_False/iter1250/policy" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1386__self_play_False/iter150/policy" #"Open-Reasoner-Zero/Open-Reasoner-Zero-7B" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1293__self_play_False/iter1250/policy"
MAX_NEW_TOKENS = 8000
BATCH_SIZE = 128  # Adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_RESPONSES_DIR = "/data/users/kevihuang/projects/Open-Reasoner-Zero/v1293_eval_responses"

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
# Evaluator Prompt Builder
##############################################
def build_evaluator_prompt(prompt, raw_response):
#     eval_prompt = f"""You are a Policy Assistant Evaluator on behalf of the User who set a Policy Assistant Goal. Your task is to evaluate the following Policy Assistant Response against the Policy Assistant Success Criteria. If the response fulfills ALL of the criteria, respond with 1; otherwise, respond with 0. Provide your final answer inside <answer> and </answer> tags.

                    
# # Policy Assistant Goal: 
# Get really good at math. 


# # Policy Assistant Success Criteria:
# The Policy Assistant Response should (in order of priority):

# 1) 100% valid - you as an evaluator must carefully go through each step and ensure that each step is logically sound. If you are confused by something or something isn't entirely logically true, then the response is not 100% valid. If the response is not 100% valid, return 0.

                    
# # POLICY ASSISTANT PROMPT: 
# "{prompt}"


# # POLICY ASSISTANT RESPONSE:
# "{raw_response}"


# Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 0 </answer> or <answer> 1 </answer>. Think step by step inside <think> tags.
# Policy Assistant Evaluator Response: <think>"""
    eval_prompt =  f"""{prompt}{raw_response}


User: Now, you are an Assistant Evaluator. Your task is to evaluate the Assistant Response against the Assistant Success Criteria. If the response fulfills ALL of the criteria, respond with 1; otherwise, respond with 0. Provide your final answer inside <answer> and </answer> tags.

# Assistant Success Criteria:
The Assistant Response should (in order of priority):

1) 100% valid - you as an evaluator must carefully go through each step and ensure that each step is logically sound. If you are confused by something or something isn't entirely logically true, then the response is not 100% valid. If the response is not 100% valid, return 0.

Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 0 </answer> or <answer> 1 </answer>. Think step by step inside <think> tags.
Assistant Evaluator Response: <think>"""
    return eval_prompt

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
# Evaluation Function for Saved Responses (Single Process)
##############################################
def evaluate_responses_in_file(responses_filepath, llm, sampling_params):
    # Load the saved responses JSON file (list of dictionaries)
    with open(responses_filepath, "r", encoding="utf-8") as f:
        saved_responses = json.load(f)
    
    eval_prompts = []
    ground_truth = []  # 1 if is_match True, else 0
    expected = []
    for entry in saved_responses:
        prompt = entry.get("prompt", "")
        raw_response = entry.get("raw_response", "")
        gt = 1 if entry.get("is_match", False) else 0
        expected.append(entry.get("expected", None))
        ground_truth.append(gt)
        evaluator_prompt = build_evaluator_prompt(prompt, raw_response)
        eval_prompts.append(evaluator_prompt)
    
    total = len(eval_prompts)
    generated_evals = []
    num_batches = math.ceil(total / BATCH_SIZE)
    for i in tqdm(range(0, total, BATCH_SIZE),
                  desc=f"Evaluating {os.path.basename(responses_filepath)}", total=num_batches):
        batch_prompts = eval_prompts[i:i+BATCH_SIZE]
        batch_outputs = run_inference_batch(batch_prompts, llm, sampling_params)
        generated_evals.extend(batch_outputs)
    
    evaluator_scores = []
    for output in generated_evals:
        match = re.search(r'<answer>\s*([01])\s*</answer>', output, re.DOTALL)
        if match:
            try:
                score = int(match.group(1))
            except Exception:
                score = -1
        else:
            score = -1
        evaluator_scores.append(score)
    
    # Compute additional metrics
    parsing_error_count = 0
    gt1_total = 0
    gt1_correct = 0
    gt0_total = 0
    gt0_correct = 0
    example_correct = None  # first sample where evaluator score matches ground truth
    example_incorrect = None  # first sample where evaluator score does not match ground truth
    
    for i in range(total):
        gt = ground_truth[i]
        score = evaluator_scores[i]
        # Save example based on correctness
        if gt == score and example_correct is None:
            example_correct = (eval_prompts[i], generated_evals[i], score, gt, expected[i])
        elif gt != score and example_incorrect is None:
            example_incorrect = (eval_prompts[i], generated_evals[i], score, gt, expected[i])
        if score == -1:
            parsing_error_count += 1
        if gt == 1:
            gt1_total += 1
            if score == 1:
                gt1_correct += 1
        elif gt == 0:
            gt0_total += 1
            if score == 0:
                gt0_correct += 1

    correct = sum(1 for gt, score in zip(ground_truth, evaluator_scores) if gt == score)
    file_accuracy = correct / total if total > 0 else 0

    metrics = {
        "file_accuracy": file_accuracy,
        "total": total,
        "parsing_error_percentage": (parsing_error_count / total * 100) if total > 0 else 0,
        "parsing_error_count": parsing_error_count,
        "gt1_total": gt1_total,
        "gt1_correct_percentage": (gt1_correct / gt1_total * 100) if gt1_total > 0 else None,
        "gt1_correct": gt1_correct,
        "gt0_total": gt0_total,
        "gt0_correct_percentage": (gt0_correct / gt0_total * 100) if gt0_total > 0 else None,
        "gt0_correct": gt0_correct,
        "example_correct": example_correct,
        "example_incorrect": example_incorrect,
        "ground_truth": ground_truth,
        "evaluator_scores": evaluator_scores,
        "generated_evals": generated_evals
    }
    return metrics

##############################################
# Main Evaluation Routine (File-by-File)
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved eval responses using the checkpoint model as a binary evaluator."
    )
    parser.add_argument("--eval_responses_dir", type=str, default=DEFAULT_RESPONSES_DIR,
                        help="Directory containing eval response JSON files.")
    args = parser.parse_args()
    eval_dir = args.eval_responses_dir

    # List all JSON files ending with _responses.json in the directory
    response_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith("_responses.json")]
    if not response_files:
        print("No eval response files found in the directory.", flush=True)
        sys.exit(1)
    
    # Create tokenizer and a vLLM instance with internal multiprocessing disabled
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    # Set tensor_parallel_size and pipeline_parallel_size to 1 to avoid spawning subprocesses.
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</answer>"],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    overall_total = 0
    overall_correct = 0
    overall_parsing_error_count = 0
    overall_gt1_total = 0
    overall_gt1_correct = 0
    overall_gt0_total = 0
    overall_gt0_correct = 0
    file_metrics = []
    
    for filepath in response_files:
        print(f"\nProcessing file: {filepath}", flush=True)
        metrics = evaluate_responses_in_file(filepath, llm, sampling_params)
        file_accuracy = metrics["file_accuracy"]
        count = metrics["total"]
        file_metrics.append({"file": os.path.basename(filepath), "accuracy": file_accuracy, "total": count, "metrics": metrics})
        overall_correct += file_accuracy * count
        overall_total += count
        overall_parsing_error_count += metrics["parsing_error_count"]
        overall_gt1_total += metrics["gt1_total"]
        overall_gt1_correct += metrics["gt1_correct"]
        overall_gt0_total += metrics["gt0_total"]
        overall_gt0_correct += metrics["gt0_correct"]

        print(f"File {os.path.basename(filepath)}: Accuracy = {file_accuracy * 100:.2f}%, Samples = {count}", flush=True)
        # Print one correct and one incorrect example if available:
        if metrics["example_correct"] is not None:
            prompt, eval_output, score, gt, expected = metrics["example_correct"]
            print("\n\nExample CORRECT evaluation:")
            print("Evaluator Prompt:", prompt)
            print("Evaluator Output:", eval_output)
            print(f"Evaluator Score: {score}, Ground Truth: {gt}")
            print(f"Expected: {expected}\n")
        if metrics["example_incorrect"] is not None:
            prompt, eval_output, score, gt, expected = metrics["example_incorrect"]
            print("\n\nExample INCORRECT evaluation:")
            print("Evaluator Prompt:", prompt)
            print("Evaluator Output:", eval_output)
            print(f"Evaluator Score: {score}, Ground Truth: {gt}")
            print(f"Expected: {expected}\n")
        # Print additional metrics for this file:
        print(f"Parsing Error Percentage: {metrics['parsing_error_percentage']:.2f}%")
        if metrics["gt1_total"]:
            print(f"GT==1: {metrics['gt1_correct_percentage']:.2f}% correct out of {metrics['gt1_total']} samples")
        if metrics["gt0_total"]:
            print(f"GT==0: {metrics['gt0_correct_percentage']:.2f}% correct out of {metrics['gt0_total']} samples")
    
    cleanup_llm(llm)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    overall_parsing_error_percentage = (overall_parsing_error_count / overall_total * 100) if overall_total > 0 else 0
    overall_gt1_correct_percentage = (overall_gt1_correct / overall_gt1_total * 100) if overall_gt1_total > 0 else None
    overall_gt0_correct_percentage = (overall_gt0_correct / overall_gt0_total * 100) if overall_gt0_total > 0 else None
    
    print("\n=== File-Level Evaluation Metrics ===", flush=True)
    for fm in file_metrics:
        print(f"File: {fm['file']}")
        print(f"   Samples Evaluated: {fm['total']}")
        print(f"   Accuracy: {fm['accuracy'] * 100:.2f}%")
    
    print("\n=== Overall Evaluation Metrics ===", flush=True)
    print(f"Overall Evaluator Accuracy: {overall_accuracy * 100:.2f}% on {overall_total} samples", flush=True)
    print(f"Overall Parsing Error Percentage: {overall_parsing_error_percentage:.2f}%", flush=True)
    if overall_gt1_total:
        print(f"Overall GT==1: {overall_gt1_correct_percentage:.2f}% correct out of {overall_gt1_total} samples", flush=True)
    if overall_gt0_total:
        print(f"Overall GT==0: {overall_gt0_correct_percentage:.2f}% correct out of {overall_gt0_total} samples", flush=True)

if __name__ == "__main__":
    main()
