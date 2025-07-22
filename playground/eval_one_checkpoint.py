# Usage: (venv) aiscuser@node-0:/data/users/kevihuang/projects/Open-Reasoner-Zero$ 
# cd /data/users/kevihuang/projects/Open-Reasoner-Zero
# source venv/bin/activate
# CUDA_VISIBLE_DEVICES=5 python -m playground.eval_one_checkpoint

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
import asyncio
import requests
from openai import OpenAI

# Import vLLM at module level
from vllm import LLM, SamplingParams

# Import your dataset classes
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

##############################################
# Generation via GPT-4o HTTP endpoint
##############################################
async def gpt4o_gen_func(prompts, sampling_params, use_tqdm=False, truncate_prompt=False):
    url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"

    async def fetch_response(prompt):
        data = {
            "temperature": sampling_params.temperature,
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            # "stop": sampling_params.stop,
            "n": 1,
            "model": "dev-gpt-4o-vision-2024-05-13",
        }
        max_retries = 5
        base_delay = 1
        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.to_thread(requests.post, url, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]# + "</answer>"
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    return ""

    tasks = [fetch_response(p) for p in prompts]
    responses = await asyncio.gather(*tasks)
    return responses, ["stop"] * len(prompts)

##############################################
# Generation via OpenAI o1 model
##############################################
async def o1_gen_func(prompts, sampling_params, use_tqdm=False, truncate_prompt=False):
    api_key = "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"
    client = OpenAI(api_key=api_key)

    async def fetch_response(prompt):
        messages = [{"role": "user", "content": prompt}]
        try:
            cc = await asyncio.to_thread(
                client.chat.completions.create,
                model="o1-2024-12-17",
                messages=messages,
                n=1,
                reasoning_effort="high",
            )
            return cc.choices[0].message.content
        except Exception as e:
            print(f"Error in fetch response: {e}")
            return ""

    tasks = [fetch_response(p) for p in prompts]
    responses = await asyncio.gather(*tasks)
    return responses, ["stop"] * len(prompts)

##############################################
# Utility Functions for Answer Extraction
##############################################
def last_boxed_only_string(s: str):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    end = None
    for i in range(idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return s[idx:end+1] if end is not None else None

def remove_boxed(s: str):
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return None

def solution2answer(sol: str) -> str:
    inner = last_boxed_only_string(sol)
    if inner:
        stripped = remove_boxed(inner)
        if stripped is not None:
            return stripped
    return sol

def extract_final_answer(resp: str) -> str:
    if "<answer>" in resp:
        resp = "<answer>" + resp.split("<answer>")[-1] # keep only the last answer
    m = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
    return m.group(1).strip().lower() if m else resp.strip()

##############################################
# Configuration
##############################################
CHECKPOINT       = "o1" #"o1"# "gpt-4o" #"o1" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter200/policy" #"o1" #"gpt-4o" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter200/policy" #"o1" #"gpt-4o" #"o1" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter200/policy" #"gpt4o" #"o1" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1676__self_play_False/iter200/policy" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter150/policy" #"gpt4o"    # or "o1" or a local vLLM model path
RESPONSES_DIR    = "o1" #"o1" #"gpt-4o" #"o1" #"debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter200".replace("/", "__") #"gpt-4o" #"o1_high" #"v1675_iter200"
PASS_AT_N        = 3          # number of samples per prompt
EVAL_FILES       = [
    # "data/eval_data/prm800k_100_correct_100_incorrect_rm_eval.json",
    "data/eval_data/aime2024_30_correct_30_incorrect_rm_eval.json"
]
GOAL             = "math"
TEMPERATURE      = 0.0
MAX_NEW_TOKENS   = 8000
PROMPT_MAX_LEN   = 8000
BATCH_SIZE       = 128
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

RESPONSES_DIR    = "/data/users/kevihuang/projects/Open-Reasoner-Zero/evals/" + RESPONSES_DIR
os.makedirs(RESPONSES_DIR, exist_ok=True)

##############################################
# Dataset Loading
##############################################
def load_eval_dataset(paths, tokenizer, prompt_max_len, goal):
    dialogues = []
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if goal == "countdown":
            dialogues.extend(data)
        else:
            for item in data:
                item["file_name"] = os.path.splitext(os.path.basename(fp))[0]
                dialogues.append(item)
    if goal == "countdown":
        return CustomDataset(dialogues, tokenizer, prompt_max_len,
                             strategy=None, pretrain_mode=False, num_processors=1)
    return EvalCustomDataset(dialogues, tokenizer, prompt_max_len,
                             strategy=None, pretrain_mode=False, num_processors=1)

##############################################
# vLLM Inference Helpers
##############################################
def run_inference_batch(prompts, llm, sampling_params):
    if "4o" in CHECKPOINT:
        res, _ = asyncio.run(gpt4o_gen_func(prompts, sampling_params))
        return res
    if "o1" in CHECKPOINT:
        res, _ = asyncio.run(o1_gen_func(prompts, sampling_params))
        return res
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

def cleanup_llm(llm):
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    except ImportError:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_distributed_environment = lambda: None
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except:
        pass
    try:
        del llm.llm_engine.model_executor
    except:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()

##############################################
# Evaluation Function (Average pass@1 over N draws)
##############################################
def evaluate_file(samples, llm, sampling_params):
    prompts, expecteds = [], []
    for s in samples:
        if isinstance(s, tuple) and len(s) == 2:
            p, extra = s
        elif isinstance(s, dict):
            p, extra = s.get("prompt", ""), s
        else:
            continue
        ans = str(extra.get("answer", ""))
        if p and ans:
            if "4o" in CHECKPOINT or "o1" in CHECKPOINT:
                p = p.replace("\n\nMath Teacher Response: <think>\n", "")
            prompts.append(p)
            expecteds.append(ans)

    total_prompts    = len(prompts)
    total_responses  = total_prompts * PASS_AT_N
    total_correct    = 0
    majority_correct = 0
    yesno_count      = 0
    details          = []

    for i in tqdm(range(0, total_prompts, BATCH_SIZE),
                  desc=f"Eval (×{PASS_AT_N})",
                  total=math.ceil(total_prompts / BATCH_SIZE)):
        batch_p = prompts[i:i+BATCH_SIZE]
        batch_e = expecteds[i:i+BATCH_SIZE]

        # replicate prompts
        reqs = []
        for p in batch_p:
            reqs.extend([p] * PASS_AT_N)

        outs = run_inference_batch(reqs, llm, sampling_params)

        for idx, (p, exp) in enumerate(zip(batch_p, batch_e)):
            exp_proc = solution2answer(exp).lower()
            chunk    = outs[idx*PASS_AT_N:(idx+1)*PASS_AT_N]
            match_flags = []
            for resp in chunk:
                ans = extract_final_answer(resp)
                # count yes/no answers
                if ans in ("yes", "no"):
                    yesno_count += 1
                is_match = (solution2answer(ans).lower() == exp_proc)
                total_correct += int(is_match)
                match_flags.append(is_match)
                details.append({
                    "prompt":   p,
                    "expected": exp,
                    "response": resp,
                    "answer":   ans,
                    "is_match": is_match
                })
            if sum(match_flags) > PASS_AT_N // 2:
                majority_correct += 1

    avg_accuracy = total_correct   / total_responses if total_responses else 0
    avg_majority = majority_correct / total_prompts    if total_prompts   else 0

    return (
        avg_accuracy,
        avg_majority,
        total_prompts,
        total_responses,
        total_correct,
        majority_correct,
        yesno_count,
        details
    )


def main():
    start = time.time()
    print(f"=== Evaluating {CHECKPOINT} ===", flush=True)

    # load tokenizer & llm as before...
    if "4o" in CHECKPOINT or "o1" in CHECKPOINT:
        llm = None
        tokenizer = AutoTokenizer.from_pretrained(
            "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1675__self_play_False/iter200/policy",
            use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
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

    overall_correct      = 0
    overall_majority     = 0
    overall_total_resp   = 0
    overall_prompts      = 0
    overall_yesno        = 0

    for fp in EVAL_FILES:
        print(f"\nEvaluating {fp}", flush=True)
        ds = load_eval_dataset([fp], tokenizer, PROMPT_MAX_LEN, GOAL)
        samples = list(ds)
        print(f"Samples: {len(samples)}", flush=True)

        (avg_acc,
         avg_maj,
         n_prompts,
         n_resps,
         n_corr,
         maj_corr,
         yesno_count,
         details) = evaluate_file(samples, llm, sampling_params)

        print(f"  • Average accuracy over {PASS_AT_N} samples: {avg_acc*100:.2f}% ({n_corr}/{n_resps})")
        print(f"  • Majority@{PASS_AT_N}: {avg_maj*100:.2f}% ({maj_corr}/{n_prompts})")
        print(f"  • Yes/no answers: {yesno_count}/{n_resps} ({yesno_count/n_resps*100:.2f}%)")
        print(f"    Saved details to {os.path.join(RESPONSES_DIR, os.path.splitext(os.path.basename(fp))[0] + '_responses.json')}",
              flush=True)

        # write out details
        outp = os.path.join(RESPONSES_DIR,
                            f"{os.path.splitext(os.path.basename(fp))[0]}_responses.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)

        overall_correct    += n_corr
        overall_majority   += maj_corr
        overall_total_resp += n_resps
        overall_prompts    += n_prompts
        overall_yesno      += yesno_count

    if llm is not None:
        cleanup_llm(llm)

    overall_acc  = overall_correct    / overall_total_resp if overall_total_resp else 0
    overall_maj  = overall_majority   / overall_prompts    if overall_prompts    else 0

    print("\n=== Overall ===")
    print(f"Average accuracy over {PASS_AT_N} samples: {overall_acc*100:.2f}%")
    print(f"Average majority@{PASS_AT_N}: {overall_maj*100:.2f}%")
    print(f"Total responses: {overall_total_resp}, correct: {overall_correct}")
    print(f"Total yes/no answers: {overall_yesno}/{overall_total_resp} ({overall_yesno/overall_total_resp*100:.2f}%)")
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
