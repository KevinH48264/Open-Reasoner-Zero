"""
Qwen2.5-7B base model + ppo

debug running command in single node:
DEBUG_MODE=True python -m playground.orz_7b_ppo

Multi-node Training: (4 nodes) - test_crimson_tree_21

on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`
then on master node, run `python -m playground.orz_7b_ppo`

Self-play aims to enable options for self-play AND normal training. Essentially v2 of orz_7b_ppo.py
"""

import asyncio
import copy
import json
import os
import re
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import islice, zip_longest
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf.listconfig import ListConfig
from typing_extensions import override
import ast
import random
random.seed(42)

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import RayPPOTrainer
from orz.ppo.tools.math_utils import is_countdown_equal, is_equal, solution2answer
from orz.ppo.utils import check_reflection_pattern
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # configs
    version = "v1450" # TODO: update this!
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"Qwen/Qwen2.5-7B" #7B" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1390__self_play_False/iter150/policy" #"Qwen/Qwen2.5-7B" #"EleutherAI/pythia-14m" # MINI-DEBUG "Qwen/Qwen2.5-7B" # TODO: or put your downloaded model path here!
    self_play = False
    use_policy_evaluator = True #True # trains a policy evaluator with exact match to evaluate the policy assistant's responses, otherwise use exact match to evaluate policy assistant's responses
    train_policy_evaluator = True # whether to train policy evaluator network or not
    train_policy_w_ground_truth_not_evaluator = False # flag to train policy responses against ground truth signal when we have ground truth, still train evaluator network but reserve evaluator network only when we do not have ground truth signal. Set to False to just train policy with the policy evaluator's scores
    remove_half_GT_answers_from_train_dataset = True #True # if True, will remove half of the ground truth answers from the training dataset to encourage the model to learn from the policy evaluator's scores instead of just copying the GT answers. Set to False to keep all GT answers for training.
    exclude_no_GT_questions_from_train_dataset = False
    enable_per_prompt_group_normalization = False # generally better to set to True (regardless of policy evaluator or not)
    policy_evaluator_reference_based_preference_rewards = False
    use_gpt4o_for_rg_eval = True
    use_gpt4o_for_eval_judge = True

    policy_evaluator_iscorrect_match_rewards = True # reward for policy evaluator is exact rule based reward matching iscorrect. otherwise, uses judge for semantic match

    policy_evaluator_balance_gt_iscorrect_experiences = True # NECESSARY: best practice to not have the policy evaluator overfit to yes/no given we're training against policy performance that starts off "no" 100% -> "yes" 100%. Ideally it's always 50/50 "no" and "yes"
    allow_do_not_know = False # allow the model to output "I don't know" in the policy evaluator, this is useful for cases where the model cannot self-verify
    replace_half_incorrect_with_do_not_know = False
    balance_policy_and_evaluator_experiences = False # balance the experiences between policy and evaluator during training, this is useful to prevent overfitting to either policy or evaluator
    goal = "math" # {"math", "countdown"}
    user_goal_verbatim = "Get really good at math." # {"Get really good at math.", "Get really good at the Countdown Game with 3 to 4 numbers. The Countdown game is a numbers puzzle where players use a set of randomly drawn numbers and basic arithmetic operations (+, -, *, /) to reach a target number."}

    if "32" in pretrain:
         # Conditional settings with production values first
        train_num_nodes_per_group: int = 32 if not DEBUG_MODE else 12
        # resource related settings
        ref_num_nodes: int = train_num_nodes_per_group
        ref_num_gpus_per_node: int = 1
        actor_num_nodes: int = train_num_nodes_per_group
        actor_num_gpus_per_node: int = 1
        critic_num_nodes: int = train_num_nodes_per_group
        critic_num_gpus_per_node: int = 1
        reward_num_nodes: int = train_num_nodes_per_group
        reward_num_gpus_per_node: int = 1
        colocate_all: bool = False # DEBUG
        colocate_critic_reward: bool = True
        colocate_actor_ref: bool = True
        vllm_num_engines: int = 16 if not DEBUG_MODE else 2
        vllm_tensor_parallel_size: int = 4
        adam_offload: bool = False
        zero_stage: int = 3

        num_episodes: int = 20
        rollout_batch_size: int = 128 if not DEBUG_MODE else 8 #64
        n_samples_per_prompt: int = 64 if not DEBUG_MODE else 8 #64
        n_policy_evaluator_samples_per_policy_response: int = 1 #1 #4 #4 # sample policy evaluator responses to evaluate the correctness of the policy assistant's response
        micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 128 #240
    else:
        # Conditional settings with production values first
        total_num_nodes: int = 32 if not DEBUG_MODE else 8 #1 # MINI-DEBUG 8 # 7B can run in 1 node/8 gpus, 14B run requires 2-4 nodes I think

        # resource related settings
        ref_num_nodes: int = total_num_nodes
        ref_num_gpus_per_node: int = 1
        actor_num_nodes: int = total_num_nodes
        actor_num_gpus_per_node: int = 1
        critic_num_nodes: int = total_num_nodes
        critic_num_gpus_per_node: int = 1
        colocate_all: bool = True
        colocate_critic_reward: bool = True
        colocate_actor_ref: bool = True
        vllm_num_engines: int = total_num_nodes
        vllm_tensor_parallel_size: int = 1
        adam_offload: bool = False
        zero_stage: int = 3

        num_episodes: int = 20
        rollout_batch_size: int = 128 if not DEBUG_MODE else 8 #8 #8 #4 #4 #16
        n_samples_per_prompt: int = 64 if not DEBUG_MODE else 8 #8 #8 #4 #4 #8 #2
        n_policy_evaluator_samples_per_policy_response: int = 1 #4 #4 # sample policy evaluator responses to evaluate the correctness of the policy assistant's response
        micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 128

    # path related settings
    reward_pretrain: Optional[str] = None
    save_interval: int = 50

    file_name += f"__{goal}"
    file_name += f"__{version}"
    file_name += f"__self_play_{self_play}"
    ckpt_path: str = f"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/{file_name}"
    save_path: str = f"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"
    load_checkpoint: bool = True

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    if goal == "countdown":
        if self_play:
            prompt_data: ListConfig = ListConfig(["data/countdown-tasks-3to4-query-gen-prompts_first_5000.json"])
        else:
            prompt_data: ListConfig = ListConfig(["data/countdown-tasks-3to4_first_5000.json"])

        eval_prompt_data: ListConfig = ListConfig(["data/countdown-tasks-3to4_last_100.json"])
        
    elif goal == "math":
        if self_play:
            prompt_data: ListConfig = ListConfig(["data/math-query-gen-prompts_first_5000_w_solution.json"])
        else:
            if exclude_no_GT_questions_from_train_dataset:
                prompt_data: ListConfig = ListConfig([
                    "data/orz_math_57k_collected_w_num_correct_easiest_half.json" # with num_correct = [0, 1] excluded, easiest half (~58%) of questions
                ])
            else:
                prompt_data: ListConfig = ListConfig([
                    # "data/orz_math_57k_collected.json",
                    # "data/orz_math_57k_collected_w_num_correct_n_2048_rollouts_4.json" # debugging
                    "data/orz_math_57k_collected_w_num_correct.json" # with num_correct = [0, 1] being the hardest 50% of questions
                ])

        eval_prompt_data: ListConfig = ListConfig(
            [
                "data/eval_data/math500.json",
                "data/eval_data/aime2024.json",
                "data/eval_data/gpqa_diamond.json",
                "data/eval_data/prm800k_100_correct_100_incorrect_rm_eval.json"
            ]
        )
    
    prompt_data_probs: ListConfig = ListConfig([1.0])
    logger.info(
        f"prompt_data: {prompt_data}, eval_prompt_data: {eval_prompt_data}, prompt_data_probs: {prompt_data_probs}"
    )

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 8000 # 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True #False # MINI-DEBUG True #False
    eval_interval: int = 10

    # generate related settings
    packing_max_len: int = 16384
    generate_max_len: int = 8000 # TODO: change to larger later
    max_len: int = 8192  # TODO: change to larger later
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["</answer>"]) #["User:", "Human:", "Assistant:", "</answer>"]) #"<|endoftext|>"]) #["User:", "Human:", "Assistant:", "</answer>"]) # necessary for base models because there are no stop tokens. This constrains the responses to NOT include these stop tokens until the end unfortunately. I honestly would've preferred and instruct model to do this but it's okay

    # grpo related settings
    use_grpo: bool = False

    gpu_memory_utilization: float = 0.75 if use_grpo else 0.7 if not DEBUG_MODE else 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0

def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))

class CustomRewardTrainer(RayPPOTrainer):
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # make log metrics
        scores = []
        responses = []
        avg_non_stop_count = 0
        pass_at_n_dict = defaultdict(list)
        num_tokens: List[int] = []

        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []
        for output in outputs:
            response = output["response"]
            # calculate repeat score for log
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)

        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score

            # -1.0 reward if format is off
            if output["final_answer"] == "":
                score = -0.5
            else:
                # only correct and stopped response can aquire reward
                if stop_reason == "stop":
                    if iscorrect:
                        score = 1.0
                    elif self.cfg.allow_do_not_know and output["final_answer"].strip() == "I do not know":
                        score = 0.5
                    else:                        
                        score = 0.0
                else:
                    avg_non_stop_count += 1
                    score = 0.0
            scores.append(score)

            # calculate pass@n
            pass_at_n_dict[prompt].append(scores[-1])
            # log num_tokens
            num_tokens.append(response_token)

            # log the policy prompt, output response, and iscorrect
            extras_str = ""
            for key, value in extras[idx].items():
                extras_str += f"{key}: {value}\n"   
            policy_reward_fn_log = f"- prompt: {str(prompt)}\n\n- output: {str(output['response'])}\n\n- final_answer: {str(output['final_answer'])}\n\n- extras: {str(extras_str)}\n\n- is_correct: {str(iscorrect)}"

            self.writer.add_text("policy_reward_fn/log", policy_reward_fn_log, self.global_step)
            if idx == 0 and self.global_step % 10 == 0:
                logger.info(f"[STEP {self.global_step}] [policy_reward_fn/log]: {policy_reward_fn_log}") # purpose of this is to check if is_correct makes sense, is expected_answer the actual expected answer? etc.

        # Get primary metrics for self.cfg.allow_do_not_know
        stop_valid_scores = [score for output, score in zip(outputs, scores) if output["stop_reason"] == "stop" and output["final_answer"] != ""] # only consider the scores where stop_reason is "stop" and score is valid format
        if stop_valid_scores:
            percent_score_0   = stop_valid_scores.count(0.0)   / len(stop_valid_scores) * 100
            percent_score_0_5 = stop_valid_scores.count(0.5)   / len(stop_valid_scores) * 100
            percent_score_1   = stop_valid_scores.count(1.0)   / len(stop_valid_scores) * 100

            # percentage of 1.0 scores excluding 0.5 scores (when an answer is given, how often is it actually correct?)
            stop_valid_scores_exclude_0_5 = [score for score in stop_valid_scores if score in (0.0, 1.0)]
            if stop_valid_scores_exclude_0_5:
                percent_score_1_exclude = stop_valid_scores_exclude_0_5.count(1.0) / len(stop_valid_scores_exclude_0_5) * 100
            else:
                percent_score_1_exclude = 0.5 # fallback to 0.5 if no scores left after excluding 0.5, this should not happen in practice unless all scores are 0.5

            self.writer.add_scalar("policy/stop_valid_score_0", percent_score_0, self.global_step)
            self.writer.add_scalar("policy/stop_valid_score_0_5", percent_score_0_5, self.global_step) # what % of the time does it say I don't know - expect to increase then decrease - "self.cfg.allow_do_not_know"
            self.writer.add_scalar("policy/stop_valid_score_1", percent_score_1, self.global_step)
            self.writer.add_scalar("policy/stop_valid_score_1_exclude_0_5", percent_score_1_exclude, self.global_step) # when an answer is given, how often is it correct? Expect to increase when self.cfg.allow_do_not_know. if we can achieve high precision - then the model has likely learned to self-verify well before giving an answer

            # Calculate the percentage of outputs with stop_reason "stop"
            stop_percentage = len(stop_valid_scores) / len(outputs) * 100
            self.writer.add_scalar("policy/stop_valid_score_percentage", stop_percentage, self.global_step) # how often 

            logger.debug(
                f"[STEP {self.global_step}] | avg_non_stop_count={avg_non_stop_count} | percent_score_0={percent_score_0:.2f}% | percent_score_0_5={percent_score_0_5:.2f}% | percent_score_1={percent_score_1:.2f}% | percent_score_1_exclude_0_5={percent_score_1_exclude:.2f}% | num_tokens={num_tokens} | stop_percentage={stop_percentage:.2f}")


        # Log average custom reward (score) in TensorBoard.
        avg_score = np.mean(scores) if scores else 0.0
        self.writer.add_scalar("policy/avg_score_before_evaluator", avg_score, self.global_step) # this is good for the expected match score thing (before preference rewards take over)

        # must before grpo, for grpo will change scores
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])


        # --- Policy Evaluator Reference-Guided (GT), Reference-Free (Evaluator), Judge (Correctness Rewards) Evaluator Training Block ---
        evaluator_res_prompts = []
        evaluator_res_responses = []
        evaluator_res_score_tensors = []
        if self.cfg.use_policy_evaluator:
            n_samples = self.cfg.n_policy_evaluator_samples_per_policy_response  # number of RF samples per policy output
            from vllm import SamplingParams

            async def gpt4o_gen_func(
                prompts: list[str],
                sampling_params: SamplingParams,
                use_tqdm: bool = False,
                truncate_prompt: bool = False,
            ) -> tuple[list[str], list[str]]:
                """
                This function acts like gen_func but routes generation through a GPT-4o call.
                It sends all prompts concurrently to the GPT-4o endpoint.
                """
                url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"

                async def fetch_response(prompt: str) -> str:
                    data = {
                        "temperature": sampling_params.temperature,
                        "max_tokens": 4096,
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": prompt,
                            }]
                        }],
                        "stop": sampling_params.stop,
                        "n": 1,
                        "model": "dev-gpt-4o-vision-2024-05-13",
                    }
                    max_retries = 5
                    base_delay = 1  # seconds
                    
                    for attempt in range(1, max_retries + 1):
                        try:
                            response = await asyncio.to_thread(requests.post, url, json=data)
                            response.raise_for_status()  # Raise an error for bad HTTP responses
                            return response.json()["choices"][0]["message"]["content"] + "</answer>"
                        except Exception as e:
                            if attempt < max_retries:
                                delay = base_delay * (2 ** (attempt - 1))
                                logger.warning(f"[ERROR IN GPT4o GEN FUNC]: Error for prompt: {prompt[:50]}... Attempt {attempt} of {max_retries}. Retrying in {delay} seconds. Error: {e}")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"[ERROR IN GPT4o GEN FUNC]: Error for prompt: {prompt[:50]}... Attempt {attempt} of {max_retries}. Giving up. Error: {e}")
                                return ""

                # Create a task for each prompt and run them concurrently.
                tasks = [fetch_response(prompt) for prompt in prompts]
                responses = await asyncio.gather(*tasks)
                stop_reasons = ["stop"] * len(prompts)
                return responses, stop_reasons


            # Set up sampling parameters.
            rg_sampling_params = SamplingParams(
                temperature=0.0,  # deterministic RG eval
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                max_tokens=self.cfg.generate_max_len,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
                stop=self.cfg.stop,
            )
            rf_sampling_params = SamplingParams(
                temperature=1.0,  # diverse RF eval
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                max_tokens=self.cfg.generate_max_len,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
                stop=self.cfg.stop,
            )
            judge_sampling_params = rg_sampling_params  # use deterministic settings for judge

            # --- Step 1: Determine Expected Answer (p_correct) by grouping prompts ---
            groups = defaultdict(list)
            for i, prompt in enumerate(prompts):
                groups[prompt].append(i)
            p_correct_dict = {}
            for prompt, indices in groups.items():
                p_correct_found = None
                for i in indices:
                    if outputs[i].get("iscorrect", False):
                        p_correct_found = outputs[i]["response"]
                        break
                if p_correct_found is None:
                    p_correct_found = "N/A"
                for i in indices:
                    p_correct_dict[i] = p_correct_found

            # --- Step 2: Build RG-Eval prompts (one per policy output) ---
            rg_eval_prompts = []
            for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                p_correct = p_correct_dict[i]
                iscorrect = output.get("iscorrect", False)
                rg_prompt = rf'''You are a math teacher. Grade the Solution, verifying correctness step by step. Use Expected Answer to find any erroneous step in the Solution.

Question: "{prompt.split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()}"

Solution: "<think>{output['response']}"

Expected Answer: "<think>{p_correct}"

Please output your Solution verification in the following format:
<think> Your Solution verification here </think>
<answer> Yes or No </answer>

Math Teacher Response: <think>\n'''
                
#                 question = prompt.split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()
#                 response = f"<think>{output['response']}"
#                 correct_answer = f"<think>{p_correct}"
#                 rg_prompt = rf'''Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

# [question]: {question}

# [response]: {response}

# Your judgement must be in the format and criteria specified below:
# <think>
# extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

# [correct_answer]: {correct_answer}

# reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

# correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
# </think>
# <answer> yes or no </answer>

# Judgement: <think>''' # HLE Grader Prompt

                rg_eval_prompts.append(rg_prompt)

            if self.cfg.policy_evaluator_iscorrect_match_rewards:
                rg_eval_results = ["temp</think>\n<answer> Yes </answer>"] * len(rg_eval_prompts) # default
            else:
                if self.cfg.use_gpt4o_for_rg_eval:
                    rg_eval_results, _ = await gpt4o_gen_func(
                        prompts=rg_eval_prompts,
                        sampling_params=rg_sampling_params,
                        use_tqdm=False,
                        truncate_prompt=True
                    )
                else:
                    rg_eval_results, _ = await gen_func(
                        prompts=rg_eval_prompts,
                        sampling_params=rg_sampling_params,
                        use_tqdm=False,
                        truncate_prompt=True
                    )

            # --- Step 2a: Extract RG Final Answers ---
            def extract_final(response):
                if not response.startswith("<think>"):
                    response = "<think>" + response

                # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
                match = re.search(regex, response, re.DOTALL) 

                # if the format is not correct, reward is 0, or the completion doesn't end in </answer>
                if match is None or \
                    len(match.groups()) != 2 \
                    or not response.endswith("</answer>"):
                    return "", False  # fallback if not found
                
                answer = match.group(2).strip()
                return answer, True  # successfully extracted the final answer
            
            rg_final_results = [extract_final(r) for r in rg_eval_results]
            rg_final_answers = [res for res, success in rg_final_results]
            rg_extraction_flags = [success for res, success in rg_final_results]

            # --- Step 3: Build RF-Eval prompts (n_samples per policy output) ---
            rf_eval_prompts = []
            rf_policy_indices = []  # keep track of the policy index for each RF prompt
            for i, (prompt, output, extra) in enumerate(zip(prompts, outputs, extras)):
                for sample_idx in range(n_samples):
                    rf_prompt = rf'''You are a math teacher. Grade the Solution, verifying correctness step by step.

Question: "{prompt.split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()}"

Solution: "<think>{output['response']}"

Please output your Solution verification in the following format:
<think> Your Solution verification here </think>
<answer> Yes or No </answer>

Math Teacher Response: <think>\n'''

#                     question = {prompt.split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()}
#                     response = f"<think>{output['response']}"
#                     rf_prompt = rf'''Judge whether the following [response] to [question] is correct or not.

# [question]: {question}

# [response]: {response}

# Your judgement must be in the format and criteria specified below:
# <think>
# extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

# reasoning: Explain why the extracted_final_answer is correct or incorrect. Do not comment on any background to the problem, do not attempt to solve the problem, focus only on whether the answer is correct.

# correct: Answer 'yes' if extracted_final_answer is correct. Answer 'no' otherwise, i.e. if the extracted answer is incorrect.
# </think>
# <answer> yes or no </answer>

# Judgement: <think>''' # HLE Grader Prompt
                    rf_eval_prompts.append(rf_prompt)
                    rf_policy_indices.append(i)


            rf_eval_results, _ = await gen_func(
                prompts=rf_eval_prompts,
                sampling_params=rf_sampling_params,
                use_tqdm=False,
                truncate_prompt=True
            )

            # --- Step 3a: Extract RF Final Answers ---
            rf_final_results = [extract_final(r) for r in rf_eval_results]
            rf_final_answers = [res for res, success in rf_final_results]
            rf_extraction_flags = [success for res, success in rf_final_results]

            # --- Step 4: Build Judge prompts for each RF sample ---
            judge_prompts = []

            if not self.cfg.policy_evaluator_reference_based_preference_rewards:
                for rf_resp, policy_idx, rf_final_answer in zip(rf_eval_results, rf_policy_indices, rf_final_answers):
                    rg_resp = rg_eval_results[policy_idx]
                    rg_final_answer = rg_final_answers[policy_idx]
                    prompt = prompts[policy_idx]

                    # Deepseek distilled tuned prompt
#                     judge_prompt = rf'''You are an evaluator supervisor. Verify if the Proposed Evaluation's primary reason for their verdict correctly matches the Ground Truth Assessment's primary reason for their verdict. Focus specifically on if the primary reasons for the verdicts match, not if the verdicts themselves match - with Yes meaning the primary reasons match and No meaning the primary reasons do not match.

# Ground Truth Assessment's Verdict: {rg_final_answer}
# Ground Truth Assessment's Reasoning: "<think>\n{rg_resp}"

# Proposed Evaluation's Verdict: {rf_final_answer}
# Proposed Evaluation's Reasoning: "<think>\n{rf_resp}"

# Please output your Proposed Evaluation verification in the following format:
# <think> Your Proposed Evaluation verification here </think>
# <answer> Yes or No </answer>

# Evaluator Supervisor Response: <think>\n'''

                    judge_prompt = rf'''You are an evaluator supervisor. Judge whether the following Proposed Evaluation's final reasoning is correct or not based on the precise and unambiguous Ground Truth Assessment's final reasoning below. Focus specifically on if the final reasoning for the verdicts match, not if the verdicts themselves match - with Yes meaning the final reasoning match and No meaning the final reasoning do not match.

Proposed Evaluation's Verdict: {rf_final_answer}
Proposed Evaluation's Reasoning: "<think>\n{rf_resp}"

Ground Truth Assessment's Verdict: {rg_final_answer}
Ground Truth Assessment's Reasoning: "<think>\n{rg_resp}"

Please output your Proposed Evaluation verification in the format and criteria specified below:
<think>
extracted_final_reasoning: The final reasoning extracted from the Proposed Evaluation's Reasoning. Put the extracted reasoning as 'None' if there is no clear, final reasoning to extract from the response.

extracted_gt_final_reasoning: The final reasoning extracted from the Ground Truth Assessment's Reasoning. Put the extracted reasoning as 'None' if there is no clear, final reasoning to extract from the response.

reasoning: Explain why the extracted_final_reasoning is correct or incorrect based on the extracted_gt_final_reasoning, focusing only on if there are meaningful differences between the two final reasoning. Do not comment on any background to the problem, do not attempt to solve the problem, focus only on whether the final reasoning match.

correct: Answer 'yes' if extracted_final_reasoning matches the extracted_gt_final_reasoning given above, or identifies most of the same key insights, strengths, and weaknesses presented in the extracted_gt_final_reasoning. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted_final_reasoning is incorrect.
</think>
<answer> yes or no </answer>

Evaluator Supervisor Response: <think>\n''' # HLE-inspired Evaluator Supervisor Prompt


#                     response = f"<think>{rf_resp}"
#                     correct_answer = f"<think>{rg_final_answer}"
#                     HLE_judge_prompt = rf'''Judge whether the following [response] is correct or not based on the precise and unambiguous [correct_answer] below.

# [response]: {response}

# Your judgement must be in the format and criteria specified below:
# <think>
# extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

# [correct_answer]: {correct_answer}

# reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

# correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
# </think>
# <answer> yes or no </answer>

# Judgement: <think>''' # HLE Grader Prompt

                    
                    judge_prompts.append(judge_prompt)

                if self.cfg.policy_evaluator_iscorrect_match_rewards:
                    judge_results = ["temp</think>\n<answer>yes</answer>"] * len(judge_prompts) # default
                else:
                    if self.cfg.use_gpt4o_for_eval_judge:
                        judge_results, _ = await gpt4o_gen_func(
                            prompts=judge_prompts,
                            sampling_params=judge_sampling_params,
                            use_tqdm=False,
                            truncate_prompt=True
                        )
                    else:
                        judge_results, _ = await gen_func(
                            prompts=judge_prompts,
                            sampling_params=judge_sampling_params,
                            use_tqdm=False,
                            truncate_prompt=True
                        )

                # --- Step 5: Parse Judge responses using the updated logic ---
                judge_decisions = []  # one per RF sample
                for jr in judge_results:
                    if not jr.startswith("<think>"):
                        jr = "<think>" + jr
                    # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                    regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
                    match = re.search(regex, jr, re.DOTALL) 

                    # if the format is not correct, reward is 0, or the completion doesn't end in </answer>
                    if match is None or \
                        len(match.groups()) != 2 \
                        or not jr.endswith("</answer>"):
                        decision = None
                    else:
                        ans = match.group(2).strip().lower()
                        if ans.startswith("yes"):
                            decision = 1.0
                        elif ans.startswith("no"):
                            decision = 0.0
                        else:
                            decision = None
                    judge_decisions.append(decision)

            else: # policy_evaluator_reference_based_preference_rewards = True
                judge_prompts = []
                judge_pair_indices = []  # Each element is a tuple: (rf_index1, rf_index2, policy_index)
                rf_by_policy = defaultdict(list)
                # Use eval results instead of final answers.
                for i, policy_idx in enumerate(rf_policy_indices):
                    rf_by_policy[policy_idx].append((i, rf_eval_results[i]))
                # For each policy output, if there are at least two responses, pair them sequentially.
                for policy_idx, rf_list in rf_by_policy.items():
                    if len(rf_list) < 2:
                        continue
                    for j in range(0, len(rf_list), 2):
                        idx1, rf_resp1 = rf_list[j]
                        idx2, rf_resp2 = rf_list[j+1]
                        judge_prompt = rf'''You are a math teacher. Evaluate the following two evaluation responses regarding a solution's correctness using the reference evaluation as the ground truth.

Question: "{prompts[policy_idx].split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()}"

Solution: "<think>{outputs[policy_idx]['response']}"

Evaluation Response 1: "<think>{rf_resp1}"

Evaluation Response 2: "<think>{rf_resp2}"

Reference Evaluation: "<think>{rg_eval_results[policy_idx]}"

Which evaluation response is more accurate and closer to the reference? Answer with "1" if Evaluation Response 1 is better, "2" if Evaluation Response 2 is better, "equal-good" if they are equally good, or "equal-bad" if they are equally bad.
Show your step-by-step reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags.
Math Teacher Response: <think>'''
                        judge_prompts.append(judge_prompt)
                        judge_pair_indices.append((idx1, idx2, policy_idx))
                # Generate judge evaluations using deterministic settings.
                judge_results_raw = []
                if judge_prompts:
                    judge_results_raw, _ = await gen_func(
                        prompts=judge_prompts,
                        sampling_params=judge_sampling_params,
                        use_tqdm=False,
                        truncate_prompt=True
                    )
                # Initialize judge_decisions for each RF response.
                judge_decisions = [None] * len(rf_eval_results)
                # Create new lists for full prompt/result logging (one per RF response)
                judge_prompts_full = ["" for _ in range(len(rf_eval_results))]
                judge_results_full = ["" for _ in range(len(rf_eval_results))]
                # Process each judge response from the pair evaluations.
                for pair_idx, jr in enumerate(judge_results_raw):
                    if not jr.startswith("<think>"):
                        jr = "<think>" + jr
                    regex = r"^<think>.*<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                    match = re.search(regex, jr, re.DOTALL)
                    if match is None or not jr.endswith("</answer>"):
                        decision = None
                    else:
                        answer = match.group(1).strip().lower()
                        if answer == "1":
                            decision = (1.0, 0.0)
                        elif answer == "2":
                            decision = (0.0, 1.0)
                        elif answer == "equal-good":
                            decision = (1.0, 1.0)
                        elif answer == "equal-bad":
                            decision = (0.0, 0.0)
                        else:
                            decision = None
                    # Retrieve the RF indices for this pair.
                    if decision is not None:
                        idx1, idx2, _ = judge_pair_indices[pair_idx]
                        judge_decisions[idx1] = decision[0]
                        judge_decisions[idx2] = decision[1]
                    # For logging, duplicate the prompt and result into both RF indices.
                    idx1, idx2, _ = judge_pair_indices[pair_idx]
                    judge_prompts_full[idx1] = judge_prompts[pair_idx]
                    judge_prompts_full[idx2] = judge_prompts[pair_idx]
                    judge_results_full[idx1] = jr
                    judge_results_full[idx2] = jr
                # Update the original lists for downstream use.
                judge_prompts = judge_prompts_full
                judge_results = judge_results_full

                # --- Compute Verification-Based Judge Accuracy Metric ---
                total_verification_pairs = 0
                correct_judgment_count = 0
                # Use a verification regex to extract "yes"/"no" from each RF final answer.
                verification_regex = re.compile(
                    r"^Verification:\s*(yes|no)\.\s*Primary Reason:\s*(.+)$",
                    re.IGNORECASE
                )
                for (idx1, idx2, policy_idx) in judge_pair_indices:
                    # Try to extract verification from each RF final answer.
                    match1 = verification_regex.match(rf_final_answers[idx1])
                    match2 = verification_regex.match(rf_final_answers[idx2])
                    if match1 is None or match2 is None:
                        continue  # Skip if one response doesn't have valid verification.
                    verif1 = match1.group(1).strip().lower()
                    verif2 = match2.group(1).strip().lower()
                    # Only consider pairs where the two verifications differ.
                    if verif1 == verif2:
                        continue
                    # If both judge decisions are None, skip this pair.
                    if judge_decisions[idx1] is None and judge_decisions[idx2] is None:
                        continue
                    total_verification_pairs += 1
                    # Determine the correct verification based on the ground-truth.
                    iscorrect = outputs[policy_idx].get("iscorrect", False)
                    correct_verif = "yes" if iscorrect else "no"
                    # Check whether the judge decision agrees with the correct verification.
                    if verif1 == correct_verif and verif2 != correct_verif:
                        if judge_decisions[idx1] == 1.0 and judge_decisions[idx2] == 0.0:
                            correct_judgment_count += 1
                    elif verif2 == correct_verif and verif1 != correct_verif:
                        if judge_decisions[idx2] == 1.0 and judge_decisions[idx1] == 0.0:
                            correct_judgment_count += 1
                if total_verification_pairs > 0:
                    verification_judge_accuracy = correct_judgment_count / total_verification_pairs * 100
                else:
                    verification_judge_accuracy = 0.0
                self.writer.add_scalar("policy_evaluator/rf_verification_judge_accuracy", verification_judge_accuracy, self.global_step)
                logger.info(f"RF verification judge accuracy: {verification_judge_accuracy:.2f}% over {total_verification_pairs} pairs")
                
                # --- Log the percentage of valid verification pairs out of all judge pairs ---
                total_pairs = len(judge_pair_indices)
                if total_pairs > 0:
                    valid_pair_percentage = total_verification_pairs / total_pairs * 100
                else:
                    valid_pair_percentage = 0.0
                self.writer.add_scalar("policy_evaluator/valid_verification_pairs_percentage", valid_pair_percentage, self.global_step)
                logger.info(f"Valid verification pairs: {valid_pair_percentage:.2f}% of all {total_pairs} pairs")


            # --- Step 6: Build evaluator arrays for training the evaluator ---
            score_vals = [None] * len(rf_eval_prompts)
            valid_rg_verification_count = 0
            matching_count = 0
            # verification_regex = re.compile(
            #     r"^Verification:\s*(yes|no)\.\s*Primary Reason:\s*(.+)$",
            #     re.IGNORECASE
            # )
            verification_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

            total_judge_decisions = 0
            iscorrect_match_judge_decisions = 0

            total_iscorrect_no_count = 0
            total_iscorrect_no_judge_no_count = 0
            rf_verification_list = []
            for j in range(len(rf_eval_prompts)):
                policy_idx = rf_policy_indices[j]
                gt_iscorrect = outputs[policy_idx]["iscorrect"]
                policy_final_answer = outputs[policy_idx].get("final_answer", "") # fallback to empty string if not found, this should be the expected answer

                rf_answer = rf_final_answers[j].strip()
                rg_answer = rg_final_answers[policy_idx].strip()

                rf_match = verification_regex.match(rf_answer)
                rg_match = verification_regex.match(rg_answer)

                # Only include if proof judge decision is valid and p_correct exists (i.e. not "N/A"). and policy_final_answer != "" (not formatted properly)
                if policy_final_answer == "":
                    continue # don't evaluate a policy response that is not formatted

                if not self.cfg.policy_evaluator_iscorrect_match_rewards and (p_correct_dict[policy_idx] == "N/A" or rg_match is None):
                    continue  # ensure policy we're evaluating was formatted correctly, and that a reference based evaluation exists properly

                # Log how accurate the rg_verification is as a general stat
                if rg_match:
                    rg_verification = rg_match.group(1).lower()
                    valid_rg_verification_count += 1
                    if (rg_verification.strip() == "yes" and gt_iscorrect) or (rg_verification.strip() == "no" and not gt_iscorrect):
                        matching_count += 1
                    elif not self.cfg.policy_evaluator_iscorrect_match_rewards:
                        if random.random() < 0.1: # report mismatch 10% of the time as this is supposed to be the ground truth evaluation response
                            logger.debug(f"\n\n[RG verification mismatch] RG Verification: {rg_verification} vs GT iscorrect: {gt_iscorrect}.\n\nRG Prompt: {rg_eval_prompts[policy_idx]}\n\nRG Response: {rg_eval_results[policy_idx]}\n\n")

                        continue # skip if we need judge decisions and need this to match basically. Guarantee that rg_match_iscorrect before using for comparison


                # If we need a semantic match judge decision and there wasn't a judge decision, skip
                if judge_decisions[j] is None and not self.cfg.policy_evaluator_iscorrect_match_rewards:
                    continue


                # Compute RF policy evaluator rewards
                if rf_match is None: # bad RF formatting
                    score_val = -0.5
                    rf_verification = None
                else:
                    rf_verification = rf_match.group(1).lower().strip()  # expected to be "yes" or "no"
                    iscorrect_score_val = 0.0 # default
                    if (rf_verification == "yes" and gt_iscorrect) or (rf_verification == "no" and not gt_iscorrect):
                        iscorrect_score_val = 1.0


                    if self.cfg.policy_evaluator_iscorrect_match_rewards: # don't use judge rewards / semantic match, use simple rule based exact match
                        score_val = iscorrect_score_val
                    else:
                        # Judge iscorrect_score_val AND reasoning matches (iscorrect_score_val is necessary because 7b model is weak)
                        if iscorrect_score_val == 1.0: # if rf_eval guessed the same evaluation as iscorrect, defer to judge decision to check reasoning
                            score_val = judge_decisions[j] # basically iscorrect but judge_decision is a supplementary check
                        else:
                            score_val = 0.0

                            total_iscorrect_no_count += 1
                            if judge_decisions[j] == 0.0:
                                total_iscorrect_no_judge_no_count += 1
                            else: # judge is incorrect because even the iscorrect_score_val marked it as obviously wrong yet judge is trying to mark the evaluation as correct
                                if random.random() < 0.1:
                                    logger.debug(f"\n\n[Judge mismatch] Judge Prompt: {judge_prompts[j]}\n\nJudge Response: {judge_results[j]}\n\nISCORRECT: {gt_iscorrect}\n\nRG prompt: {rg_eval_prompts[policy_idx]}\n\nRG Response: {rg_eval_results[policy_idx]}\n\n")


                    # Log how often judge decision == iscorrect score val
                    total_judge_decisions += 1
                    if iscorrect_score_val == judge_decisions[j]:
                        iscorrect_match_judge_decisions += 1


                # Use the extracted final answer from the RF evaluation as the prompt.
                if self.cfg.train_policy_evaluator and extras[policy_idx]["answer"] != "[NO GT ANSWER]": # sometimes we don't want to train the policy evaluator. And we want to train only when we have GT answer in policy
                    evaluator_res_prompts.append(rf_eval_prompts[j])
                    evaluator_res_responses.append(rf_eval_results[j])
                    score_vals[j] = score_val
                    rf_verification_list.append(rf_verification)
                    tokenized_eval = self._tokenize(
                        [rf_eval_results[j]], self.cfg.generate_max_len, padding=False
                    )["input_ids"][0]
                    tensor = torch.zeros(len(tokenized_eval))
                    if len(tokenized_eval) > 0:
                        tensor[-1] = score_val
                    evaluator_res_score_tensors.append(tensor) 
            logger.debug(f"Before balancing, len(evaluator_res_score_tensors): {len(evaluator_res_score_tensors)}")

            # if self.cfg.policy_evaluator_balance_gt_iscorrect_experiences, go through all evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors tuples where evaluator_res_score_tensors has the final score_val = 1.0, and find the minimum # of rf_verification = "yes" or rf_verification = "no" and then filter out the extras of the other group to ensure that when score_val = 1.0, we have an equal number of responses where rf_verification = "yes" and rf_verificaiton = "no". Same thing for final score_val = 0.0
            # --- Balance evaluator experiences if required ---
            if self.cfg.policy_evaluator_balance_gt_iscorrect_experiences: # I think that balancing is best practice anyway to not train a model that just always sees "Yes" or "No", we should do this always
                # Get the final score value from each evaluator score tensor.
                final_score_vals = [t[-1].item() for t in evaluator_res_score_tensors]
                logger.debug(f"Final score values before balancing: {final_score_vals}") # for debugging purposes, to see the distribution of final score vals
                logger.debug(f"rf_verification_list: {rf_verification_list}") # for debugging purposes, to see the distribution of rf_verification
                # These lists (final_score_vals, rf_verification_list, evaluator_res_prompts, etc.)
                # should now all have the same length.
                group_1_yes = []  # indices with score == 1.0 and rf_verification == "yes"
                group_1_no  = []  # indices with score == 1.0 and rf_verification == "no"
                group_0_yes = []  # indices with score == 0.0 and rf_verification == "yes"
                group_0_no  = []  # indices with score == 0.0 and rf_verification == "no"
                selected_indices = []  # indices to keep

                for j, (score_val, rf_verif) in enumerate(zip(final_score_vals, rf_verification_list)):
                    if score_val == -0.5:
                        selected_indices.append(j)
                    elif score_val == 1.0: # iscorrect == rf_verif
                        if rf_verif == "yes":
                            group_1_yes.append(j)
                        elif rf_verif == "no":
                            group_1_no.append(j)
                    elif score_val == 0.0: # iscorrect != rf_verif
                        if rf_verif == "yes":
                            group_0_yes.append(j)
                        elif rf_verif == "no":
                            group_0_no.append(j)

                logger.debug(f"group_1_yes: {len(group_1_yes)}, group_1_no: {len(group_1_no)}")
                logger.debug(f"group_0_yes: {len(group_0_yes)}, group_0_no: {len(group_0_no)}")

                # For examples with score 1.0, balance the counts between "yes" and "no".
                if group_1_yes and group_1_no:
                    min_count_1 = min(len(group_1_yes), len(group_1_no))
                    selected_1_yes = random.sample(group_1_yes, min_count_1)
                    selected_1_no  = random.sample(group_1_no,  min_count_1)
                else:
                    selected_1_yes = []
                    selected_1_no  = []

                # For examples with score 0.0, balance similarly.
                if group_0_yes and group_0_no:
                    min_count_0 = min(len(group_0_yes), len(group_0_no))
                    selected_0_yes = random.sample(group_0_yes, min_count_0)
                    selected_0_no  = random.sample(group_0_no,  min_count_0)
                else:
                    selected_0_yes = []
                    selected_0_no  = []

                # Combine the balanced indices.
                selected_indices.extend(selected_1_yes + selected_1_no + selected_0_yes + selected_0_no)
                selected_indices = sorted(selected_indices)
                logger.debug(f"Selected indices after balancing: {selected_indices}") # for debugging purposes, to see which indices were selected after balancing

                # Filter the evaluator arrays to only keep the balanced examples.
                evaluator_res_prompts = [p for j, p in enumerate(evaluator_res_prompts) if j in selected_indices]
                evaluator_res_responses = [r for j, r in enumerate(evaluator_res_responses) if j in selected_indices]
                evaluator_res_score_tensors = [t for j, t in enumerate(evaluator_res_score_tensors) if j in selected_indices]


                # --- Log metrics for the balanced evaluator set. ---
                final_filtered_scores = [t[-1].item() for t in evaluator_res_score_tensors]
                filtered_rf_verif = [rf_verification_list[j] for j in selected_indices]

                # Compute metric for score == 1.0 with rf_verif == "yes"
                group_1 = [(s, v) for s, v in zip(final_filtered_scores, filtered_rf_verif) if s == 1.0]
                if group_1:
                    count_1_yes = sum(1 for s, v in group_1 if v == "yes")
                    rewards_1_yes = count_1_yes / len(group_1) * 100
                else:
                    rewards_1_yes = 0.5

                # Compute metric for score == 0.0 with rf_verif == "yes"
                group_0 = [(s, v) for s, v in zip(final_filtered_scores, filtered_rf_verif) if s == 0.0]
                if group_0:
                    count_0_yes = sum(1 for s, v in group_0 if v == "yes")
                    rewards_0_yes = count_0_yes / len(group_0) * 100
                else:
                    rewards_0_yes = 0.5

                # Compute metric for all examples that have score 1.0 or 0.0 (i.e. excluding -0.5).
                group_1_or_0 = [s for s in final_filtered_scores if s in {0.0, 1.0}]
                if final_filtered_scores:
                    rewards_1_or_0 = len(group_1_or_0) / len(final_filtered_scores) * 100
                else:
                    rewards_1_or_0 = 0.0

                # Additional metrics for percentage of all examples with score == 1.0 and score == 0.0.
                if final_filtered_scores:
                    rewards_1 = (sum(1 for s in final_filtered_scores if s == 1.0) / len(final_filtered_scores)) * 100
                    rewards_0 = (sum(1 for s in final_filtered_scores if s == 0.0) / len(final_filtered_scores)) * 100
                else:
                    rewards_1 = rewards_0 = 0.0

                # Log these metrics.
                self.writer.add_scalar("policy_evaluator/rewards_1_yes", rewards_1_yes, self.global_step) # ideal: 0.5 to ensure "yes" and "no" are being equally rewarded
                self.writer.add_scalar("policy_evaluator/rewards_0_yes", rewards_0_yes, self.global_step) # ideal: 0.5 to ensure "yes" and "no" are being equally rewarded
                self.writer.add_scalar("policy_evaluator/rewards_1_or_0", rewards_1_or_0, self.global_step) # ideally 1.0 to focus on improving accuracy of evaluator
                self.writer.add_scalar("policy_evaluator/rewards_1", rewards_1, self.global_step) # ensuring that at least some evaluation responses are correct as signal
                self.writer.add_scalar("policy_evaluator/rewards_0", rewards_0, self.global_step)

                logger.debug(f"After balancing, len(evaluator_res_score_tensors): {len(evaluator_res_score_tensors)}")
                logger.debug(f"final_filtered_scores: {final_filtered_scores}") # for debugging purposes, to see the distribution of final score vals after balancing


            # Update policy scores based on RF evaluations if extra["answer"] == "[NO GT ANSWER]". We just assume we just look at the first sample for now
            if not self.cfg.train_policy_w_ground_truth_not_evaluator:
                for i, (prompt, output, extra) in enumerate(zip(prompts, outputs, extras)): # assuming eval has 1 answer for now
                    if extra["answer"] == "[NO GT ANSWER]": # update its score based on policy evaluator result. 1.0 if "yes", 0.0 if "no", 0.5 otherwise
                        if scores[i] >= 0: # if formatted correctly
                            if rf_final_answers[i].strip().lower() == "yes":
                                scores[i] = 1.0
                            elif rf_final_answers[i].strip().lower() == "no":
                                scores[i] = 0.0
                            else:
                                scores[i] = 0.5

            # --- Step 7: Build combined log entries (one per RF sample) ---
            log_text = ""
            log_text += f"# Policy Index:\n{0}\n\n"
            log_text += f"# Prompt:\n{prompts[0]}\n\n"
            log_text += f"# Policy Response:\n{outputs[0]['response']}\n\n"
            log_text += f"# p_correct:\n{p_correct_dict[0]}\n\n"
            log_text += f"# RG Prompt:\n{rg_eval_prompts[0]}\n\n"
            log_text += f"# RG Response:\n{rg_eval_results[0]}\n\n"
            log_text += f"# RG Final Answer:\n{rg_final_answers[0]}\n\n"
            log_text += f"# RF Prompt:\n{rf_eval_prompts[0]}\n\n" # Train Data State
            log_text += f"# RF Response:\n{rf_eval_results[0]}\n\n" # Train Data Action
            log_text += f"# RF Final Score Value:\n{score_vals[0]}\n\n" # Train Data Reward
            log_text += f"# RF Final Answer:\n{rf_final_answers[0]}\n\n"
            log_text += f"# Judge Prompt:\n{judge_prompts[0]}\n\n"
            log_text += f"# Judge Response:\n{judge_results[0]}\n\n"
            log_text += f"# Judge Decision:\n{judge_decisions[0]}\n\n"
            log_text += f"# RF Final Score Value:\n{score_vals[0]}\n\n"
            self.writer.add_text("policy_evaluator/individual_evaluation", log_text, self.global_step)
            logger.debug(f"[STEP {self.global_step}] [Policy Evaluator Full Log]: \n\n{log_text}")

            # --- Step 8: Metrics Logging Section ---
            # Compute the percentage of policy outputs with a valid p_correct (not "N/A")
            num_valid_p_correct = sum(1 for val in p_correct_dict.values() if val != "N/A")
            p_correct_exists_percentage = (num_valid_p_correct / len(p_correct_dict)) * 100 if p_correct_dict else 0.0
            self.writer.add_scalar("policy_evaluator/p_correct_exists_percentage", p_correct_exists_percentage, self.global_step)
            logger.info(f"p_correct exists percentage: {p_correct_exists_percentage:.2f}%")


            # --- Reference Guided Evaluation Metrics ---
            rg_total = len(rg_final_answers)
            rg_match_count = 0
            rg_yes_count = 0
            for answer in rg_final_answers:
                match = verification_regex.match(answer.strip())
                if match is not None:
                    rg_match_count += 1
                    if match.group(1).lower() == "yes":
                        rg_yes_count += 1

            rg_match_percentage = (rg_match_count / rg_total) * 100 if rg_total > 0 else 0.0
            rg_yes_percentage = (rg_yes_count / rg_match_count) * 100 if rg_match_count > 0 else 0.0

            # Log the RG metrics.
            self.writer.add_scalar("policy_evaluator/extract_rg_verify_resp", rg_match_percentage, self.global_step)
            self.writer.add_scalar("policy_evaluator/extract_rg_verify_yes", rg_yes_percentage, self.global_step)

            # Compute percentage (only if there is at least one valid RG verification)
            rg_match_iscorrect_percentage = (
                (matching_count / valid_rg_verification_count) * 100 if valid_rg_verification_count > 0 else 100
            )
            self.writer.add_scalar("policy_evaluator/rg_match_iscorrect_percentage", rg_match_iscorrect_percentage, self.global_step)
            logger.info(f"RG match iscorrect percentage: {rg_match_iscorrect_percentage:.2f}%") # PRIMARY METRIC FOR RG EVAL GROUND TRUTH EVALUATION. When able to parse, how often does rg eval match iscorrect rating?


            # --- Reference Free Evaluation Metrics ---
            total_rf = len(rf_final_answers)
            valid_rf_count = 0          # Count of RF answers with a valid extraction
            matching_rf_count = 0       # Overall matching count for valid extractions
            matching_rf_rg_count = 0

            # Counters for "yes" extractions
            yes_extraction_count = 0
            yes_and_gt_correct_count = 0

            # Counters for "no" extractions
            no_extraction_count = 0
            no_and_not_gt_correct_count = 0

            # Counters for ground truth:
            gt_positive_total = 0  # Number of valid extractions with gt_iscorrect True
            gt_negative_total = 0  # Number of valid extractions with gt_iscorrect False

            for j, answer in enumerate(rf_final_answers):
                policy_idx = rf_policy_indices[j]
                gt_iscorrect = outputs[policy_idx].get("iscorrect", False)
                rg_final_answer = rg_final_answers[policy_idx].strip()
                rg_match = verification_regex.match(rg_final_answer.strip())                    
                m = verification_regex.match(answer.strip())
                if m is not None:
                    rf_verification = m.group(1).lower().strip()  # "yes" or "no"

                    if rf_verification == "yes" or rf_verification == "no":
                        valid_rf_count += 1 # measure how often we get a "yes" or "no" basically

                    rg_verification = rg_match.group(1).lower().strip() if rg_match else None

                    if rf_verification == rg_verification:
                        matching_rf_rg_count += 1

                    # Overall matching rate calculation (precision over all responses)
                    if (rf_verification == "yes" and gt_iscorrect) or (rf_verification == "no" and not gt_iscorrect):
                        matching_rf_count += 1

                    # Breakdown for "yes" and "no"
                    if rf_verification == "yes":
                        yes_extraction_count += 1
                        if gt_iscorrect:
                            yes_and_gt_correct_count += 1
                    else:
                        no_extraction_count += 1
                        if not gt_iscorrect:
                            no_and_not_gt_correct_count += 1

                    # For recall calculation over valid extractions
                    if gt_iscorrect:
                        gt_positive_total += 1
                    else:
                        gt_negative_total += 1

            # Precision-like metrics (computed over total responses)
            rf_eval_score_can_extract_verify = (valid_rf_count / total_rf * 100) if total_rf > 0 else 0.0
            rf_eval_score_match_gt_iscorrect = (matching_rf_count / total_rf * 100) if total_rf > 0 else 0.0
            rf_eval_score_match_rg = (matching_rf_rg_count / total_rf * 100) if total_rf > 0 else 0.0

            # Additional precision metrics over valid extractions:
            extract_rf_verify_yes = (yes_extraction_count / valid_rf_count * 100) if valid_rf_count > 0 else 0.0
            extract_rf_verify_gt_iscorrect = ( (gt_positive_total) / valid_rf_count * 100) if valid_rf_count > 0 else 0.0
            extract_rf_verify_yes_and_gt_iscorrect = (yes_and_gt_correct_count / yes_extraction_count * 100) if yes_extraction_count > 0 else 0.0
            extract_rf_verify_no_and_not_gt_iscorrect = (no_and_not_gt_correct_count / no_extraction_count * 100) if no_extraction_count > 0 else 0.0

            # Now, compute recall metrics:
            # For "yes" recall: out of all ground truth positives, how many did we extract as "yes"?
            extract_rf_verify_yes_recall = (yes_and_gt_correct_count / gt_positive_total * 100) if gt_positive_total > 0 else 0.0
            # For "no" recall: out of all ground truth negatives, how many did we extract as "no"?
            extract_rf_verify_no_recall = (no_and_not_gt_correct_count / gt_negative_total * 100) if gt_negative_total > 0 else 0.0

            # Log all metrics.
            self.writer.add_scalar("policy_evaluator/rf_eval_score_can_extract_verify", rf_eval_score_can_extract_verify, self.global_step) # verify answer format is valid

            self.writer.add_scalar("policy_evaluator/rf_eval_score_match_gt_iscorrect", rf_eval_score_match_gt_iscorrect, self.global_step) # PRIMARY POLICY EVALUATOR METRIC: accurate evaluation score over all rf responses. Consider step 100 as the baseline for learning as that is when the policy AND policy evaluator get the format down and actually focus on learning
            self.writer.add_scalar("policy_evaluator/rf_eval_score_match_rg", rf_eval_score_match_rg, self.global_step) # CO-PRIMARY POLICY EVALUATOR METRIC: how often does the rf eval match the rg eval

            self.writer.add_scalar("policy_evaluator/extract_rf_verify_yes", extract_rf_verify_yes, self.global_step) # how often rf eval is just saying yes or no when there is a rf evaluation score
            self.writer.add_scalar("policy_evaluator/extract_rf_verify_gt_iscorrect", extract_rf_verify_gt_iscorrect, self.global_step) # how often rf eval should be when there is a rf evaluation score
            self.writer.add_scalar("policy_evaluator/extract_rf_verify_yes_and_gt_iscorrect", extract_rf_verify_yes_and_gt_iscorrect, self.global_step) # precision on when rf eval says answer is correct, how often it gt say it's correct?
            self.writer.add_scalar("policy_evaluator/extract_rf_verify_no_and_not_gt_iscorrect", extract_rf_verify_no_and_not_gt_iscorrect, self.global_step) # precision on when rf eval says answer is incorrect, how often it gt say it's incorrect?
            self.writer.add_scalar("policy_evaluator/extract_rf_verify_yes_recall", extract_rf_verify_yes_recall, self.global_step) # recall out of all gt = Correct, how often did evlauator say it's correct?
            self.writer.add_scalar("policy_evaluator/extract_rf_verify_no_recall", extract_rf_verify_no_recall, self.global_step) # recall out of all gt = Incorrect, how often did the evaluator also say it's incorrect?

            logger.info(
                f"[STEP {self.global_step}] RF extraction rate: {rf_eval_score_can_extract_verify:.2f}% over {total_rf} responses, "
                f"match rate: {rf_eval_score_match_gt_iscorrect:.2f}% over {valid_rf_count} valid extractions, "
                f"yes percentage: {extract_rf_verify_yes:.2f}% ({yes_extraction_count}/{valid_rf_count}), "
                f"GT iscorrect rate: {extract_rf_verify_gt_iscorrect:.2f}% ({gt_positive_total}/{valid_rf_count}), "
                f"yes+GT precision: {extract_rf_verify_yes_and_gt_iscorrect:.2f}% ({yes_and_gt_correct_count}/{yes_extraction_count}), "
                f"no+notGT precision: {extract_rf_verify_no_and_not_gt_iscorrect:.2f}% ({no_and_not_gt_correct_count}/{no_extraction_count}), "
                f"yes recall: {extract_rf_verify_yes_recall:.2f}% ({yes_and_gt_correct_count}/{gt_positive_total}), "
                f"no recall: {extract_rf_verify_no_recall:.2f}% ({no_and_not_gt_correct_count}/{gt_negative_total})"
            )



            # General format metrics
            # Compute the percentage of RG responses where extraction was successful. (answer existed)
            num_rg_extracted = sum(1 for flag in rg_extraction_flags if flag)
            extract_final_rg_resp_percentage = (num_rg_extracted / len(rg_extraction_flags)) * 100 if rg_extraction_flags else 0.0
            self.writer.add_scalar("policy_evaluator/extract_final_rg_resp", extract_final_rg_resp_percentage, self.global_step)
            logger.info(f"RG extraction success: {extract_final_rg_resp_percentage:.2f}% of responses")

            # Compute the percentage of RF responses where extraction was successful.
            num_rf_extracted = sum(1 for flag in rf_extraction_flags if flag)
            extract_final_rf_resp_percentage = (num_rf_extracted / len(rf_extraction_flags)) * 100 if rf_extraction_flags else 0.0
            self.writer.add_scalar("policy_evaluator/extract_final_rf_resp", extract_final_rf_resp_percentage, self.global_step)
            logger.info(f"RF extraction success: {extract_final_rf_resp_percentage:.2f}% of responses")


            # --- Additional Metrics for Judge Decisions ---
            total_judge_decisions = len(judge_decisions)
            valid_judge_decisions = [d for d in judge_decisions if d is not None]
            num_valid = len(valid_judge_decisions)
            percent_judge_valid = (num_valid / total_judge_decisions) * 100 if total_judge_decisions > 0 else 0.0
            num_judge_1 = sum(1 for d in valid_judge_decisions if d == 1.0)
            num_judge_0 = sum(1 for d in valid_judge_decisions if d == 0.0)
            percent_judge_1 = (num_judge_1 / total_judge_decisions) * 100 if total_judge_decisions > 0 else 0.0
            percent_judge_0 = (num_judge_0 / total_judge_decisions) * 100 if total_judge_decisions > 0 else 0.0

            self.writer.add_scalar("policy_evaluator/judge_valid_percentage", percent_judge_valid, self.global_step)
            self.writer.add_scalar("policy_evaluator/judge_1", percent_judge_1, self.global_step)
            self.writer.add_scalar("policy_evaluator/judge_0", percent_judge_0, self.global_step)
            self.writer.add_scalar("policy_evaluator/judge_0_or_1", percent_judge_valid, self.global_step)
            
            # Log the percentage of judge decisions that equal the rule-based iscorrect score. # Proxy metric for if the judge is doing well or not, but hard to measure if the reasoning evaluations are making sense or not. Need to log
            if total_judge_decisions > 0:
                judge_equals_iscorrect_percentage = iscorrect_match_judge_decisions / total_judge_decisions * 100
            else:
                judge_equals_iscorrect_percentage = 0.0
            self.writer.add_scalar("policy_evaluator/judge_equals_iscorrect_score", judge_equals_iscorrect_percentage, self.global_step)
            logger.info(f"Judge decisions equal rule-based iscorrect score: {judge_equals_iscorrect_percentage:.2f}% ({iscorrect_match_judge_decisions}/{total_judge_decisions})")

            # PRIMARY JUDGE METRIC: minimum baseline - say evaluation response is wrong when the final evaluation is wrong by ground truth (didn't even guess the answer right 50/50). This metric should be high - when the evaluation is obviously incorrect, the judge should also say "no". only problem is if rg_eval is also wrong, so check policy_evaluator/rg_match_iscorrect_percentage as well to ensure it's high
            if total_iscorrect_no_count > 0:
                judge_no_when_iscorrect_no_percentage = total_iscorrect_no_judge_no_count / total_iscorrect_no_count * 100
            else:
                judge_no_when_iscorrect_no_percentage = 0.0
            self.writer.add_scalar("policy_evaluator/judge_no_when_iscorrect_no_percentage", judge_no_when_iscorrect_no_percentage, self.global_step)
            logger.info(f"Judge decisions = 0 when rule-based iscorrect score = 0: {judge_no_when_iscorrect_no_percentage:.2f}% ({total_iscorrect_no_judge_no_count}/{total_iscorrect_no_count})")

            logger.info(f"Judge decisions: {percent_judge_1:.2f}% 1's, {percent_judge_0:.2f}% 0's, {percent_judge_valid:.2f}% valid (0 or 1) out of {total_judge_decisions} decisions")
        # --- END OF POLICY EVALUATOR BLOCK ---


        # GRPO
        self.writer.add_scalar("grpo/raw_policy_rewards", np.mean(scores), self.global_step) # log the raw reward before normalization (if there is normalization). basically the iscorrect %
        if self.cfg.use_grpo or self.cfg.enable_per_prompt_group_normalization:
            # grpo reward normalization
            pass_at_n_dict = defaultdict(list) # recompute pass_at_n_dict
            for prompt, score in zip(prompts, scores):
                pass_at_n_dict[prompt].append(score)

            # Now, apply GRPO normalization using the updated groups.
            for i, prompt in enumerate(prompts):
                group_scores = pass_at_n_dict[prompt]
                group_mean = np.mean(group_scores)
                group_std = np.std(group_scores)
                if group_std > 0:
                    scores[i] = (scores[i] - group_mean) / group_std
                else:
                    scores[i] = scores[i]  # no change if there is no variation


        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_repeat_score": sum(repeat_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # make histogram for correct and incorrect response length
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # make a pre-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # compute policy prompts, responses, and scores
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor, extra in zip(prompts, responses, score_tensors, extras):
            if (self.cfg.use_policy_evaluator and not self.cfg.train_policy_w_ground_truth_not_evaluator) or extra["answer"] != "[NO GT ANSWER]": # only keep responses that have a valid GT answer or if using the policy evaluator
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)


        # Count the # of experiences in replay buffer because too many policy experiences --> policy mode instead of evaluating mode
        self.writer.add_scalar(
            "experiences/policy_experiences_count_before_balancing", len(res_score_tensors), self.global_step
        )
        self.writer.add_scalar(
            "experiences/evaluator_experiences_count_before_balancing", len(evaluator_res_score_tensors), self.global_step
        )
        res_score_tensors_len_before_balancing = len(res_score_tensors)
        evaluator_res_score_tensors_len_before_balancing = len(evaluator_res_score_tensors)
        if self.cfg.balance_policy_and_evaluator_experiences:
            if len(res_score_tensors) > len(evaluator_res_score_tensors):
                n = len(evaluator_res_score_tensors)
                if n > 0:  # Only sample if there are samples available
                    combined = list(zip(res_prompts, res_responses, res_score_tensors))
                    sampled = random.sample(combined, n)
                    res_prompts, res_responses, res_score_tensors = map(list, zip(*sampled))
                else:
                    # Handle the case when evaluator experiences are empty
                    res_prompts, res_responses, res_score_tensors = [], [], []
            elif len(res_score_tensors) < len(evaluator_res_score_tensors):
                n = len(res_score_tensors)
                if n > 0:
                    combined = list(zip(evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors))
                    sampled = random.sample(combined, n)
                    evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors = map(list, zip(*sampled))
                else:
                    # Handle the case when policy experiences are empty
                    evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors = [], [], []

        self.writer.add_scalar(
            "experiences/policy_experiences_count_after_balancing", len(res_score_tensors), self.global_step
        )
        self.writer.add_scalar(
            "experiences/evaluator_experiences_count_after_balancing", len(evaluator_res_score_tensors), self.global_step
        )
        self.writer.add_scalar("experiences/policy_experiences_percent_used", len(res_score_tensors) / res_score_tensors_len_before_balancing * 100 if res_score_tensors_len_before_balancing > 0 else 0.0, self.global_step)
        self.writer.add_scalar("experiences/evaluator_experiences_percent_used", len(evaluator_res_score_tensors) / evaluator_res_score_tensors_len_before_balancing * 100 if evaluator_res_score_tensors_len_before_balancing > 0 else 0.0, self.global_step)

        return res_prompts, res_responses, res_score_tensors, evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors


    # This is when each query is evaluated separately.
    @override
    async def custom_query_reward_fn(
        self,
        query_prompts: List[str],
        query_outputs: List[Any],
        query_output_extras: List[dict],
        policy_outputs: List[str],
        policy_custom_rewards: List[torch.Tensor],
        R: int,  # Number of policy completions per query
    ) -> List[torch.Tensor]:
        """
        Compute custom query rewards in a format consistent with custom_reward_fn.
        
        For each query:
        1. If valid_prompt is False, the reward is 0.0.
        2. Otherwise, compute avg_success_reward as the average of the last-token reward 
            from the corresponding R policy outputs (from policy_custom_rewards). 
            Then, compute:
            
            query_reward = 1 - 2 * abs(0.5 - avg_success_reward)
            
            (Optionally clamped to [0.0, 1.0].)
        
        Returns:
        A list of reward tensors, one per query output, where each tensor has zeros for all tokens
        except its last token, which is set to the computed query reward.



        Rewards necessary:
        1. Diversity reward from previous query responses


        Rewards unclear:
        1. Format
        2. GPT-4o evaluator score
        3. Query difficulty (moderate difficulty, hard difficulty)
        4. Query Preference Reward
        5. Query Solvability Reward
        6. Logical correctness reward


        Rewards potentially worth trying:
        - [PROXY - logical correctness for now] Query Correctness - Verifying the verification script - Avg success from some query evaluator (to see if they arrive at the same expected correct answer) given the solution and such
        - [TODO] Query Simplicity and Elegance - Average # of tokens the solution is to reach the target answer
        - [PROXY - GPT-4o evaluator score] Query Impact - Does it contribute to the policy network’s understanding of mathematics?
        - [PROXY - QUERY DIFFICULTY] REWARD: Policy Moderate Difficulty - This can be based on # of attempts to reach the solution
        - [TODO] REWARD: Policy High Engagement - Perhaps how much thinking was required by the policy?


        """
        reward_tensors = []
        query_outputs_responses = []
        query_decision_rewards = []
        query_reward_responses = []
        query_difficulty_rewards = []
        lm_eval_rewards_list = []
        non_stop_count = 0  # count of query outputs whose stop_reason is not "stop"
        

        # --- NEW: Compute Raw Average Policy Response Length for each valid query ---
        # We will compute the raw (non-normalized) token count for each policy response,
        # then average per query group.
        raw_avg_policy_response_lengths = [0.0] * len(query_prompts)
        valid_indices = []
        for i in range(len(query_prompts)):
            # Check if the query prompt is valid.
            if query_output_extras[i].get("valid_prompt", False):
                valid_indices.append(i)
                group_policy_lengths = []
                start = i * R
                end = (i + 1) * R
                for response in policy_outputs[start:end]:
                    # Tokenize without padding so we get the true token count.
                    token_ids = self._tokenize(response, self.cfg.generate_max_len, padding=False)["input_ids"]
                    group_policy_lengths.append(len(token_ids))
                if group_policy_lengths:
                    raw_avg_policy_response_lengths[i] = sum(group_policy_lengths) / len(group_policy_lengths)
                else:
                    raw_avg_policy_response_lengths[i] = 0.0
            else:
                raw_avg_policy_response_lengths[i] = 0.0


        # --- COMPUTING PREFERENCES FIRST --- #
        # Step 1: Compute average success reward (query average success) for each query.
        preference_rewards = [None] * len(query_prompts)
        avg_success_rewards = [0.0] * len(query_prompts)
        valid_indices = []
        for i in range(len(query_prompts)):
            group_rewards = policy_custom_rewards[i * R : (i + 1) * R]
            last_rewards = [max(0.0, min(r[-1].item(), 1.0)) for r in group_rewards if r.numel() > 0]
            if len(last_rewards) == 0:
                avg_success = 0.0
            else:
                avg_success = sum(last_rewards) / len(last_rewards)
            avg_success_rewards[i] = avg_success

            # Check if the prompt is valid.
            if not query_output_extras[i].get("valid_prompt", False):
                preference_rewards[i] = -0.5  # Default reward for invalid prompts.
            else:
                preference_rewards[i] = 0.5  # Fallback reward for valid queries before ranking.
                valid_indices.append(i)

        # default outputs for logging
        all_criteria_output = ""
        max_benefit_output = ""
        diversity_output = ""
        clarity_output = ""
        logical_output = ""
        educational_output = ""
        ranking_output = ""
        common_context = query_prompts[0]

        # Step 2: For valid queries, build one ranking prompt that includes the full query context and the Query Average Success. # Not really necessary currently - doesn't make a difference
        if valid_indices:
            # Use the first query prompt as context (assuming they are all the same).
            common_context = query_prompts[0]
            ranking_items = []
            # Build a numbered list with both the response and its Query Average Success.
            for rel_idx, idx in enumerate(valid_indices):
                # response_text = query_outputs[idx]['final_answer']
                response_text = "<think>" + query_outputs[idx]['response']
                avg_success_str = f"{avg_success_rewards[idx]:.2f}"
                ranking_items.append(
                    f"Query Assistant Response {rel_idx}: {response_text}\n\nQuery Average Policy Assistant Success Rate: {avg_success_str}\n\nQuery Average Policy Assistant Response Length: {raw_avg_policy_response_lengths[idx]:.0f} tokens\n\n"
                )

        # --- COMPUTING DIVERSITY REWARDS --- #
        diversity_rewards = [0.0] * len(query_prompts)
        for idx in valid_indices:
            diversity_rewards[idx] = 0.5

        if valid_indices:
            try:
                # Build a diversity prompt similar to the preference prompt.
                diversity_prompt = (
                    "You are GPT-4o. Your task is to evaluate the diversity of the following Query Assistant responses compared "
                    "to the provided context. The context below includes previous Query Assistant responses and their average success scores. "
                    "Please analyze the current responses by separating the current responses into two groups: 1) diverse - which means that responses in this group are more diverse than the previous Query Assistant responses and the other current responses in the non-diverse group and 2) non-diverse - which means that responses in this group are quite similar to the previous Query Assistant responses and at least less diverse than the other responses in the diverse group. Then, return a comma-separated list of indices (starting at 0) corresponding to the responses that "
                    "are considered diverse / in the diverse group (i.e. introduce novel topics, new skills and techniques to solve, unique aspects). "
                    "Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags."
                    "\n\nContext:\n" + f"{common_context}\n\nCurrent Responses:\n" + "\n".join(ranking_items)
                )

                diversity_data = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": diversity_prompt,
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                diversity_url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
                diversity_output = ""
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.post(diversity_url, json=diversity_data)
                        diversity_response_content = response.json()['choices'][0]['message']['content']
                        diversity_output = diversity_response_content
                        break
                    except Exception as e:
                        logger.error(f"Error in diversity ranking request attempt {attempt}: {e}")
                        if attempt == max_retries - 1:
                            diversity_output = ""
                if diversity_output:
                    match = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', diversity_output, re.DOTALL)
                    if match:
                        try:
                            diversity_str = match.group(1)
                            # Convert each token to int only if it's a digit
                            diversity_list = [int(x.strip()) for x in diversity_str.split(",") if x.strip().isdigit()]
                            # Validate that each index is within the valid range.
                            valid_diversity_indices = [i for i in diversity_list if 0 <= i < len(valid_indices)]
                            logger.info(f"[Diversity rewards] Valid diversity indices: {valid_diversity_indices} out of all valid_indices: {valid_indices}")
                            for rel_idx, idx in enumerate(valid_indices):
                                diversity_rewards[idx] = 1.0 if rel_idx in valid_diversity_indices else 0.0
                        except Exception as e:
                            logger.error(f"Error processing diversity ranking output: {e}")
                    else:
                        logger.error("Could not extract diversity ranking from diversity evaluator response.")
                    
                    # Log the diversity evaluator interaction for debugging.
                    diversity_output = f"--- DIVERSITY EVALUATOR PROMPT: {diversity_prompt}\n\n--- DIVERSITY EVALUATOR RESPONSE: {diversity_output}"
                    logger.info(f"Diversity ranking output: {diversity_output}")
                    logger.info(f"Diversity rewards: {diversity_rewards}")
            except Exception as e:
                logger.error(f"Error in diversity reward computation block: {e}")


        # --- NEW: COMPUTING MAX BENEFIT/MIN HARM REWARDS --- #
        max_benefit_min_harm_rewards = [0.0] * len(query_prompts)
        if valid_indices:
            for idx in valid_indices:
                max_benefit_min_harm_rewards[idx] = 0.5
            try:
                max_benefit_prompt = (
                    "You are GPT-4o. Your task is to evaluate the following Query Assistant responses based on how well they maximize benefit and minimize harm for both the user and the Policy Assistant. "
                    "Please separate the responses into two groups: 1) those that effectively and uncontentiously maximize benefit and minimize harm (assign reward 1.0) and 2) those that might not or do not (assign reward 0.0). "
                    "Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags as a comma-separated list of response indices (starting at 0) that belong to the first group."
                    "\n\nContext:\n" + f"{common_context}\n\nCurrent Responses:\n" + "\n".join(ranking_items)
                )
                data_max_benefit = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": max_benefit_prompt,
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
                max_benefit_output = ""
                for attempt in range(3):
                    try:
                        response = requests.post(url, json=data_max_benefit)
                        max_benefit_response_content = response.json()['choices'][0]['message']['content']
                        max_benefit_output = max_benefit_response_content
                        break
                    except Exception as e:
                        logger.error(f"Error in max benefit ranking request attempt {attempt}: {e}")
                        if attempt == 2:
                            max_benefit_output = ""
                if max_benefit_output:
                    match = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', max_benefit_output, re.DOTALL)
                    if match:
                        try:
                            max_benefit_str = match.group(1)
                            max_benefit_list = [int(x.strip()) for x in max_benefit_str.split(",") if x.strip().isdigit()]
                            valid_max_benefit_indices = [i for i in max_benefit_list if 0 <= i < len(valid_indices)]
                            logger.info(f"[Max Benefit/Min Harm rewards] Valid indices: {valid_max_benefit_indices} out of {valid_indices}")
                            for rel_idx, idx in enumerate(valid_indices):
                                max_benefit_min_harm_rewards[idx] = 1.0 if rel_idx in valid_max_benefit_indices else 0.0
                        except Exception as e:
                            logger.error(f"Error processing max benefit ranking output: {e}")
                    else:
                        logger.error("Could not extract max benefit ranking from evaluator response.")
                    max_benefit_output = f"--- MAX BENEFIT/MIN HARM EVALUATOR PROMPT: {max_benefit_prompt}\n\n--- RESPONSE: {max_benefit_output}"
                    logger.info(max_benefit_output)
                else:
                    logger.error("No response for max benefit/min harm evaluator.")
            except Exception as e:
                logger.error(f"Error in max benefit/min harm reward computation: {e}")

        # --- NEW: COMPUTING CLARITY REWARDS --- #
        clarity_rewards = [0.0] * len(query_prompts)
        if valid_indices:
            for idx in valid_indices:
                clarity_rewards[idx] = 0.5
            try:
                clarity_prompt = (
                    "You are GPT-4o. Your task is to evaluate the following Query Assistant responses based on their clarity, specifically task generation criteria #4. "
                    "Please separate the responses into two groups: 1) tasks that are clear and easy to understand (assign reward 1.0) and 2) tasks that are unclear or confusing (assign reward 0.0). "
                    "Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags as a comma-separated list of response indices (starting at 0) that belong to the clear group."
                    "\n\nContext:\n" + f"{common_context}\n\nCurrent Responses:\n" + "\n".join(ranking_items)
                )
                data_clarity = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": clarity_prompt,
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
                clarity_output = ""
                for attempt in range(3):
                    try:
                        response = requests.post(url, json=data_clarity)
                        clarity_response_content = response.json()['choices'][0]['message']['content']
                        clarity_output = clarity_response_content
                        break
                    except Exception as e:
                        logger.error(f"Error in clarity ranking request attempt {attempt}: {e}")
                        if attempt == 2:
                            clarity_output = ""
                if clarity_output:
                    match = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', clarity_output, re.DOTALL)
                    if match:
                        try:
                            clarity_str = match.group(1)
                            clarity_list = [int(x.strip()) for x in clarity_str.split(",") if x.strip().isdigit()]
                            valid_clarity_indices = [i for i in clarity_list if 0 <= i < len(valid_indices)]
                            logger.info(f"[Clarity rewards] Valid clarity indices: {valid_clarity_indices} out of {valid_indices}")
                            for rel_idx, idx in enumerate(valid_indices):
                                clarity_rewards[idx] = 1.0 if rel_idx in valid_clarity_indices else 0.0
                        except Exception as e:
                            logger.error(f"Error processing clarity ranking output: {e}")
                    else:
                        logger.error("Could not extract clarity ranking from evaluator response.")
                    clarity_output = f"--- CLARITY EVALUATOR PROMPT: {clarity_prompt}\n\n--- RESPONSE: {clarity_output}"
                    logger.info(clarity_output)
                else:
                    logger.error("No response for clarity evaluator.")
            except Exception as e:
                logger.error(f"Error in clarity reward computation: {e}")

        # --- NEW: COMPUTING EDUCATIONAL REWARDS --- #
        educational_rewards = [0.0] * len(query_prompts)
        if valid_indices:
            for idx in valid_indices:
                educational_rewards[idx] = 0.5
            try:
                educational_prompt = (
                    "You are GPT-4o. Your task is to evaluate the following Query Assistant responses based on their educational value, specifically task generation success criteria #6. "
                    "Please separate the responses into two groups: 1) those that uncontentiously provide high educational value (assign reward 1.0) and 2) those that might not or do not (assign reward 0.0). "
                    "Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags as a comma-separated list of response indices (starting at 0) that belong to the high educational value group."
                    "\n\nContext:\n" + f"{common_context}\n\nCurrent Responses:\n" + "\n".join(ranking_items)
                )
                data_educational = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": educational_prompt,
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
                educational_output = ""
                for attempt in range(3):
                    try:
                        response = requests.post(url, json=data_educational)
                        educational_response_content = response.json()['choices'][0]['message']['content']
                        educational_output = educational_response_content
                        break
                    except Exception as e:
                        logger.error(f"Error in educational ranking request attempt {attempt}: {e}")
                        if attempt == 2:
                            educational_output = ""
                if educational_output:
                    match = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', educational_output, re.DOTALL)
                    if match:
                        try:
                            educational_str = match.group(1)
                            educational_list = [int(x.strip()) for x in educational_str.split(",") if x.strip().isdigit()]
                            valid_educational_indices = [i for i in educational_list if 0 <= i < len(valid_indices)]
                            logger.info(f"[Educational rewards] Valid educational indices: {valid_educational_indices} out of {valid_indices}")
                            for rel_idx, idx in enumerate(valid_indices):
                                educational_rewards[idx] = 1.0 if rel_idx in valid_educational_indices else 0.0
                        except Exception as e:
                            logger.error(f"Error processing educational ranking output: {e}")
                    else:
                        logger.error("Could not extract educational ranking from evaluator response.")
                    educational_output = f"--- EDUCATIONAL EVALUATOR PROMPT: {educational_prompt}\n\n--- RESPONSE: {educational_output}"
                    logger.info(educational_output)
                else:
                    logger.error("No response for educational evaluator.")
            except Exception as e:
                logger.error(f"Error in educational reward computation: {e}")


        # --- COMPUTING LOGICAL CORRECTNESS REWARDS --- #
        # Initialize logical correctness rewards to 0.0 for each query.
        logical_rewards = [0.0] * len(query_prompts)
        for idx in valid_indices:
            logical_rewards[idx] = 0.5

        if valid_indices:
            # Build a list of details for each valid query.
            logical_items = []
            for rel_idx, idx in enumerate(valid_indices):
                final_answer = query_outputs[idx]['final_answer']
                logical_items.append(
                    f"Query Assistant Response {rel_idx}: {final_answer}\n\n"
                )

            # Use the first query prompt as common context (assuming similar context across queries).
            common_context = query_prompts[0]
            logical_prompt = (
                "You are GPT-4o. Your task is to evaluate the logical correctness of the following Query Assistant responses' answer values based solely on their corresponding solutions and prompts, specifically task generation success criteria #5. "
                "For each response, decide if the proposed solution to the prompt is a full and complete proof and that the expected answer is uncontentiously the one and only correct answer to the prompt. "
                "Show your step-by-step reasoning inside <think> and </think> tags, then return your final answer inside <answer> tags as a comma-separated list of indices (starting at 0) corresponding to responses that are uncontentiously logically correct. "
                "For example, if responses 0 and 2 are uncontentiously logically correct, you should return: 0,2.\n\n"
                "Context:\n" + common_context + "\n\n"
                "Responses:\n" + "".join(logical_items)
            )

            data_logical = {
                "temperature": 0.0,
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": logical_prompt,
                    }]
                }],
                "stop": ["<|im_end|>"],
                "n": 1,
                "model": "dev-gpt-4o-vision-2024-05-13"
            }
            logical_output = ""
            try:
                response_logical = requests.post(url, json=data_logical)
                logical_output_content = response_logical.json()['choices'][0]['message']['content']
                logical_output = logical_output_content  # For logging/debugging.
                match_logical = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', logical_output, re.DOTALL)
                if match_logical:
                    ranking_str = match_logical.group(1)
                    ranking_list = [int(x.strip()) for x in ranking_str.split(",") if x.strip().isdigit()]
                    n_valid = len(valid_indices)
                    logger.info(f"[Logical rewards] Valid ranking list indices: {ranking_list} out of {list(range(n_valid))}")
                    for rel_idx, idx in enumerate(valid_indices):
                        logical_rewards[idx] = 1.0 if rel_idx in ranking_list else 0.0
                else:
                    logger.error("Could not extract logical correctness indices from GPT-4o response.")
            except Exception as e:
                logger.error(f"Error in logical correctness evaluation: {e}")

            logical_output = f"--- LOGICAL EVALUATOR PROMPT: {logical_prompt}\n\n--- LOGICAL EVALUATOR RESPONSE: {logical_output}"
            logger.info(f"Logical evaluator output: {logical_output}")
            logger.info(f"Logical correctness rewards: {logical_rewards}")

        # --- NEW: COMPUTING ALL CRITERIA FULFILLED REWARD --- #
        # --- COMPUTING ALL CRITERIA FULFILLED REWARD --- #
        # Define a remote function for evaluating a single pair.
        @ray.remote(num_cpus=1)
        def evaluate_pair(data_pair):
            url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, json=data_pair)
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.error(f"Error in evaluate_pair attempt {attempt}: {e}")
                    if attempt == max_retries - 1:
                        return ""
            return ""

        all_criteria_fulfilled_rewards = [0.0] * len(query_prompts)
        if valid_indices:
            for idx in valid_indices:
                all_criteria_fulfilled_rewards[idx] = 0.5  # default value for valid queries

            # Create a shuffled copy of valid_indices to form random pairs.
            shuffled_valid_indices = valid_indices.copy()
            random.shuffle(shuffled_valid_indices)
            
            pair_rewards = {}  # Mapping from index to computed reward (0.0 or 1.0).
            pair_log_entries = []  # List of dictionaries for structured logging.
            all_criteria_output_log = []  # List to collect logging info for each pair.

            # Determine the number of complete pairs (if odd, the leftover remains at 0.5).
            num_pairs = len(shuffled_valid_indices) // 2
            pair_data_list = []
            pair_indices_list = []

            for i in range(num_pairs):
                idx1 = shuffled_valid_indices[2 * i]
                idx2 = shuffled_valid_indices[2 * i + 1]
                
                # Retrieve the ranking text corresponding to these valid indices.
                # (ranking_items was built in order corresponding to valid_indices.)
                ranking_text1 = "Query Assistant Response 0: " + ranking_items[ valid_indices.index(idx1) ].split(":", 1)[1].strip()
                ranking_text2 = "Query Assistant Response 1: " + ranking_items[ valid_indices.index(idx2) ].split(":", 1)[1].strip()
                current_pair_responses_str = ranking_text1 + "\n\n\n\n" + ranking_text2

                # Build the pair evaluation prompt.
                pair_prompt = f'''You are a Query Assistant Evaluator on behalf of the User who set a Policy Assistant Goal. Your task is to evaluate the Current Query Assistant Responses by selecting 1 response that fulfills MORE of the Task Generation Success Criteria provided in the Query Assistant Prompt better than the unselected response. Please separate the responses into two groups: 1) those that meet more of the criteria (assign reward 1.0) and 2) those that meet less of the criteria (assign reward 0.0). Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags as a response index (starting at 0) for which response fulfills more of the criteria. Note: fulfilling more higher-priority criteria should be considered as fulfilling more of the success criteria than fulfilling more lower-priority criteria.

QUERY ASSISTANT PROMPT:
{common_context}


CURRENT QUERY ASSISTANT RESPONSES:
{current_pair_responses_str}'''

                data_pair = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": pair_prompt,
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                pair_data_list.append(data_pair)
                pair_indices_list.append((idx1, idx2))
                # Create a structured log entry for this pair.
                pair_log_entries.append({
                    "pair_indices": (idx1, idx2),
                    "prompt": pair_prompt,
                    "response": None  # to be filled later
                })


            # Launch all pair evaluations in parallel.
            tasks = [evaluate_pair.remote(data_pair) for data_pair in pair_data_list]
            pair_outputs = ray.get(tasks)

            # Process each pair's output.
            for ((idx1, idx2), pair_output, log_entry) in zip(pair_indices_list, pair_outputs, pair_log_entries):
                log_entry["response"] = pair_output
                if pair_output:
                    match = re.search(r'<answer>\s*([01])\s*</answer>', pair_output, re.DOTALL)
                    if match:
                        decision = match.group(1).strip()
                        if decision == "0":
                            pair_rewards[idx1] = 1.0
                            pair_rewards[idx2] = 0.0
                        elif decision == "1":
                            pair_rewards[idx1] = 0.0
                            pair_rewards[idx2] = 1.0
                    else:
                        logger.error("Could not parse pair evaluator output; leaving rewards as default (0.5)")
                else:
                    logger.error("No response from pair evaluator; leaving rewards as default (0.5)")

            # Log the compiled pair evaluations.
            # all_criteria_output = json.dumps(pair_log_entries, indent=4)
            all_criteria_output = ""
            for entry in pair_log_entries:
                for key, value in entry.items():
                    all_criteria_output += f"#{key}:\n{value}\n\n"
            logger.info(f"--- ALL CRITERIA EVALUATOR:\n{all_criteria_output}")
            self.writer.add_text("query/all_criteria_evaluator_output", all_criteria_output, self.global_step)

            # Update the rewards for indices that were evaluated via pairing.
            for idx, reward in pair_rewards.items():
                all_criteria_fulfilled_rewards[idx] = reward

        avg_all_criteria = np.mean(all_criteria_fulfilled_rewards)
        self.writer.add_scalar("query/avg_all_criteria_fulfilled_reward", avg_all_criteria, self.global_step) # expect this to be around 0.5


                # old code for just mass evaluating all as one group basically
#             try:
#                 current_query_assistant_responses_str = "\n".join(ranking_items)
#                 num_invalid = len(query_prompts) - len(valid_indices)
#                 # winning_group_size = len(ranking_items) // 2
#                 winning_group_size = min(len(ranking_items), len(ranking_items) // 2 + num_invalid)  # I want the winning group size to be 50% of all responses
#                 all_criteria_prompt = f'''You are a Query Assistant Evaluator on behalf of the User who set a Policy Assistant Goal. Your task is to evaluate the Current Query Assistant Responses by selecting ~{winning_group_size} responses that fulfill MORE of the Task Generation Success Criteria provided in the Query Assistant Prompt than the unselected responses. Please separate the responses into two groups: 1) those that meet more of the criteria (assign reward 1.0) and 2) those that meet less of the criteria (assign reward 0.0). Show your step-by-step reasoning inside <think> and </think> tags, then provide your final answer inside <answer> and </answer> tags as a comma-separated list of response indices (starting at 0) that fulfill more of the criteria.
                    
# QUERY ASSISTANT PROMPT:

# {common_context}


# CURRENT QUERY ASSISTANT RESPONSES:

# {current_query_assistant_responses_str}'''
#                 data_all_criteria = {
#                     "temperature": 0.0,
#                     "max_tokens": 4096,
#                     "messages": [{
#                         "role": "user",
#                         "content": [{
#                             "type": "text",
#                             "text": all_criteria_prompt,
#                         }]
#                     }],
#                     "stop": ["<|im_end|>"],
#                     "n": 1,
#                     "model": "dev-gpt-4o-vision-2024-05-13"
#                 }
#                 url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
#                 all_criteria_output = ""
#                 for attempt in range(3):
#                     try:
#                         response = requests.post(url, json=data_all_criteria)
#                         all_criteria_response_content = response.json()['choices'][0]['message']['content']
#                         all_criteria_output = all_criteria_response_content
#                         break
#                     except Exception as e:
#                         logger.error(f"Error in all criteria evaluator request attempt {attempt}: {e}")
#                         if attempt == 2:
#                             all_criteria_output = ""
#                 if all_criteria_output:
#                     match = re.search(r'<answer>\s*([\d\s,]+)\s*</answer>', all_criteria_output, re.DOTALL)
#                     if match:
#                         try:
#                             all_criteria_str = match.group(1)
#                             all_criteria_list = [int(x.strip()) for x in all_criteria_str.split(",") if x.strip().isdigit()]
#                             valid_all_criteria_indices = [i for i in all_criteria_list if 0 <= i < len(valid_indices)]
#                             logger.info(f"[All Criteria Fulfilled Rewards] Valid indices: {valid_all_criteria_indices} out of {valid_indices}")
#                             for rel_idx, idx in enumerate(valid_indices):
#                                 # all_criteria_fulfilled_rewards[idx] = 1.0 if rel_idx in valid_all_criteria_indices else 0.0

#                                 if winning_group_size == len(ranking_items):
#                                     all_criteria_fulfilled_rewards[idx] = 1.0
#                                 else: # if winning group size is less than the total number of valid responses, pick and choose!
#                                     all_criteria_fulfilled_rewards[idx] = 1.0 if rel_idx in valid_all_criteria_indices else 0.0
#                         except Exception as e:
#                             logger.error(f"Error processing all criteria evaluator output: {e}")
#                     else:
#                         logger.error("Could not extract all criteria evaluator output.")
#                     all_criteria_output = f"--- ALL CRITERIA EVALUATOR PROMPT: {all_criteria_prompt}\n\n--- RESPONSE: {all_criteria_output}"
#                     logger.info(all_criteria_output)
#                 else:
#                     logger.error("No response for all criteria evaluator.")
#             except Exception as e:
#                 logger.error(f"Error in all criteria reward computation: {e}")
            
#             # Log the average value for all criteria fulfilled reward.
#             avg_all_criteria = np.mean([all_criteria_fulfilled_rewards[idx] for idx in valid_indices]) if valid_indices else 0.0
#             self.writer.add_scalar("query/avg_all_criteria_fulfilled_reward", avg_all_criteria, self.global_step)


        # --- NEW: Compute Query Solution Simplicity Length and Avg Policy Engagement Response Length Rewards --- #
        # Initialize reward arrays for all queries.
        solution_length_rewards = [0.0] * len(query_prompts)
        avg_policy_response_length_rewards = [0.0] * len(query_prompts)

        if valid_indices:
            # Compute solution lengths for valid queries.
            solution_lengths = []
            for idx in valid_indices:
                sol = query_output_extras[idx].get("solution", query_outputs[idx].get("final_answer", ""))
                logger.info(f"[Query Solution Simplicity Debug] Solution for query {idx}: {sol}\n\n")
                # Tokenize the solution to get its token count.
                if isinstance(sol, str) and sol:
                    tokens = self._tokenize(sol, self.cfg.generate_max_len, padding=False)["input_ids"]
                else:
                    tokens = []
                solution_lengths.append((idx, len(tokens)))
            
            # Normalization for solution lengths (smaller is better).
            lengths = [length for _, length in solution_lengths]
            min_sol_length = min(lengths)
            max_sol_length = max(lengths)
            for idx, length in solution_lengths:
                if max_sol_length == min_sol_length:
                    reward = 0.5 # Default reward for equal lengths.
                else:
                    reward = (max_sol_length - length) / (max_sol_length - min_sol_length)
                solution_length_rewards[idx] = reward

            # Compute average policy response length for each valid query.
            avg_response_lengths = []
            for idx in valid_indices:
                # For each valid query, extract its R policy responses.
                start = idx * R
                end = (idx + 1) * R
                policy_resps = [policy_outputs[i] for i in range(start, end)]
                # Tokenize each policy response.
                lengths = [
                    len(self._tokenize(resp, self.cfg.generate_max_len, padding=False)["input_ids"])
                    for resp in policy_resps
                ]
                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                avg_response_lengths.append((idx, avg_length))
            
            # Normalization for policy response lengths (longer is better).
            avg_lengths = [avg for _, avg in avg_response_lengths]
            min_avg_length = min(avg_lengths)
            max_avg_length = max(avg_lengths)
            for idx, avg in avg_response_lengths:
                if max_avg_length == min_avg_length:
                    reward = 0.5 # Default reward for equal lengths.
                else:
                    reward = (avg - min_avg_length) / (max_avg_length - min_avg_length)
                avg_policy_response_length_rewards[idx] = reward

            # Optionally, log the averages to TensorBoard.
            avg_sol_length = sum(lengths) / len(lengths)
            avg_policy_resp_length = sum(avg_lengths) / len(avg_lengths)
            self.writer.add_scalar("query/avg_solution_simplicity_response_length", avg_sol_length, self.global_step)
            self.writer.add_scalar("query/avg_policy_engagement_response_length", avg_policy_resp_length, self.global_step)


        # Initialize accumulators for logging component rewards
        total_difficulty_reward = 0.0
        total_preference_reward = 0.0
        total_solvability_reward = 0.0
        num_valid_queries = 0

        # Loop over each query output and corresponding extras.
        decided_count = 0
        for idx, (prompt, output, extra) in enumerate(zip(query_prompts, query_outputs, query_output_extras)):
            valid = extra.get("valid_prompt", False)
            response_content = ""
            query_difficulty_reward = 0.0
            decision = ""
            lm_eval_reward = 0.0
            query_lm_eval_task_success_prompt = ""
            if output["final_answer"] == "":
                computed_reward = -1.0
            elif not valid:
                computed_reward = -0.5 #0.0 #-0.5 #-1.0 #0.5 #0.0 #-1.0 #0.0 #-0.5 # - 1.0 is the default value for invalid prompts to incentivize formatted valid prompts! and also for normalization to not penalize formatted valid but bad prompts --> format instability
            else:
                num_valid_queries += 1
                computed_reward = 0.0 # default fallback if LM evaluator doesn't work

                # logger.info(f"[Query Reward Func] Data: {data}")
                # logger.info(f"[Query Reward Func] Response: {response_content}")


                # REWARD 1: Query Difficulty Moderate # for some reason it leads to query/query_valid_dict collapse
                # Gather corresponding policy custom rewards for this query.
                group_rewards = policy_custom_rewards[idx * R : (idx + 1) * R]
                # Extract the last token reward from each policy output.
                last_rewards = [max(0.0, min(r[-1].item(), 1.0)) for r in group_rewards if r.numel() > 0] # constrain and don't let format overbear on the reward
                if len(last_rewards) == 0:
                    avg_success_reward = 0.0
                else:
                    avg_success_reward = sum(last_rewards) / len(last_rewards)
                # Compute the query reward so that it is highest when avg_success_reward is near 0.5.
                # query_difficulty_reward = 1 - 2 * abs(0.5 - avg_success_reward)
                query_difficulty_reward = 1 - avg_success_reward # if avg_success_reward > 0.0 else 0.0 # hard > moderate now that we have solvability rewards as well # anti-policy correctness reward
                total_difficulty_reward += query_difficulty_reward
                # if avg_success_reward == 0.0:
                #     computed_reward = 0.0 # incentivize hard problems, but solvable ones


                # REWARD 0: Query Correctness Score from LM Evaluator (Stage 1: Frozen GPT-4o, Stage 2: Trained Query Network, Fallback frozen GPT-4o, Stage 3: Only Trained Query Network)
                data = {
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": '''You are GPT-4o. Your task is to evaluate a Query Assistant Response (and its corresponding Query Average Policy Assistant Success and Query Average Policy Assistant Response Length)—generated by a Qwen-2.5-7B base model—against the Task Generation Success Criteria defined in the Query Assistant Prompt.
                            
Please evaluate whether the Query Assistant Response and its corresponding Query Average Policy Assistant Success score and Query Average Policy Assistant Response Length (after running a couple policy rollouts and computing the average success and response length of those policy rollouts against the target success criteria in the Query Assistant Response) fulfills ALL of these criteria. Show your step-by-step reasoning inside <think> and </think> tags, then provide your final decision inside <answer> and </answer> tags as a single word: "yes" if it meets the criteria or "no" if it does not.

<think>
[Provide your detailed evaluation and reasoning here]
</think>
<answer>
[yes/no]
</answer>'''
                        }, {
                            "type": "text",
                            "text": f'''QUERY ASSISTANT PROMPT: {prompt}''',
                        }, {
                            "type": "text",
                            "text": f'''QUERY ASSISTANT RESPONSE: <think>{output['response']}\n\nQUERY AVERAGE POLICY ASSISTANT SUCCESS RATE: {avg_success_reward:.2f}\n\nQUERY AVERAGE POLICY ASSISTANT RESPONSE LENGTH: {raw_avg_policy_response_lengths[idx]} tokens'''
                        }]
                    }],
                    "stop": ["<|im_end|>"],
                    "n": 1,
                    "model": "dev-gpt-4o-vision-2024-05-13"
                }
                query_lm_eval_task_success_prompt = f'''You are GPT-4o. Your task is to evaluate a Query Assistant Response (and its corresponding Query Average Policy Assistant Success and Query Average Policy Assistant Response Length)—generated by a Qwen-2.5-7B base model—against the Task Generation Success Criteria defined in the Query Assistant Prompt.
                            
Please evaluate whether the Query Assistant Response and its corresponding Query Average Policy Assistant Success score and Query Average Policy Assistant Response Length (after running a couple policy rollouts and computing the average success and response length of those policy rollouts against the target success criteria in the Query Assistant Response) fulfills ALL of these criteria. Show your step-by-step reasoning inside <think> and </think> tags, then provide your final decision inside <answer> and </answer> tags as a single word: "yes" if it meets the criteria or "no" if it does not.

<think>
[Provide your detailed evaluation and reasoning here]
</think>
<answer>
[yes/no]
</answer>

QUERY ASSISTANT PROMPT: {prompt}

QUERY ASSISTANT RESPONSE: <think>{output['response']}\n\nQUERY AVERAGE POLICY ASSISTANT SUCCESS RATE: {avg_success_reward:.2f}\n\nQUERY AVERAGE POLICY ASSISTANT RESPONSE LENGTH: {raw_avg_policy_response_lengths[idx]} tokens
'''

                url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"
                response_content = ""
                for _ in range(3):
                    try:
                        response = requests.post(url, json=data)
                        response_content = response.json()['choices'][0]['message']['content']
                        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_content, re.DOTALL)
                        if match:
                            extracted_answer = match.group(1).strip()
                            decision = extracted_answer.lower()

                        if decision == "yes":
                            lm_eval_reward = 1.0
                            decided_count += 1
                        elif decision == "no":
                            lm_eval_reward = 0.0
                            decided_count += 1
                        
                        break
                    except Exception as e:
                        logger.error(f"Error in sending request: {e}")


                ### QUERY REWARDS ###
                ## Query-based rewards
                # --- LLM relative measures --- #
                ### 1. Query Correctness Reward (based on task generation success criteria # Note: a LM eval grader currently is not strong enough to give a good score based on all factors currently, so breaking it up into components)
                # computed_reward += lm_eval_reward # Can just track this as a proxy that the sub rewards are working as expected basically

                ### 0. (all criteria) All Criteria Fulfilled Reward
                computed_reward += all_criteria_fulfilled_rewards[idx]

                # ### 1.1 (maximize benefit, minimize harm)
                # computed_reward += max_benefit_min_harm_rewards[idx]

                # ### 1.2. (novel) Query Diversity reward (TODO: may need promoting policy diversity reward as well, but currently really not necessary to improve policy just yet). TODO: Ideally you use a more objective measure like embedding cosine similarity distance or something like in Diversifying AI paper (empirically having one group be 1.0 and another group be 0.0 seems to works well enough for now)
                # computed_reward += diversity_rewards[idx]  # diversity reward component. network WILL devolve to same type query if no diversity (ex. what is the Xth Fibonacci number) and increase X over time # confirmed this is necessary v1220 and beyond (when I ran v1220 with a bug in diversity reward so no diversity reward - devolved to asking the same question)

                # ### 1.4 (clear)
                # computed_reward += clarity_rewards[idx]

                # ### 1.5 logically correct
                # computed_reward += logical_rewards[idx]  # NEW logical correctness reward component.

                # ### 1.6 educational
                # computed_reward += educational_rewards[idx]



                # # --- objective measures --- #
                # ### 3. (educational) Query Short Solution Response Length Reward (based on Normalized Query Solution Length) = Educational value reward (shortcut simple query solution is ideal [insight], long policy thinking is ideal)
                # computed_reward += solution_length_rewards[idx] # how simple the solution is

                # ## Policy-based rewards - Query difficulty and solvability check
                # ### 4. (difficult yet tractable) Query Policy Solvability Reward (based on Query Avg Success Rate)
                # if avg_success_reward > 0.0: # aka query is not too hard/impossible
                #     computed_reward += 1.0 # solvability reward

                # ### 5. (difficult yet tractable) Anti-Policy Correctness Reward (based on Query Avg Success Rate)
                # computed_reward += query_difficulty_reward # difficulty reward # aka query is not too easy

                # ### 6. (educational) Anti-Policy Short Response Length Reward (based on Normalized Policy Avg Response Length)
                # computed_reward += avg_policy_response_length_rewards[idx] # how long it took to think for the policy # aka query requires policy to exercise deep thinking / search


                if avg_success > 0.0:
                    total_solvability_reward += 1.0

            lm_eval_rewards_list.append(lm_eval_reward)
            query_decision_rewards.append(computed_reward)
            query_reward_responses.append(f"\n\nREWARD 0:\n\n{all_criteria_output}\n\nREWARD 1:\n\n{max_benefit_output}\n\nREWARD 2:\n\n{diversity_output}\n\nREWARD 4:\n\n{clarity_output}\n\nREWARD 5:\n\n{logical_output}\n\nREWARD 6:\n\n{educational_output}\n\nPROXY REWARD:\n\nQUERY LM EVAL TASK SUCCESS PROMPT: \n\n{query_lm_eval_task_success_prompt}\n\nQUERY LM EVAL TASK SUCCESS RESPONSE: {response_content}")
            query_difficulty_rewards.append(query_difficulty_reward)

            # # NEW: Update query reward by the evaluator LM reward if it exists
            # evaluator_lm_reward = output.get("evaluator_lm_reward", 0.0)
            # if evaluator_lm_reward > 0.0: # if evaluator_lm_reward == 0.0, we just keep it as is
            #     computed_reward += evaluator_lm_reward # good format and potentially helpful, so *= query_difficulty score # multiplicative seems to be better than additive
            # elif evaluator_lm_reward < 0.0:
            #     computed_reward = evaluator_lm_reward # bad format or potentially harmful, so auto -0.5 or -1.0

            # Check the stop_reason field; if it's not "stop", count it.
            stop_reason = output.get("stop_reason", "")
            if stop_reason != "stop":
                non_stop_count += 1

            # Tokenize the query output to determine its token length.
            query_response = output.get("response", "")
            tokenized = self._tokenize(query_response, self.cfg.generate_max_len, padding=False)
            token_ids = tokenized["input_ids"]
            tensor_length = len(token_ids)
            reward_tensor = torch.zeros(tensor_length)
            if tensor_length > 0:
                reward_tensor[-1] = computed_reward
            reward_tensors.append(reward_tensor)
            query_outputs_responses.append(query_response)


        # Compute the percentage of queries that received a definitive decision (i.e. reward != 0.5)
        if query_decision_rewards:
            percentage_decided = decided_count / len(query_decision_rewards) * 100  # as a percentage
        else:
            percentage_decided = 0.0

        # Log the percentage to TensorBoard with the tag "query/reward_yes_or_no"
        self.writer.add_scalar("query/reward_yes_or_no", percentage_decided, self.global_step)

        # Compute and log the average non-stop count metric.
        if len(query_outputs) > 0:
            avg_query_non_stop_count = non_stop_count / len(query_outputs)
        else:
            avg_query_non_stop_count = 0.0
        self.writer.add_scalar("query/avg_non_stop_count", avg_query_non_stop_count, self.global_step)


        # Log overall LM evaluator reward.
        if lm_eval_rewards_list:
            avg_lm_eval_reward = sum(lm_eval_rewards_list) / len(lm_eval_rewards_list)
        else:
            avg_lm_eval_reward = 0.0
        self.writer.add_scalar("query/avg_lm_eval_reward", avg_lm_eval_reward, self.global_step)

        # After computing diversity_rewards (with defaults set to 0.5 for valid queries and updated to 1.0 where indicated)
        if valid_indices:
            # Calculate the average only over valid queries
            avg_diversity_reward = sum(diversity_rewards[idx] for idx in valid_indices) / len(valid_indices)
        else:
            avg_diversity_reward = 0.0
        self.writer.add_scalar("query/avg_diversity_reward", avg_diversity_reward, self.global_step)

        # Log overall logical correctness reward.
        if valid_indices:
            avg_logical_reward = sum(logical_rewards[idx] for idx in valid_indices) / len(valid_indices)
        else:
            avg_logical_reward = 0.0
        self.writer.add_scalar("query/avg_logical_reward", avg_logical_reward, self.global_step)

        # --- Log average values for the new reward components ---
        avg_max_benefit = np.mean([max_benefit_min_harm_rewards[idx] for idx in valid_indices]) if valid_indices else 0.0
        avg_clarity = np.mean([clarity_rewards[idx] for idx in valid_indices]) if valid_indices else 0.0
        avg_educational = np.mean([educational_rewards[idx] for idx in valid_indices]) if valid_indices else 0.0

        self.writer.add_scalar("query/avg_max_benefit_min_harm_reward", avg_max_benefit, self.global_step)
        self.writer.add_scalar("query/avg_clarity_reward", avg_clarity, self.global_step)
        self.writer.add_scalar("query/avg_educational_reward", avg_educational, self.global_step)

        # Log additional reward component averages.
        if num_valid_queries > 0:
            avg_difficulty_reward = total_difficulty_reward / num_valid_queries
            avg_preference_reward = total_preference_reward / num_valid_queries
            avg_solvability_reward = total_solvability_reward / num_valid_queries
        else:
            avg_difficulty_reward = avg_preference_reward = avg_solvability_reward = 0.0
        self.writer.add_scalar("query/avg_difficulty_reward", avg_difficulty_reward, self.global_step)
        self.writer.add_scalar("query/avg_preference_reward", avg_preference_reward, self.global_step)
        self.writer.add_scalar("query/avg_solvability_reward", avg_solvability_reward, self.global_step)

        return query_prompts, query_outputs_responses, reward_tensors, query_reward_responses, query_difficulty_rewards, preference_rewards, lm_eval_rewards_list, diversity_rewards, logical_rewards, max_benefit_min_harm_rewards, clarity_rewards, educational_rewards,solution_length_rewards, avg_policy_response_length_rewards, raw_avg_policy_response_lengths, all_criteria_fulfilled_rewards


    @override
    async def custom_evaluator_extract_reward_fn(
        self,
        evaluator_prompts: List[str],
        evaluator_responses: List[str],
        num_valid_options: List[int],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """
        Computes evaluator extraction rewards for the verified-solution extraction phase.
        
        For each evaluator response, this function:
        1. Checks that the response follows the format:
            <think> ... </think>
            <answer> ... </answer>
        2. Extracts the text inside the <answer> ... </answer> block.
        3. Verifies that the extracted text (after stripping and uppercasing) is a single letter that
            is in the valid range. Here we assume valid letters are the first (num_valid_options + 1) letters
            of the alphabet.
        4. Returns a reward tensor (with zeros for all tokens except its final token set to 1.0 if valid, or 0.0 otherwise).
        
        Returns:
        A tuple of (evaluator_prompts, evaluator_responses, reward_tensors).
        """
        reward_tensors = []
        # Process each evaluator response with its corresponding num_valid_options.
        for response, nvo in zip(evaluator_responses, num_valid_options):
            allowed_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: nvo + 1]
            tokenized = self._tokenize(response, self.cfg.generate_max_len, padding=False)
            seq_len = len(tokenized["input_ids"])
            r_tensor = torch.zeros(seq_len)

            if not response.startswith("<think>"):
                response = "<think>" + response
        
            # Use regex to match the expected format.
            # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
            regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
            match = re.search(regex, response, re.DOTALL)
            if match is not None and len(match.groups()) == 2 and response.endswith("</answer>"):
                answer_text = match.group(2).strip().upper()
                reward_value = 1.0 if len(answer_text) == 1 and answer_text in allowed_letters else 0.0
            else:
                reward_value = 0.0

            if seq_len > 0:
                r_tensor[-1] = reward_value
            reward_tensors.append(r_tensor)
        
        # For debugging: log average evaluator extraction reward.
        avg_reward = np.mean([r[-1].item() for r in reward_tensors if r.numel() > 0])
        self.writer.add_scalar("evaluator_extract/avg_reward", avg_reward, self.global_step)
        
        return evaluator_prompts, evaluator_responses, reward_tensors



    @override
    async def custom_evaluator_reward_fn(
        self,
        evaluator_prompts: List[str],
        evaluator_responses: List[str],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """
        Computes evaluator rewards for the check_verified_solution phase.
        For each evaluator response, this function verifies that the response follows the
        format "<think> ... </think>\n*<answer> ... </answer>" and then checks if the text inside
        the <answer> tags (after stripping and lowercasing) is either "yes" or "no".
        
        It returns the original evaluator_prompts, the evaluator_responses, and a list of reward tensors.
        Each reward tensor is all zeros except that the final token is set to the computed reward 
        (1.0 if the answer is either "yes" or "no", and 0.0 otherwise).
        """
        reward_tensors = []
        # Process each evaluator response.
        for response in evaluator_responses:
            # Tokenize the response to determine the sequence length.
            tokenized = self._tokenize(response, self.cfg.generate_max_len, padding=False)
            token_ids = tokenized["input_ids"]
            seq_len = len(token_ids)
            r_tensor = torch.zeros(seq_len)

            if not response.startswith("<think>"):
                response = "<think>" + response
            
            # Regex to check if the response follows the expected format.
            # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
            regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
            match = re.search(regex, response, re.DOTALL)
            if match is not None and len(match.groups()) == 2 and response.endswith("</answer>"):
                answer_text = match.group(2).strip().lower()
                # Check if the extracted answer is either 'yes' or 'no'
                reward_value = 1.0 if answer_text in {"yes", "no"} else 0.0
            else:
                reward_value = 0.0
            
            # Place the computed reward at the final token of the tensor.
            if seq_len > 0:
                r_tensor[-1] = reward_value
            reward_tensors.append(r_tensor)
        
        # Log the average reward for debugging.
        avg_reward = np.mean([r[-1].item() for r in reward_tensors if r.numel() > 0])
        self.writer.add_scalar("evaluator_check/avg_reward", avg_reward, self.global_step)
        
        return evaluator_prompts, evaluator_responses, reward_tensors


    @override # this override's RayPPOTrainer's generate_vllm
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        exp_type: str = "",
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams
        # read sampling params from self.cfg

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        # # --- NEW: Replace responses with GPT-4o responses if exp_type == "policy" --- #
        # if exp_type == "policy":
        #     url = "https://skilled-chigger-mentally.ngrok-free.app/api/send_request"

        #     async def get_gpt4o_response(prompt: str) -> str:
        #         data = {
        #             "temperature": self.cfg.temperature,
        #             "max_tokens": 4096,
        #             "messages": [{
        #                 "role": "user",
        #                 "content": [{
        #                     "type": "text",
        #                     "text": prompt,
        #                 }]
        #             }],
        #             "n": 1,
        #             "model": "dev-gpt-4o-vision-2024-05-13"
        #         }
        #         max_retries = 3
        #         for attempt in range(max_retries):
        #             try:
        #                 def request_fn():
        #                     response = requests.post(url, json=data)
        #                     return response.json()['choices'][0]['message']['content']
        #                 result = await asyncio.to_thread(request_fn)
        #                 if result.startswith("<think>"):
        #                     result = result[len("<think>"):]
        #                 return result
        #             except Exception as e:
        #                 logger.error(f"GPT-4o request attempt {attempt + 1} failed: {e}")
        #                 if attempt < max_retries - 1:
        #                     await asyncio.sleep(2)  # wait before retrying
        #                 else:
        #                     return ""
        #     gpt4o_responses = await asyncio.gather(*[get_gpt4o_response(prompt) for prompt in prompts])
        #     responses = gpt4o_responses
        #     stop_reasons = ["stop"] * len(prompts)
        # # --- END GPT-4o replacement --- #


        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            results = []
            for response in responses:
                if not response.startswith("<think>"):
                    response = "<think>" + response

                # Check if the format is correct # allowing zero or more newlines in between </think> and <answer> because of Qwen formatting for now, which can be fixed via SFT and is NOT a focus of training
                # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
                match = re.search(regex, response, re.DOTALL) 
                # if the format is not correct, reward is 0, or the completion doesn't end in </answer>
                if match is None or \
                    len(match.groups()) != 2 \
                    or not response.endswith("</answer>"):
                    results.append("")
                    continue
                
                # Extract the "answer" part from the completion
                answer = match.group(2).strip()

                results.append(answer)
            return results

        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        # 直接从context中提取最终结果
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answers = [answer for batch in batched_results for answer in batch]

    
        # --- OLD REWARD FUNCTION HERE --- #
        # 判断对错
        global executor
        equal_tasks = []
        for prompt, extra, final_answer in zip(prompts, extras, final_answers):
            if self.cfg.goal == "countdown":
                # COUNTDOWN REWARD #
                # add support for countdown by trying to extract the answer from the final_answer and then evaluating it
                equal_tasks.append(is_countdown_equal(final_answer, extra))

            elif self.cfg.goal == "math":
                # Original Math
                # equal_tasks.append(is_equal(solution2answer(str(extra["answer"])), solution2answer(str(final_answer)), executor)) # allows for \boxed or without \boxed # just ensuring these are strings as well because solution2answer expects strings (sometimes we parse out as ints)

                # LLM comparison for v1290
                # if exp_type == "policy":
                #     expected_answer = '"' + str(extra["answer"]) + '"'
                #     if "solution" in extra:
                #         expected_answer += f"\n\nSolution behind Expected Answer for more context: {str(extra['solution'])}"

                #     equal_tasks.append(gpt4o_compare(prompt, expected_answer, str(final_answer)))
                # else: # fast comparison
                equal_tasks.append(is_equal(solution2answer(str(extra["target"])), solution2answer(str(final_answer)), executor)) # allows for \boxed or without \boxed # just ensuring these are strings as well because solution2answer expects strings (sometimes we parse out as ints)
                # use target because that'll always have GT answer. answer might say no GT answer
        equal_results = await asyncio.gather(*equal_tasks)

        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, responses, final_answers, stop_reasons, equal_results
        ):

            # because "I don't know" is not in the prior, we will set the cases where iscorrect = false to "I don't know"
            if self.cfg.replace_half_incorrect_with_do_not_know:
                # Replace half of the incorrect answers with "do not know" to simulate uncertainty in the model.
                if not iscorrect and final_answer and np.random.rand() < 0.5:
                    response = response.split("<answer>")[0] + "<answer>I don't know</answer>" # replace the answer with "I don't know"
                    final_answer = "I don't know"  # ensure final_answer reflects this change

            results.append(
                dict(
                    response=response,
                    iscorrect=iscorrect,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                    extra=extra
                )
            )
    
        return results


    @override
    async def eval(self):
        if self.cfg.goal == "countdown":
            logger.info("Start evaluating on val set")
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.generate_max_len,
                stop=self.cfg.stop,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )

            from torch.utils.data import DataLoader

            dataset = self.eval_dataset
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
            prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

            output_for_save = []
            log_dict = defaultdict(float)
            for batch in dataloader:
                prompts = list(batch[0]) # a tuple/list of prompt strings
                extras_dict = batch[1]  # a dictionary of lists ('answer': [<answer_1>, <answer_2>, ...], 'nums': [<nums_1>, <nums_2>, ...])
                # Reconstruct a list of extras (one per sample in the batch)
                extras = [
                    {k: extras_dict[k][i] for k in extras_dict}
                    for i in range(len(prompts))
                ] # list of dictionaries where extras[i] = {'answer': <answer_i>, 'nums': <nums_i>}

                # answers = list(batch[1]["answer"])
                # file_names = list(batch[1]["file_name"])
                outputs = []
                for i, llm in enumerate(self.vllm_engines):
                    outputs.append(
                        llm.generate.remote(
                            prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                        )
                    )
                outputs = await asyncio.gather(*outputs)
                outputs = sum(outputs, [])

                # final_answers = []
                # pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
                # for output in outputs:
                #     matches = re.findall(pattern, output.outputs[0].text)
                #     if len(matches) > 0:
                #         final_answers.append(matches[-1])
                #     else:
                #         final_answers.append("")

                # Use the same regex extraction as before.
                final_answers = []
                # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
                for output in outputs:
                    text = output.outputs[0].text
                    # Prepend <think> if missing.
                    if not text.startswith("<think>"):
                        text = "<think>" + text
                    match = re.search(regex, text, re.DOTALL)
                    if match is None or len(match.groups()) != 2 or not text.endswith("</answer>"):
                        final_answers.append("")
                    else:
                        final_answers.append(match.group(2).strip())

                # Evaluate correctness for each output using countdown evaluation.
                for prompt, output, final_answer, extra in zip(prompts, outputs, final_answers, extras):
                    # Use the countdown-based correctness check.
                    iscorrect = await is_countdown_equal(final_answer, extra)
                    output_for_save.append(
                        dict(
                            prompt=prompt,
                            output=output.outputs[0].text,
                            final_answer=final_answer,
                            answer=extra.get("answer", ""),
                            iscorrect=iscorrect,
                        )
                    )
                    log_dict["total_response_len_in_char"] += len(output.outputs[0].text)
                    log_dict["correct"] += iscorrect
                    log_dict["total"] += 1

                # for prompt, output, final_answer, answer, file_name in zip(
                #     prompts, outputs, final_answers, answers, file_names
                # ):
                #     label = solution2answer(answer)
                #     prefix_response = solution2answer(final_answer)
                #     iscorrect = await is_equal(label, prefix_response, executor)
                #     output_for_save.append(
                #         dict(
                #             prompt=prompt,
                #             output=output.outputs[0].text,
                #             final_answer=final_answer,
                #             answer=answer,
                #             iscorrect=iscorrect,
                #         )
                #     )
                #     log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                #     log_dict[f"{file_name}/correct"] += iscorrect
                #     log_dict[f"{file_name}/total"] += 1

            log_dict["eval_accuracy"] = log_dict["correct"] / log_dict["total"] if log_dict["total"] > 0 else 0.0
            dump_file_name = f"eval_output_iter{self.global_step}_accuracy{log_dict['eval_accuracy']:.4f}.jsonl"
            with open(os.path.join(self.cfg.save_path, dump_file_name), "w") as f:
                for item in output_for_save:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
            logger.info(logging_str)
            for k, v in log_dict.items():
                self.writer.add_scalar(f"evals/{k}", v, self.global_step)

            # # get all file_names from self.cfg.eval_prompt_data
            # all_file_names: List[str] = [
            #     os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
            # ]
            # for file_name in all_file_names:
            #     log_dict[f"{file_name}/response_len_in_char"] = (
            #         log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            #     )
            #     log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            #     log_dict.pop(f"{file_name}/total_response_len_in_char")
            #     log_dict.pop(f"{file_name}/correct")
            #     log_dict.pop(f"{file_name}/total")
            # # calculate average accuracy
            # log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            #     all_file_names
            # )

            # dump_file_name = f"eval_output_iter{self.global_step}"
            # # join all acc from all_file_names
            # for file_name in all_file_names:
            #     dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
            # dump_file_name += ".jsonl"
            # # dump as jsonl
            # with open(
            #     os.path.join(
            #         self.cfg.save_path,
            #         dump_file_name,
            #     ),
            #     "w",
            # ) as f:
            #     for item in output_for_save:
            #         f.write(
            #             json.dumps(item, ensure_ascii=False) + "\n",
            #         )

            # logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
            # logger.info(logging_str)
            # for k, v in log_dict.items():
            #     self.writer.add_scalar(f"evals/{k}", v, self.global_step)

        elif self.cfg.goal == "math": # lazy way right now
            logger.info("Start evaluating on val set")
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.generate_max_len,
                stop=self.cfg.stop,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )

            from torch.utils.data import DataLoader

            dataset = self.eval_dataset
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
            prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

            output_for_save = []
            log_dict = defaultdict(float)
            for batch in dataloader:
                prompts = list(batch[0])
                answers = list(batch[1]["answer"])
                file_names = list(batch[1]["file_name"])
                outputs = []
                for i, llm in enumerate(self.vllm_engines):
                    outputs.append(
                        llm.generate.remote(
                            prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                        )
                    )
                outputs = await asyncio.gather(*outputs)
                outputs = sum(outputs, [])

                # final_answers = []
                # pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
                # for output in outputs:
                #     matches = re.findall(pattern, output.outputs[0].text)
                #     if len(matches) > 0:
                #         final_answers.append(matches[-1])
                #     else:
                #         final_answers.append("")
                # Use the same regex extraction as before.
                final_answers = [] # following the updated syntax from Countdown (not using boxed)
                # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                regex = r"^(.*?)<answer>([\s\S]*)<\/answer>$"
                for output in outputs:
                    text = output.outputs[0].text
                    # Prepend <think> if missing.
                    if not text.startswith("<think>"):
                        text = "<think>" + text
                    match = re.search(regex, text, re.DOTALL)
                    if match is None or len(match.groups()) != 2 or not text.endswith("</answer>"):
                        final_answers.append("")
                    else:
                        final_answers.append(match.group(2).strip())

                for prompt, output, final_answer, answer, file_name in zip(
                    prompts, outputs, final_answers, answers, file_names
                ):
                    label = solution2answer(answer)
                    prefix_response = solution2answer(final_answer)
                    iscorrect = await is_equal(label, prefix_response, executor)
                    output_for_save.append(
                        dict(
                            prompt=prompt,
                            output=output.outputs[0].text,
                            final_answer=final_answer,
                            answer=answer,
                            iscorrect=iscorrect
                        )
                    )
                    log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                    log_dict[f"{file_name}/correct"] += iscorrect
                    log_dict[f"{file_name}/total"] += 1

            # get all file_names from self.cfg.eval_prompt_data
            all_file_names: List[str] = [
                os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
            ]

            for file_name in all_file_names:
                log_dict[f"{file_name}/response_len_in_char"] = (
                    log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
                )
                log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
                log_dict.pop(f"{file_name}/total_response_len_in_char")
                log_dict.pop(f"{file_name}/correct")
                log_dict.pop(f"{file_name}/total")
            # calculate average accuracy
            log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
                all_file_names
            )

            dump_file_name = f"eval_output_iter{self.global_step}"
            # join all acc from all_file_names
            for file_name in all_file_names:
                dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
            dump_file_name += ".jsonl"
            # dump as jsonl
            with open(
                os.path.join(
                    self.cfg.save_path,
                    dump_file_name,
                ),
                "w",
            ) as f:
                for item in output_for_save:
                    f.write(
                        json.dumps(item, ensure_ascii=False) + "\n",
                    )

            logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
            logger.info(logging_str)
            for k, v in log_dict.items():
                self.writer.add_scalar(f"evals/{k}", v, self.global_step)


class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
            self_play=self.cfg.self_play,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
            remove_half_GT_answers_from_train_dataset=self.cfg.remove_half_GT_answers_from_train_dataset,  # this will remove half of the GT answers from the training dataset to let the policy evaluator generate scores instead of using iscorrect
            allow_do_not_know=self.cfg.allow_do_not_know
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            if self.cfg.goal == "countdown":
                with open(file_path, "r") as f:
                    dialogues.extend(json.load(f))
            elif self.cfg.goal == "math":
                with open(file_path, "r") as f:
                    loaded_data = json.load(f)
                    for loaded_data_item in loaded_data:
                        # only keep file name, without suffix
                        loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                    dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")

        if self.cfg.goal == "countdown":
            prompts_dataset = CustomDataset(
                dialogues,
                self.tokenizer,
                self.cfg.prompt_max_len,
                self.strategy,
                pretrain_mode=False,
                num_processors=1,
            )
        elif self.cfg.goal == "math":
            prompts_dataset = EvalCustomDataset(
                dialogues,
                self.tokenizer,
                self.cfg.prompt_max_len,
                self.strategy,
                pretrain_mode=False,
                num_processors=1,
                allow_do_not_know=self.cfg.allow_do_not_know,  # allow do not know for eval dataset as well
            )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
