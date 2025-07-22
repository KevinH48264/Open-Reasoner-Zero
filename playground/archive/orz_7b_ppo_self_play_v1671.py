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
    version = "v1671" # TODO: update this!
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B" #"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1481__self_play_False/iter450/policy"  #"Qwen/Qwen2.5-7B" # TODO: or put your downloaded model path here!
    train_policy_w_ground_truth_not_evaluator = False # flag to train policy responses against ground truth signal when we have ground truth, still train evaluator network but reserve evaluator network only when we do not have ground truth signal. Set to False to just train policy with the policy evaluator's scores
    remove_half_GT_answers_from_train_dataset = True #True # if True, will remove half of the ground truth answers from the training dataset to encourage the model to learn from the policy evaluator's scores instead of just copying the GT answers. Set to False to keep all GT answers for training.
    set_unpaired_to_draw = False # unpaired wins and loses just turn into draws (reward=0.5)


    # ARCHIVE CONFIGS
    self_play = False
    use_policy_evaluator = True #True # trains a policy evaluator with exact match to evaluate the policy assistant's responses, otherwise use exact match to evaluate policy assistant's responses
    train_policy_evaluator = True # whether to train policy evaluator network or not
    policy_evaluator_iscorrect_match_rewards = True # reward for policy evaluator is exact rule based reward matching iscorrect. otherwise, uses judge for semantic match
    allow_do_not_know = False # allow the model to output "I don't know" in the policy evaluator, this is useful for cases where the model cannot self-verify
    replace_half_incorrect_with_do_not_know = False
    balance_policy_and_evaluator_experiences = False # balance the experiences between policy and evaluator during training, this is useful to prevent overfitting to either policy or evaluator
    train_policy_fully_with_evaluator = False # flag to train policy responses fully with the policy evaluator's scores when we do not have ground truth signal
    enable_per_prompt_group_normalization = False # generally better to set to True (regardless of policy evaluator or not)
    use_gpt4o_for_rg_eval = True
    use_gpt4o_for_eval_judge = True
    policy_evaluator_reference_based_preference_rewards = False
    half_win_half_lose_per_prompt_for_policy = True
    half_win_half_lose_per_prompt_for_evaluator = True
    train_evaluator_w_majority_vote_if_no_GT = False
    curriculum_data_enabled = True
    exclude_no_GT_questions_from_train_dataset = False # not necessary when answering the question: when adding in no-GT questions, does our eval performance increase or decrease
    balance_evaluator_prompt_gt_iscorrect = True
    policy_evaluator_balance_gt_iscorrect_experiences = False # NECESSARY: best practice to not have the policy evaluator overfit to yes/no given we're training against policy performance that starts off "no" 100% -> "yes" 100%. Ideally it's always 50/50 "no" and "yes". Actually might not be necessary if we have the half win half lose per prompt for evaluator = True which we really should have.
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
        rollout_batch_size: int = 64 if not DEBUG_MODE else 32 #32 #16 #8 #8 #8 #4 #4 #16
        n_samples_per_prompt: int = 64 if not DEBUG_MODE else 8 #32 #16 #8 #8 #8 #4 #4 #8 #2
        n_policy_evaluator_samples_per_policy_response: int = 3 #9 #4 #4 # sample policy evaluator responses to evaluate the correctness of the policy assistant's response
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
                    # "data/orz_math_57k_collected_w_num_correct_easiest_half.json" # with num_correct = [0, 1] excluded, easiest half (~58%) of questions
                    # "/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only__easiest_half.json" # 55K
                    "/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only__no_na__no_zero_pass_random_3K_easiest_half_n_1500.json"
                ])
            else:
                prompt_data: ListConfig = ListConfig([
                    # "data/orz_math_57k_collected.json",
                    # "data/orz_math_57k_collected_w_num_correct_n_2048_rollouts_4.json" # debugging
                    # "data/orz_math_57k_collected_w_num_correct.json" # with num_correct = [0, 1] being the hardest 50% of questions
                    # "/vc_data_blob/users/kevihuang/data/orz/deepmath_103k.json"
                    # "/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only.json"
                    "/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only__no_na__no_zero_pass_random_3K.json"
                    # "/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only__no_na__no_zero_pass.json" # 106K, all passable by Qwen-72B
                ])

        eval_prompt_data: ListConfig = ListConfig(
            [
                "data/eval_data/math500.json",
                "data/eval_data/aime2024.json",
                "data/eval_data/gpqa_diamond.json",
                "data/eval_data/prm800k_100_correct_100_incorrect_rm_eval.json",
                "data/eval_data/aime2024_30_correct_30_incorrect_rm_eval.json"
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

def build_judge_prompt(question: str, response: str, correct: str) -> str:
    return f'''Judge whether the following [extracted_final_answer] is correct or not based on the precise and unambiguous [correct_answer] below.

[extracted_final_answer]: {response}

[correct_answer]: {correct}

Your judgement must be in the format and criteria specified below:
<think>
reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the[correct_answer] given above, or is within a small margin of error for
numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
</think>
<answer> yes or no </answer>

Response: <think>\n'''

def parse_judge_reply(txt: str) -> bool | None:
    m = re.search(r"<answer>(.*?)</answer>", txt, re.DOTALL)
    if not m: return None
    ans = m.group(1).strip().lower()
    return True if ans == "yes" else False if ans == "no" else None

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
                score = -0.5 #0.0 #-0.5
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

        # ——— log just for prompts with NO GT ANSWER ———
        no_gt_scores = [
            s for s, e in zip(scores, extras)
            if e.get("answer", "") == "[NO GT ANSWER]"
        ]
        avg_no_gt_score = float(np.mean(no_gt_scores)) if no_gt_scores else 0.0
        self.writer.add_scalar(
            "policy/avg_score_before_evaluator_no_gt",
            avg_no_gt_score,
            self.global_step,
        )

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
                regex = r"^(.*)<answer>([\s\S]*)<\/answer>$"
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
            rf_eval_prompt_to_policy_response_map = {}
            for i, (prompt, output, extra) in enumerate(zip(prompts, outputs, extras)):
                for sample_idx in range(n_samples):
                    rf_prompt = rf'''You are a math teacher. Grade the Solution, verifying correctness step by step.

Question: "{prompt.split('This is the problem:', 1)[-1].rsplit('Assistant:', 1)[0].strip()}"

Solution: "<think>{output['response']}"

Please output your Solution verification in the following format:
<think> Your Solution verification here </think>
<answer> Yes or No </answer>

Math Teacher Response: <think>\n'''

                    rf_eval_prompts.append(rf_prompt)
                    rf_policy_indices.append(i)
                    rf_eval_prompt_to_policy_response_map[rf_prompt] = output['response']


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
                    regex = r"^(.*)<answer>([\s\S]*)<\/answer>$"
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
            verification_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

            total_judge_decisions = 0
            iscorrect_match_judge_decisions = 0

            total_iscorrect_no_count = 0
            total_iscorrect_no_judge_no_count = 0
            rf_verification_list = []
            evaluator_scores = []
            evaluator_res_policy_idxs = []
            evaluator_prompt_policy_idxs = {}
            no_gt_rf_scores: List[float] = []
            for j in range(len(rf_eval_prompts)):
                policy_idx = rf_policy_indices[j]
                evaluator_prompt_policy_idxs[rf_eval_prompts[j]] = policy_idx # map prompt to policy index for later use
                gt_iscorrect = outputs[policy_idx]["iscorrect"]
                policy_final_answer = outputs[policy_idx].get("final_answer", "") # fallback to empty string if not found, this should be the expected answer

                rf_answer = rf_final_answers[j].strip()
                rg_answer = rg_final_answers[policy_idx].strip()

                rf_match = verification_regex.match(rf_answer)
                rg_match = verification_regex.match(rg_answer)

                # Only include if proof judge decision is valid and p_correct exists (i.e. not "N/A"). and policy_final_answer != "" (not formatted properly)
                if policy_final_answer == "":
                    continue # don't evaluate a policy response that is not formatted

                if p_correct_dict[policy_idx] == "N/A" or rg_match is None:
                    continue  # ensure policy we're evaluating was formatted correctly, and that a reference based evaluation exists properly # IMPORTANT: sometimes the ground truth match is not good, so ensure that at least one policy was correct in this prompt group before judging evaluations

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
                    score_val = -0.5 #0.0 #-0.5
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

                # record for no-GT-answer cases
                if extras[policy_idx].get("answer", "") == "[NO GT ANSWER]":
                    no_gt_rf_scores.append(score_val)

                # Use the extracted final answer from the RF evaluation as the prompt.
                if self.cfg.train_policy_evaluator:
                    
                    if extras[policy_idx]["answer"] != "[NO GT ANSWER]": # sometimes we don't want to train the policy evaluator. And we want to train only when we have GT answer in policy
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
                        evaluator_scores.append(score_val)
                        evaluator_res_policy_idxs.append(policy_idx)

                    elif self.cfg.train_evaluator_w_majority_vote_if_no_GT:
                        # 1) collect all RF answers for this prompt
                        votes = [
                            rf_final_answers[k].strip().lower()
                            for k, p_idx in enumerate(rf_policy_indices)
                            if p_idx == policy_idx
                        ]
                        yes_votes = votes.count("yes")
                        no_votes  = votes.count("no")

                        # 2) figure out the majority (or None if tied/ambiguous)
                        if yes_votes > no_votes:
                            majority = "yes"
                        elif no_votes > yes_votes:
                            majority = "no"
                        else:
                            majority = "tie"

                        # 3) assign a 1.0 reward if this sample agrees with the majority, else 0.0
                        if score_val >= 0.0: # only do majority vote if formatted correctly
                            if rf_verification == majority:
                                score_val = 1.0
                            elif majority == "tie":
                                score_val = 0.5
                            else:
                                score_val = 0.0

                        # 4) now append exactly as in the GT branch
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
                        evaluator_scores.append(score_val)
                        evaluator_res_policy_idxs.append(policy_idx)
            logger.debug(f"After raw additions, len(evaluator_res_score_tensors): {len(evaluator_res_score_tensors)}")

            # Log average custom reward (evaluator_scores) in TensorBoard.
            avg_evaluator_score = np.mean(evaluator_scores) if evaluator_scores else 0.0
            self.writer.add_scalar("policy_evaluator/avg_score_before_balancing", avg_evaluator_score, self.global_step)

            # ——— log evaluator score just for policy prompts with NO GT ANSWER ———
            avg_no_gt_eval_score = float(np.mean(no_gt_rf_scores)) if no_gt_rf_scores else 0.0
            self.writer.add_scalar(
                "policy_evaluator/avg_score_before_balancing_no_gt",
                avg_no_gt_eval_score,
                self.global_step,
            )

            if self.cfg.half_win_half_lose_per_prompt_for_evaluator:
                # 1) group by prompt
                eval_groups = defaultdict(list)
                for prompt, response, score_tensor, rv in zip(
                    evaluator_res_prompts,
                    evaluator_res_responses,
                    evaluator_res_score_tensors,
                    rf_verification_list
                ):
                    eval_groups[prompt].append((response, score_tensor, rv))

                # 2) rebuild with wins, loses, + all invalids
                new_prompts = []
                new_responses = []
                new_score_tensors = []
                new_rv_flags = []

                for prompt, group in eval_groups.items():
                    # build lists of indices rather than full tuples
                    wins_idx = [
                        i for i, (_, tensor, _) in enumerate(group)
                        if abs(tensor[-1].item() - 1.0) < 1e-6
                    ]
                    loses_idx = [
                        i for i, (_, tensor, _) in enumerate(group)
                        if abs(tensor[-1].item() - 0.0) < 1e-6
                    ]
                    invalids_idx = [
                        i for i, (_, tensor, _) in enumerate(group)
                        if abs(tensor[-1].item() + 0.5) < 1e-6
                    ]

                    # only balance if we have both wins and loses
                    n = min(len(wins_idx), len(loses_idx))
                    sel_wins_idx  = random.sample(wins_idx, n)
                    sel_loses_idx = random.sample(loses_idx, n)

                    # append the balanced subset
                    for idx in sel_wins_idx + sel_loses_idx:
                        resp, tensor, flag = group[idx]
                        new_prompts.append(prompt)
                        new_responses.append(resp)
                        new_score_tensors.append(tensor)
                        new_rv_flags.append(flag)

                    # turn any *unselected* wins/loses into draws if requested
                    if self.cfg.set_unpaired_to_draw:
                        unselected_wins_idx  = [i for i in wins_idx  if i not in sel_wins_idx]
                        unselected_loses_idx = [i for i in loses_idx if i not in sel_loses_idx]
                        for idx in unselected_wins_idx + unselected_loses_idx:
                            resp, orig_tensor, flag = group[idx]
                            draw_tensor = orig_tensor.clone()
                            draw_tensor[-1] = 0.5
                            new_prompts.append(prompt)
                            new_responses.append(resp)
                            new_score_tensors.append(draw_tensor)
                            new_rv_flags.append(flag)

                    # now append *all* the invalids
                    for idx in invalids_idx:
                        resp, tensor, flag = group[idx]
                        new_prompts.append(prompt)
                        new_responses.append(resp)
                        new_score_tensors.append(tensor)
                        new_rv_flags.append(flag)

                # 3) overwrite your evaluator buffers
                evaluator_res_prompts       = new_prompts
                evaluator_res_responses     = new_responses
                evaluator_res_score_tensors = new_score_tensors
                rf_verification_list        = new_rv_flags


            # if self.cfg.policy_evaluator_balance_gt_iscorrect_experiences, go through all evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors tuples where evaluator_res_score_tensors has the final score_val = 1.0, and find the minimum # of rf_verification = "yes" or rf_verification = "no" and then filter out the extras of the other group to ensure that when score_val = 1.0, we have an equal number of responses where rf_verification = "yes" and rf_verificaiton = "no". Same thing for final score_val = 0.0
            # --- Balance evaluator experiences if required ---
            if self.cfg.policy_evaluator_balance_gt_iscorrect_experiences: # I think that balancing is best practice anyway to not train a model that just always sees "Yes" or "No", we should do this always
                assert(len(evaluator_res_prompts) == len(evaluator_res_responses) == len(evaluator_res_score_tensors) == len(rf_verification_list))
                
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
                min_count_overall = min(len(group_1_yes), len(group_1_no), len(group_0_yes), len(group_0_no))
                if group_1_yes and group_1_no:
                    min_count_1 = min_count_overall #min(len(group_1_yes), len(group_1_no))
                    selected_1_yes = random.sample(group_1_yes, min_count_1)
                    selected_1_no  = random.sample(group_1_no,  min_count_1)
                else:
                    selected_1_yes = []
                    selected_1_no  = []

                # For examples with score 0.0, balance similarly.
                if group_0_yes and group_0_no:
                    min_count_0 = min_count_overall #min(len(group_0_yes), len(group_0_no))
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

            # Update policy scores based on RF evaluations if extra["answer"] == "[NO GT ANSWER]"
            if not self.cfg.train_policy_w_ground_truth_not_evaluator:
                # Get the number of samples per policy response from configuration.
                n_samples = self.cfg.n_policy_evaluator_samples_per_policy_response
                for i, (prompt, output, extra) in enumerate(zip(prompts, outputs, extras)):
                    if extra["answer"] == "[NO GT ANSWER]" or self.cfg.train_policy_fully_with_evaluator:
                        # Only update if the score indicates a valid formatting.
                        if scores[i] >= 0:
                            # Get all RF evaluator answers for this policy output.
                            # Assumes rf_final_answers is ordered so that samples for output i 
                            # are stored consecutively from i*n_samples to (i+1)*n_samples - 1.
                            samples = rf_final_answers[i * n_samples : (i + 1) * n_samples]
                            
                            # Count the votes for "yes" and "no" (after stripping and lowering the response).
                            yes_count = sum(1 for ans in samples if ans.strip().lower() == "yes")
                            no_count = sum(1 for ans in samples if ans.strip().lower() == "no")
                            
                            # Use the majority vote:
                            if yes_count > no_count:
                                scores[i] = 1.0
                            elif no_count > yes_count:
                                scores[i] = 0.0
                            else:
                                # If the votes are tied or ambiguous, use 0.5.
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



            # --- Last debugging metrics: extract answer tags from evaluator responses ---
            answer_counts = {"yes": 0, "no": 0, "N/A": 0}
            total_answers = 0

            for resp in evaluator_res_responses:
                ans, ok = extract_final(resp)
                if ok:
                    low = ans.lower()
                    if low == "yes":
                        answer_counts["yes"] += 1
                    elif low == "no":
                        answer_counts["no"] += 1
                    else:
                        answer_counts["N/A"] += 1
                else:
                    answer_counts["N/A"] += 1
                total_answers += 1

            if total_answers > 0:
                yes_pct = answer_counts["yes"] / total_answers * 100
                no_pct  = answer_counts["no"]  / total_answers * 100
                na_pct  = answer_counts["N/A"] / total_answers * 100

                # overall distribution
                self.writer.add_scalar("policy_evaluator/ans_yes_pct", yes_pct, self.global_step)
                self.writer.add_scalar("policy_evaluator/ans_no_pct",  no_pct, self.global_step)
                self.writer.add_scalar("policy_evaluator/ans_na_pct",  na_pct, self.global_step)

                # percent yes out of just yes+no
                yes_no_total = answer_counts["yes"] + answer_counts["no"]
                yes_of_yesno = (answer_counts["yes"] / yes_no_total * 100) if yes_no_total else 0.0
                self.writer.add_scalar("policy_evaluator/ans_yes_of_yesno_pct", yes_of_yesno, self.global_step) # verify that the yes/no ratio is not skewed too much towards "yes" or "no"

                logger.info(
                    f"[STEP {self.global_step}] Evaluator answers: "
                    f"yes={answer_counts['yes']} ({yes_pct:.1f}%), "
                    f"no={answer_counts['no']} ({no_pct:.1f}%), "
                    f"N/A={answer_counts['N/A']} ({na_pct:.1f}%), "
                    f"yes/(yes+no)={yes_of_yesno:.1f}%"
                )

            # --- Now: same answer‐tag metrics, but only for score == 1.0 ---
            answer_counts_1 = {"yes": 0, "no": 0, "N/A": 0}
            total_answers_1 = 0

            for resp, tensor in zip(evaluator_res_responses, evaluator_res_score_tensors):
                if tensor[-1].item() == 1.0:
                    ans, ok = extract_final(resp)
                    if ok:
                        low = ans.lower()
                        if low == "yes":
                            answer_counts_1["yes"] += 1
                        elif low == "no":
                            answer_counts_1["no"] += 1
                        else:
                            answer_counts_1["N/A"] += 1
                    else:
                        answer_counts_1["N/A"] += 1
                    total_answers_1 += 1

            if total_answers_1 > 0:
                yes_pct_1 = answer_counts_1["yes"] / total_answers_1 * 100
                no_pct_1  = answer_counts_1["no"]  / total_answers_1 * 100
                na_pct_1  = answer_counts_1["N/A"] / total_answers_1 * 100

                # log overall distribution for score==1.0
                self.writer.add_scalar("policy_evaluator/ans_yes_pct_score1", yes_pct_1, self.global_step)
                self.writer.add_scalar("policy_evaluator/ans_no_pct_score1",  no_pct_1, self.global_step)
                self.writer.add_scalar("policy_evaluator/ans_na_pct_score1",  na_pct_1, self.global_step)

                # percent yes out of yes+no for score==1.0
                yes_no_total_1 = answer_counts_1["yes"] + answer_counts_1["no"]
                yes_of_yesno_1 = (answer_counts_1["yes"] / yes_no_total_1 * 100) if yes_no_total_1 else 0.0
                self.writer.add_scalar("policy_evaluator/ans_yes_of_yesno_pct_score1", yes_of_yesno_1, self.global_step) # sanity check that the yes/no SFT ratio is near 50% # debug mode here

                logger.info(
                    f"[STEP {self.global_step}] Score==1.0 evaluator answers: "
                    f"yes={answer_counts_1['yes']} ({yes_pct_1:.1f}%), "
                    f"no={answer_counts_1['no']} ({no_pct_1:.1f}%), "
                    f"N/A={answer_counts_1['N/A']} ({na_pct_1:.1f}%), "
                    f"yes/(yes+no)={yes_of_yesno_1:.1f}%"
                )
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


        # go through all prompt groups, and if for a prompt, the average score > 0.5, then set self.train_dataset.dialogues and find the prompt, and set the "policy_majority_win" in the extra = True for that (prompt, extra) group.
        # 1) group the scores by prompt for this batch
        batch_group_scores = defaultdict(list)
        for prompt, score in zip(prompts, scores):
            batch_group_scores[prompt].append(score)

        # 2) compute per-prompt averages
        per_prompt_avgs = [sum(sc_list) / len(sc_list) for sc_list in batch_group_scores.values()]

        # 3) collect the prompts whose avg score > 0.5
        winners = {
            prompt
            for prompt, sc_list in batch_group_scores.items()
            if (sum(sc_list) / len(sc_list)) > 0.5
        }

        # 4) log prompt win percentage
        total_groups = len(per_prompt_avgs)
        prompt_win_pct = (len(winners) / total_groups * 100) if total_groups else 0.0
        self.writer.add_scalar("policy/query_win_percentage", prompt_win_pct, self.global_step)

        # 5) log overall average score per prompt
        prompt_avg_score = sum(per_prompt_avgs) / total_groups if total_groups else 0.0
        self.writer.add_scalar("policy/query_avg_policy_score", prompt_avg_score, self.global_step)

        # 6) one pass through the dataset to flip the flag
        if winners:
            for ds_prompt, extra in self.train_dataset.dialogues:
                if ds_prompt in winners:
                    extra["policy_majority_win"] = True


        # If the half_win_half_lose flag is enabled, group by prompt and balance wins vs. loses.
        if self.cfg.half_win_half_lose_per_prompt_for_policy:
            groups = defaultdict(list)
            for prompt, response, score_tensor, extra in zip(prompts, responses, score_tensors, extras):
                if (self.cfg.use_policy_evaluator and not self.cfg.train_policy_w_ground_truth_not_evaluator) or extra["answer"] != "[NO GT ANSWER]":
                    groups[prompt].append((response, score_tensor, extra))

            res_prompts = []
            res_responses = []
            res_score_tensors = []

            for prompt, group in groups.items():
                # build lists of indices rather than full tuples
                wins_idx = [
                    i for i, (_, tensor, _) in enumerate(group)
                    if abs(tensor[-1].item() - 1.0) < 1e-6
                ]
                loses_idx = [
                    i for i, (_, tensor, _) in enumerate(group)
                    if abs(tensor[-1].item() - 0.0) < 1e-6
                ]
                invalids_idx = [
                    i for i, (_, tensor, _) in enumerate(group)
                    if abs(tensor[-1].item() + 0.5) < 1e-6
                ]

                # Only balance if we have both wins and loses
                num_to_select = min(len(wins_idx), len(loses_idx))
                selected_wins_idx  = random.sample(wins_idx, num_to_select)
                selected_loses_idx = random.sample(loses_idx, num_to_select)

                # add wins & loses
                for idx in selected_wins_idx + selected_loses_idx:
                    resp, tensor, _ = group[idx]
                    res_prompts.append(prompt)
                    res_responses.append(resp)
                    res_score_tensors.append(tensor)

                # turn any *unselected* wins/loses into draws (0.5) if requested
                if self.cfg.set_unpaired_to_draw:
                    unselected_wins_idx  = [i for i in wins_idx  if i not in selected_wins_idx]
                    unselected_loses_idx = [i for i in loses_idx if i not in selected_loses_idx]
                    for idx in unselected_wins_idx + unselected_loses_idx:
                        resp, orig_tensor, _ = group[idx]
                        draw_score = orig_tensor.clone()
                        draw_score[-1] = 0.5
                        res_prompts.append(prompt)
                        res_responses.append(resp)
                        res_score_tensors.append(draw_score)

                # now append *all* the invalid (-0.5) examples
                for idx in invalids_idx:
                    resp, tensor, _ = group[idx]
                    res_prompts.append(prompt)
                    res_responses.append(resp)
                    res_score_tensors.append(tensor)
        else:
            # Default behavior: # compute policy prompts, responses, and scores
            res_prompts = []
            res_responses = []
            res_score_tensors = []
            for prompt, response, score_tensor, extra in zip(prompts, responses, score_tensors, extras):
                if (self.cfg.use_policy_evaluator and not self.cfg.train_policy_w_ground_truth_not_evaluator) or extra["answer"] != "[NO GT ANSWER]":
                    res_prompts.append(prompt)
                    res_responses.append(response)
                    res_score_tensors.append(score_tensor)


        # balance evaluator policy responses whose gt iscorrect is True and False, so that we have equal number of GT "yes" and "no" responses for the policy evaluator training
        if self.cfg.balance_evaluator_prompt_gt_iscorrect: # Note: this assume that you are NOT training the evaluator with majority vote
            # update based on rf_eval_prompt_to_policy_response_map
            selected_prompts = set()
            for prompt in evaluator_res_prompts:
                if rf_eval_prompt_to_policy_response_map[prompt] in res_responses: # if the prompt is based on a selected policy response, add it
                    selected_prompts.add(prompt)

            # 6) rebuild evaluator buffers to include ONLY experiences from those prompts
            filtered = [
                (pr, resp, tensor, p_idx, vf)
                for pr, resp, tensor, p_idx, vf
                in zip(
                    evaluator_res_prompts,
                    evaluator_res_responses,
                    evaluator_res_score_tensors,
                    evaluator_res_policy_idxs,
                    rf_verification_list,
                )
                if pr in selected_prompts
            ]
            if filtered:
                evaluator_res_prompts, evaluator_res_responses, evaluator_res_score_tensors, evaluator_res_policy_idxs, rf_verification_list = map(list, zip(*filtered))
            else:
                evaluator_res_prompts = []
                evaluator_res_responses = []
                evaluator_res_score_tensors = []
                evaluator_res_policy_idxs = []
                rf_verification_list = []

            # TODO: add a metric where we log the # iscorrect evaluator responses whose outputs[evaluator_prompt_policy_idxs[prompt]]["iscorrect"] =True where score = 1.0 and outputs[evaluator_prompt_policy_idxs[prompt]]["iscorrect"] = False where score = 0.0

            # count how many “GT=True” cases got score==1.0
            num_pred_true_score1 = 0
            num_pred_false_score0 = 0
            for i, prompt in enumerate(evaluator_res_prompts):
                pred_is_correct = outputs[evaluator_prompt_policy_idxs[prompt]]["iscorrect"]
                score = evaluator_res_score_tensors[i][-1].item()
                if pred_is_correct and score == 1.0:
                    num_pred_true_score1 += 1
                if not pred_is_correct and score == 0.0:
                    num_pred_false_score0 += 1

            # push both counts to TensorBoard
            self.writer.add_scalar(
                "policy_evaluator/num_pred_true_score1",
                num_pred_true_score1,
                self.global_step
            ) # this tells us if we have more or less examples that are predicting "yes" or "no"
            self.writer.add_scalar(
                "policy_evaluator/num_pred_false_score0",
                num_pred_false_score0,
                self.global_step
            )


        # downstream: log avg_after as before
        final_vals = [t[-1].item() for t in evaluator_res_score_tensors]
        avg_after = float(np.mean(final_vals)) if final_vals else 0.0
        self.writer.add_scalar(
            "policy_evaluator/avg_score_after_balance_evaluator_prompt_gt_iscorrect",
            avg_after,
            self.global_step
        )

        logger.debug(f"After balancing, len(evaluator_res_score_tensors): {len(evaluator_res_score_tensors)}")



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

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            results = []
            for response in responses:
                if not response.startswith("<think>"):
                    response = "<think>" + response

                # Check if the format is correct # allowing zero or more newlines in between </think> and <answer> because of Qwen formatting for now, which can be fixed via SFT and is NOT a focus of training
                # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*)<\/answer>$"
                regex = r"^(.*)<answer>([\s\S]*)<\/answer>$"
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

    
        # NEW: semantic judge, falling back to is_equal if judge fails. again, use target as the ground truth answer
        # build one prompt per sample
        semantic_prompts = [
            build_judge_prompt(
                question=prompt.split("This is the problem:",1)[-1]
                                .rsplit("Assistant:",1)[0]
                                .strip(),
                response=fa,
                correct=extra["target"],
            )
            for prompt, fa, extra in zip(prompts, final_answers, extras)
        ]

        # run them all with deterministic sampling
        semantic_params = SamplingParams(
            temperature=0.0,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        semantic_responses, _ = await gen_func(
            prompts=semantic_prompts,
            sampling_params=semantic_params,
            use_tqdm=False,
            truncate_prompt=True,
        )
        semantic_ok = [parse_judge_reply(txt) for txt in semantic_responses]

        # now compute "equal_results", using semantic when available, else is_equal()
        async def resolve_correct(idx, extra, final):
            if semantic_ok[idx] is None:
                # fallback to exact‐match
                return await is_equal(
                    solution2answer(str(extra["target"])),
                    solution2answer(final),
                    executor,
                )
            return semantic_ok[idx]

        equal_results = await asyncio.gather(*[
            resolve_correct(i, e, fa)
            for i, (e, fa) in enumerate(zip(extras, final_answers))
        ])

        # --- Log average True rate in equal_results ---
        num_true = sum(1 for x in equal_results if x)
        avg_equal = num_true / len(equal_results) if equal_results else 0.0
        # TensorBoard
        self.writer.add_scalar("policy/avg_equal_results", avg_equal, self.global_step)
        # Optional console log
        # logger.info(f"[STEP {self.global_step}] avg_equal_results: {avg_equal:.3f} ({num_true}/{len(equal_results)})")

        # Log % of examples where the semantic judge returned a non‐None verdict
        num_sem_not_none = sum(1 for ok in semantic_ok if ok is not None)
        total = len(semantic_ok)
        pct_not_none = num_sem_not_none / total * 100 if total else 0.0
        self.writer.add_scalar(
            "policy/semantic_judge_not_none",
            pct_not_none,
            self.global_step
        )

        # --- OLD REWARD FUNCTION HERE --- #
        # global executor
        # equal_tasks = []
        # for prompt, extra, final_answer in zip(prompts, extras, final_answers):
        #     equal_tasks.append(is_equal(solution2answer(str(extra["target"])), solution2answer(str(final_answer)), executor)) # allows for \boxed or without \boxed # just ensuring these are strings as well because solution2answer expects strings (sometimes we parse out as ints)
        #     # use target because that'll always have GT answer. answer might say no GT answer
        # equal_results = await asyncio.gather(*equal_tasks)

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
            logger.info(f"Evaluating batch with {len(batch[0])} samples")
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

            # parse out the final answers from the outputs
            logger.info("Parsing final answers from outputs")
            final_answers = []
            regex = r"^(.*)<answer>([\s\S]*)<\/answer>$"
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

            # SEMANTIC GRADER
            # –– NEW: build and run semantic‐judge prompts on each sample ––
            logger.info("Building semantic judge prompts")
            semantic_prompts = []
            for prompt, final, gold in zip(prompts, final_answers, answers):
                # adapt your example’s build_judge_prompt signature
                semantic_prompts.append(
                    build_judge_prompt(
                    question=prompt.split("This is the problem:",1)[-1]
                                    .rsplit("Assistant:",1)[0]
                                    .strip(),
                    response=final,
                    correct=gold,
                    )
                )

            # fire them through your vLLM engines (deterministic sampling)
            logger.info("Running semantic judge prompts through vLLM engines")
            semantic_params = SamplingParams(
                temperature=0.0,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                max_tokens=self.cfg.generate_max_len,
                stop=self.cfg.stop,
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            )
            per_engine = (len(semantic_prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
            tasks = []
            for idx, engine in enumerate(self.vllm_engines):
                start = idx * per_engine
                end   = min(start + per_engine, len(semantic_prompts))
                if start < end:
                    tasks.append(
                        engine.generate.remote(
                            prompts=semantic_prompts[start:end],
                            sampling_params=semantic_params
                        )
                    )
            raw_lists = await asyncio.gather(*tasks)
            judge_replies = [r.outputs[0].text for sub in raw_lists for r in sub]
            semantic_ok = [ parse_judge_reply(txt) for txt in judge_replies ]

            # EXACT GRADER
            logger.info("Running exact grader")
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, executor)
                idx = len(output_for_save)
                output_for_save.append({
                    "prompt":         prompt,
                    "output":         output.outputs[0].text,
                    "final_answer":   final_answer,
                    "answer":         answer,
                    "iscorrect":      iscorrect,
                    "semantic_correct": semantic_ok[idx],
                    "file_name":      file_name,
                })
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

            # semantic accuracy
            sem_correct = sum(1 for r in output_for_save
                            if r["semantic_correct"] and r["file_name"] == file_name)
            total      = sum(1 for r in output_for_save
                            if r["file_name"] == file_name and r["semantic_correct"] is not None)
            log_dict[f"{file_name}/semantic_accuracy"] = sem_correct / total if total else 0.0


        # split into policy vs. reward eval sets
        policy_fnames = [f for f in all_file_names if "rm_eval" not in f]
        reward_fnames = [f for f in all_file_names if "rm_eval" in f]

        # compute group‐level averages
        def mean(keys):
            return sum(log_dict[k] for k in keys) / len(keys) if keys else 0.0

        # accuracy
        log_dict["eval_accuracy_policy"] = mean([f"{fn}/accuracy" for fn in policy_fnames])
        log_dict["eval_accuracy_reward"] = mean([f"{fn}/accuracy" for fn in reward_fnames])

        # response length
        log_dict["eval_response_len_policy"] = mean([f"{fn}/response_len_in_char" for fn in policy_fnames])
        log_dict["eval_response_len_reward"] = mean([f"{fn}/response_len_in_char" for fn in reward_fnames])

        # calculate average accuracy
        log_dict["eval_accuracy"] = mean([f"{fn}/accuracy" for fn in all_file_names])

        # semantic accuracy
        policy_sem_keys = [f"{fn}/semantic_accuracy" for fn in policy_fnames]
        reward_sem_keys = [f"{fn}/semantic_accuracy" for fn in reward_fnames]
        all_sem_keys    = [f"{fn}/semantic_accuracy" for fn in all_file_names]

        log_dict["eval_accuracy_policy_semantic"] = mean(policy_sem_keys)
        log_dict["eval_accuracy_reward_semantic"] = mean(reward_sem_keys)
        log_dict["eval_accuracy_semantic"]        = mean(all_sem_keys)

        # push to tensorboard
        for k in ("eval_accuracy_policy_semantic",
                "eval_accuracy_reward_semantic",
                "eval_accuracy_semantic"):
            self.writer.add_scalar(f"evals/{k}", log_dict[k], self.global_step)

        total_sem_not_none = sum(
            1 for r in output_for_save if r["semantic_correct"] is not None
        )
        total_samples = len(output_for_save)
        log_dict["eval_semantic_eval_not_none"] = (
            total_sem_not_none / total_samples if total_samples > 0 else 0.0
        )
        self.writer.add_scalar(
            "evals/eval_semantic_eval_not_none",
            log_dict["eval_semantic_eval_not_none"],
            self.global_step
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