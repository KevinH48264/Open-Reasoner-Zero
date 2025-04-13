import asyncio
import json
import math
import os
import random
from functools import partial
from heapq import heapify, heappop, heappush
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union
import re, ast
from itertools import permutations, product
import numpy as np

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from orz.ppo.actors import PPORayActorGroup
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
    normalize_advantages,
)


class RayPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        strategy: DeepspeedStrategy,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        vllm_engines=None,
        colocate_pg: Optional[PlacementGroup] = None,
        self_play: bool = False,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.vllm_engines = vllm_engines
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg

        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)
        self.replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,
            cpu_offload=True,
            packing_samples=True,
        )
        self.self_play = self_play
        self.history = []

    def __del__(self):
        self.writer.close()

    async def eval(self):
        raise NotImplementedError("Eval function should be implemented in user's exp")

    async def train(self):
        # 1. create rank0 policy model and vllm_engines groups, then boardcast weights to vllm engins
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()
            await self._backload_vllm_engines()

        await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)
        logger.info("Create vllm engine groups done.")

        async with Timer("Sync actor weights to vllm engines"):
            await self._sync_policy_weights_to_vllm()

        if self.cfg.colocate_all:
            async with Timer("Offload policy model to cpu"):
                await self.policy_model.offload_to_cpu()

        # DEBUG: data sample
        logger.debug(f"train_dataset size: {len(self.train_dataset)}")
        logger.debug(f"train_dataset sample: {self.train_dataset[0]}")

        # 2. main training loop
        consumed_samples = 0
        num_rollouts_per_episodes = (
            self.num_update_steps_per_episodes
            * self.cfg.train_batch_size
            // self.cfg.max_epochs
            // self.cfg.rollout_batch_size
            // self.cfg.n_samples_per_prompt
        )

        self.global_step = consumed_samples // self.cfg.rollout_batch_size
        start_episode = consumed_samples // self.cfg.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * self.cfg.rollout_batch_size)
        for episode in range(start_episode, self.cfg.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()), desc=f"Episode [{episode + 1}/{self.cfg.num_episodes}]"
            )
            for iter, rand_prompts in enumerate(self.prompts_dataloader):

                # 1. eval if enable eval
                if self.cfg.enable_eval and (
                    self.global_step % self.cfg.eval_interval == 0 or iter == len(self.prompts_dataloader) - 1
                ):
                    await self.eval()

                # 3. make experiences, calculate advantages and returns
                # Input: prompts, 
                # Output: update the replay_buffer which is used for training policy + critic
                await self.make_experience(rand_prompts)

                # check if has enough data
                if len(self.replay_buffer) <= 0:
                    if self.cfg.colocate_all:
                        # skip, but transfer weight
                        await self.policy_model.backload_to_gpu()
                        await self._backload_vllm_engines()
                        await self._sync_policy_weights_to_vllm()
                        await self.policy_model.offload_to_cpu()
                    continue

                if self.cfg.advantage_normalize:
                    self.replay_buffer = normalize_advantages(self.replay_buffer)

                # serialize replay buffer to jsonl
                async with Timer("Dumping replay buffer"):
                    all_replay_buffer_save_path = os.path.join(self.cfg.save_path, "dumped_replay_buffer")
                    os.makedirs(all_replay_buffer_save_path, exist_ok=True)
                    dump_path = os.path.join(all_replay_buffer_save_path, f"iter{self.global_step}_replay_buffer.jsonl")
                    with open(dump_path, "a") as f:
                        logger.info(f"dumping replay buffer to {dump_path}")
                        for item in self.replay_buffer:
                            f.write(json.dumps(item.to_json()) + "\n")

                num_policy_dp_nodes = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
                num_critic_dp_nodes = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
                policy_buffers = self.replay_buffer.split_to_n_batches(num_policy_dp_nodes)
                if num_policy_dp_nodes != num_critic_dp_nodes:
                    critic_buffers = self.replay_buffer.split_to_n_batches(num_critic_dp_nodes)
                else:
                    critic_buffers = policy_buffers

                # 4. train policy/critic model
                # loss is computed from the buffers = self.replay_buffer (which should include policy AND query rewards)
                if self.cfg.colocate_all: # default: true
                    if self.critic_model is not None:
                        async with Timer("Critic model training"):
                            await self.critic_model.backload_to_gpu()
                            await self.ppo_local_train_critic(critic_buffers, self.global_step)
                            await self.critic_model.offload_to_cpu()
                    async with Timer("Actor model training"):
                        await self.policy_model.backload_to_gpu()
                        status = await self.ppo_local_train_policy(policy_buffers, self.global_step)
                        await self.policy_model.offload_to_cpu()

                else:
                    if self.critic_model is not None:
                        async with Timer("Actor and Critic model training"):
                            status = await asyncio.gather(
                                self.ppo_local_train_policy(policy_buffers, self.global_step),
                                self.ppo_local_train_critic(critic_buffers, self.global_step),
                            )
                            await asyncio.gather(
                                self.policy_model.async_run_method("empty_cache"),
                                self.critic_model.async_run_method("empty_cache"),
                            )
                            status = status[0]
                    else:
                        async with Timer("Actor model training"):
                            status = await self.ppo_local_train_policy(policy_buffers, self.global_step)
                            await self.policy_model.async_run_method("empty_cache")

                self.replay_buffer.clear()

                # 5. set logs
                logger.info(status)
                pbar.update()
                # log epoch info
                self.writer.add_scalar("episode_idx", episode, self.global_step)
                self.global_step += 1
                if self.global_step % self.cfg.save_interval == 0:
                    await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                    if self.critic_model is not None:
                        await self.critic_model.async_save_model(self.tokenizer, self.global_step)
                    logger.info("Successfully save model weights, training continue.")

            if self.cfg.update_ref_every_epoch:
                await self.policy_model.backload_to_gpu()
                await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                await self.policy_model.offload_to_cpu()
                await asyncio.gather(
                    *self.ref_model.async_init_model_from_pretrained(
                        self.strategy, os.path.join(self.cfg.save_path, f"iter{self.global_step}", "policy")
                    )
                )
                logger.info("Successfully update ref model with policy model, training continue.")

        await self.policy_model.async_save_model(self.tokenizer, self.cfg.num_episodes * len(self.prompts_dataloader))
        logger.info("Successfully save model weights, training done.")

    '''
    Example Countdown SFT Input: [(
    '<|endoftext|>A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nUsing the numbers [51, 17, 26, 15], create an equation that equals 27. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>\\boxed{(1 + 2) / 3}</answer>. Think step by step inside <think> tags.\nAssistant: <think>', 

    {'answer': '27', 'nums': '[51, 17, 26, 15]'}
    ), ...]

    Example Countdown Self-Play Query Gen Input: [(
    '<|endoftext|>A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nGiven the assistant goal, task generation success criteria, and example final answer, generate a task that fulfills the task generation success criteria. The generated task should be:\n\n1) maximizing user benefit and helpfulness and minimizing user risk, harm, and unwanted outcomes\n\n2) novel - avoid designing a task that is drawn from the same linguistic templates or differ only in superficial details\n\n3) difficult yet tractable - the task should be challenging for the assistant to complete, without being completely intractable\n\n4) clear - the task should be specific enough to not require further user input to understand what is being asked of the assistant and should provide all the context and detail necessary\n\nShow your work in <think> </think> tags. And return the task and final answer in <answer>\\boxed{ answer here }</answer> tags.\n\n\nAssistant Goal: Get really good at the Countdown Game with 3 to 4 numbers. The Countdown game is a numbers puzzle where players use a set of randomly drawn numbers and basic arithmetic operations (+, -, *, /) to reach a target number.\n\nTask Generation Success Criteria: The task and final answer must be a dictionary with the 'target', 'nums', and 'prompt' keys. The 'target' key should have a value that is an integer, the 'nums' key should have a value that is a list of 3 or 4 integers, and the 'prompt' key should have a value that is a string in the same exact format as the example final answer prompt, but just with the nums and target updated as appropriate.\n\nExample Final Answer: <answer>\\boxed{{\n    'target': 44,\n    'nums': [6, 77, 73, 20],\n    'prompt': 'Using the numbers [6, 77, 73, 20], create an equation that equals 44. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>\\boxed{(1 + 2) / 3}</answer>. Think step by step inside <think> tags.'\n}}</answer>\nAssistant: <think>', 

    {'answer': '27', 'nums': '[51, 17, 26, 15]'}
    ), ...]
    '''
    '''
    CORE TRAINING FUNCTION / LOSS FUNCTION HERE
    Input: rand_prompts: a list of prompt strings - List[Tuple[str, dict]]
    Output: update replay_buffer with policy (+ query) experiences: Experience

    Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.

    # Currently ONLY for self-play. self-play = False is not yet really supported yet, at least for logging I think?

    n_samples_per_prompt = self.num_generations!

    when self.self_play = False
    (before query branch) sample all_inputs[0]: ('<|endoftext|>A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nUsing the numbers [60, 59, 21], create an equation that equals 20. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>\\boxed{(1 + 2) / 3}</answer>. Think step by step inside <think> tags.\nAssistant: <think>', 
    
    {'answer': '20', 'nums': '[60, 59, 21]'})

    when self.self_play = True
    (query branch) sample prompt, all_inputs[0]: ("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.\nThis is the problem:\nGiven the assistant goal, task generation success criteria, and example final answer, generate a task that fulfills the task generation success criteria. The generated task should be:\n\n1) maximizing user benefit and helpfulness and minimizing user risk, harm, and unwanted outcomes\n\n2) novel - avoid designing a task that is drawn from the same linguistic templates or differ only in superficial details\n\n3) difficult yet tractable - the task should be challenging for the assistant to complete, without being completely intractable\n\n4) clear - the task should be specific enough to not require further user input to understand what is being asked of the assistant and should provide all the context and detail necessary\n\nAssistant Goal: Get really good at the Countdown Game with 3 to 4 numbers. The Countdown game is a numbers puzzle where players use a set of randomly drawn numbers and basic arithmetic operations (+, -, *, /) to reach a target number.\n\nTask Generation Success Criteria: The task and final answer must be a dictionary with the 'target', 'nums', and 'prompt' keys. The 'target' key should have a value that is an integer, the 'nums' key should have a value that is a list of 3 or 4 integers, and the 'prompt' key should have a value that is a string in the same exact format as the example final answer prompt, but just with the nums and target updated as appropriate.\n\nExample Reasoning and Final Answer Response Format: <think>\nreasoning process here\n</think>\n<answer>\n{\n    'target': 6,\n    'nums': [1, 2, 3],\n    'prompt': 'Using the numbers [1, 2, 3], create an equation that equals 6. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. '\n}\n</answer>\nAssistant: <think>", 
    
    {'answer': '', 'nums': ''})

    WITH HISTORY
    (query branch) sample prompt, all_inputs[0]: ("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.\nThis is the problem:
    Given the assistant goal, task generation success criteria, and example final answer, generate a task that fulfills the task generation success criteria. The generated task should be:
    
    1) maximizing user benefit and helpfulness and minimizing user risk, harm, and unwanted outcomes
    
    2) novel - avoid designing a task that is drawn from the same linguistic templates or differ only in superficial details
    
    3) difficult yet tractable - the task should be challenging for the assistant to complete, without being completely intractable
    
    4) clear - the task should be specific enough to not require further user input to understand what is being asked of the assistant and should provide all the context and detail necessary
    
    Assistant Goal: Get really good at the Countdown Game with 3 to 4 numbers. The Countdown game is a numbers puzzle where players use a set of randomly drawn numbers and basic arithmetic operations (+, -, *, /) to reach a target number.\n\nTask Generation Success Criteria: The task and final answer must be a dictionary with the 'target', 'nums', and 'prompt' keys. The 'target' key should have a value that is an integer, the 'nums' key should have a value that is a list of 3 or 4 integers, and the 'prompt' key should have a value that is a string in the same exact format as the example final answer prompt, but just with the nums and target updated as appropriate.\n\nExample Reasoning and Final Answer Response Format: <think>\nreasoning process here\n</think>\n<answer>\n{\n    'target': 6,\n    'nums': [1, 2, 3],\n    'prompt': 'Using the numbers [1, 2, 3], create an equation that equals 6. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. '\n}\n</answer>\nAssistant: <think>...</think>\n<answer>...</answer>\nQuery Reward: <reward>\nAssistant: <think>...</think>\n<answer>...</answer>\nPolicy Reward: <reward>\nAssistant: <think>", 

    ^ where the examples saved are the HIGHEST scoring queries and policies
    
    {'answer': '', 'nums': ''})

    (before policy branch) sample prompt: ('<|endoftext|>A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nUsing the numbers [51, 17, 26, 15], create an equation that equals 27. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer>\\boxed{(1 + 2) / 3}</answer>. Think step by step inside <think> tags.\nAssistant: <think>', 
    
    {'answer': '27', 'nums': '[51, 17, 26, 15]', 'valid_prompt': False})
    '''
    @torch.no_grad()
    async def make_experience(self, all_inputs: Union[Tuple[str, dict], List[Tuple[str, dict]]], **generate_kwargs):
        logger.debug("starting make_experience")
        logger.debug(f"(before query branch) prompt batch size (bs): len(all_inputs): {len(all_inputs)}") # Note: when self.self_play = True, then we GENERATE bs prompts to run rollouts on. If self.self_play = False, then we just use those bs prompts to run rollouts on. Therefore, we will always have bs prompts in either case!
        experiences = []

        ############################
        # Query Branch (executed if self_play is True)
        ############################
        if self.self_play:
            # Use original inputs as query inputs.
            query_prompts = [inp[0] for inp in all_inputs]
            query_extras = [inp[1] for inp in all_inputs]

            # Update query_prompts to include last history entry based on self.history
            if self.history:
                for i, prompt in enumerate(query_prompts):
                    # prefix = prompt.split("Assistant <think>")[0].strip() # Remove the last <think> to add back later # DEBUG: Hard-coding for now
                    prefix = '''A Policy Assistant Goal set by a User, and a conversation between a Query Assistant and a Policy Assistant. The User gives the Policy Assistant Goal and tasks the Query Assistant to generate a query to help the Policy Assistant better reach the Policy Assistant Goal, then the Policy Assistant tries to solve the Query Assistant's generated query, then the Query Assistant generates another query, then the Policy Assistant tries to solve that query, etc. After each Query Assistant response, there is a Query Reward where -1.0 means the response was considered potentially harmful, and a higher score means that the query was helpful to the Policy Assistant in achieving the goal and also of appropriate difficulty to the Policy Assistant. After each Policy Assistant response, there is Policy Reward where 0.0 means that the Policy Assistant was not able to reach the same target answer proposed by the Query Assistant and 1.0 means that the Policy Assistant was able to reach the same target answer. The Query and Policy Assistant both first think about the reasoning process in the mind and then provides the Policy or Query Assistant with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
This is the problem:
Given the Policy Assistant goal, task generation success criteria, and example final answer, generate a task that fulfills the task generation success criteria. The generated task should be:

1) maximizing user and Policy Assistant benefit and helpfulness and minimizing user and Policy Assistant risk, harm, and unwanted outcomes

2) novel - avoid designing a task that is drawn from the same linguistic templates or differ only in superficial details

3) difficult yet tractable - the task should be challenging for the Policy Assistant to complete, without being completely intractable

4) clear - the task should be specific enough to not require further user input to understand what is being asked of the Policy Assistant and should provide all the context and detail necessary

Policy Assistant Goal: Get really good at math.

Task Generation Success Criteria: The task and final answer must be a dictionary with the 'target' and 'prompt' keys. The 'target' key should have a value that is an integer and the 'prompt' key should have a value that is a string. The prompt is meant to be the task given to the Policy Assistant to solve, and the Policy Assistant's final answer will be considered complete and correct if the Policy Assistant's final answer matches the 'target' value you propose as the answer to the problem. Therefore, it is likely worthwhile for you to ensure that the 'target' value you propose is indeed the only correct solution to the problem you propose as the 'prompt' value.

Example Query Assistant Reasoning and Final Answer Response Format: <think>
reasoning process here
</think>
<answer>
{\n    'target': 2,\n    'prompt': 'What is 1 + 1?'\n}
</answer>'''
                    prompt_w_history = prefix
                    
                    if len(self.history) >= 6:
                        last_query_response, last_query_reward = self.history[-6]
                        last_policy_response, last_policy_reward = self.history[-5]
                        prompt_w_history += f"\n\nQuery Assistant: {last_query_response}\n\nQuery Reward: {last_query_reward}\n\nPolicy Assistant: {last_policy_response}\n\nPolicy Reward: {last_policy_reward}"
                    
                    if len(self.history) >= 4:
                        last_query_response, last_query_reward = self.history[-4]
                        last_policy_response, last_policy_reward = self.history[-3]
                        prompt_w_history += f"\n\nQuery Assistant: {last_query_response}\n\nQuery Reward: {last_query_reward}\n\nPolicy Assistant: {last_policy_response}\n\nPolicy Reward: {last_policy_reward}"
                        
                    if len(self.history) >= 2:
                        last_query_response, last_query_reward = self.history[-2]
                        last_policy_response, last_policy_reward = self.history[-1]
                        prompt_w_history += f"\n\nQuery Assistant: {last_query_response}\n\nQuery Reward: {last_query_reward}\n\nPolicy Assistant: {last_policy_response}\n\nPolicy Reward: {last_policy_reward}"

                    prompt_w_history += "\n\nQuery Assistant: <think>"
                    query_prompts[i] = prompt_w_history
            
            # 0. generate query sequences and inference
            # 0.1 generate sequences via vllm engines
            outputs_query = []
            num_vllm_dp_groups = len(self.vllm_engines)

            async with Timer("Generate query completions via vllm engines"):
                dp_prompt_size = (len(query_prompts) + num_vllm_dp_groups - 1) // num_vllm_dp_groups
                dp_tasks = []
                for dp_rank in range(num_vllm_dp_groups):
                    dp_inputs = query_prompts[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                    dp_extras = query_extras[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                    if len(dp_inputs) <= 0:
                        continue
                    gen_func = self._get_generate_function(dp_rank)
                    dp_tasks.append(self.generate_vllm(gen_func, dp_inputs, extras=dp_extras, exp_type="query", **generate_kwargs)) 
                    # self.generate_vllm returns 
                    # [dict(
                    #     response=response, # query response
                    #     iscorrect=iscorrect, # dummy to be overwritten
                    #     stop_reason=stop_reason,
                    #     final_answer=final_answer, # extracted
                    # ), ...]


                logger.info("Start query generation")
                local_query_responses = await asyncio.gather(*dp_tasks)
                outputs_query.extend(sum(local_query_responses, []))
                logger.info("generate local query rollout batch done")

                # # offload vllm engines when colocate all models # this resets the internal state of vllm and causes some issues (policy vllm to generate gibberish), even with backloading after
                # if self.cfg.colocate_all:
                #     async with Timer("Offload vllm engines to cpu after query generation"):
                #         await self._offload_vllm_engines()
            

            # NEW: --- Evaluator Network LM Reward Calculation on the Query Generations --- (currently targeted for query generations, dummy for policy generations)
            # 1. Build a list of valid query ideas (those with non-empty final answers)
            final_answers = [out.get("final_answer", "") for out in outputs_query]
            responses = [out.get("response", "") for out in outputs_query]
            valid_indices = [i for i, fa in enumerate(final_answers) if fa != ""]
            valid_responses = [responses[i] for i in valid_indices]

            evaluator_result = None
            if valid_responses:
                # Build the combined query ideas string.
                queries_ideas_str = "\n\n".join(
                    f"Query Idea {j+1}:\n<think>{valid_responses[j]}" for j in range(len(valid_responses))
                )
                user_goal = self.cfg.user_goal_verbatim
                evaluator_prompt = f'''A conversation between a User and an Evaluator Assistant. The User asks a question, and the Evaluator Assistant solves it. The Evaluator Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.\nThe reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, i.e. your response format should be: <think> reasoning process here </think>\n<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.\nThis is the problem:
Given 1) the user goal for a Policy Assistant and 2) a Query Assistant's Query Ideas (ie. thoughts and answers) for a prompt to ask the Policy Assistant and a target success criteria to evaluate the Policy Assistant on, complete the following:

1. Harmfulness Categorization: Categorize each Query Ideas' entire thought process and final answers into a) not harmful, b) potentially harmful, and c) harmful to others nor the environment if attempted. To help evaluate harmfulness, it is worth considering: 
- Harm to others and environment: if the Policy Assistant attempts to solve the Query Idea's prompt and achieve the target success criteria, would their attempt potentially harm others' own attempts in achieving their own goals? 
- Harm to Policy Assistant: Also consider, is the Query Idea's proposed target success criteria even a reasonable, logical, and credible target success criteria for the proposed prompt for the Policy Assistant try to reach and learn better behaviour from reaching? If it is not obvious whether or not the prompt should logically lead to the given target success criteria, then the Query Idea should be categorized as potentially harmful as the Policy Assistant may learn the wrong and illogical behaviour which is not helpful to helping the Policy Assistant better reach the user goal.

2. Significance Ranking: Within each harmfulness category, rank each Query Idea based on significance relative to the user goal, with the most significant being ranked first and the least significant being ranked last in the category list. To help evaluate significance between Query Ideas, it is worth considering: if the assistant is able to solve the Query Idea's prompt and achieve the target success criteria, which end state would be more meaningful/useful/contribute to bringing the Policy Assistant to a state closer to achieving the user goal? 

User Goal for the Policy Assistant: {user_goal}

Query Ideas: 

{queries_ideas_str}

Please also show your work in <think> tags, and output the final harmfulness categorization and significance ranking of the Query Ideas within <answer> tags. Example output format: <think>
thinking here
</think>
<answer>
{{
    'not_harmful': [<int>, <int>, …, <int>],
    'potentially_harmful': [<int>, <int>, …, <int>],
    'harmful': [<int>, <int>, …, <int>]
}}
</answer>
Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> answer here </answer>. Think step by step inside <think> tags.\nEvaluator Assistant: <think>'''
                
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
                
                evaluator_responses, evaluator_stop_reasons = await gen_func(
                    prompts=[evaluator_prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    truncate_prompt=True,
                )
                evaluator_text = evaluator_responses[0]
                if not isinstance(evaluator_text, str):
                    evaluator_text = evaluator_text.outputs[0].text

                # Log the evaluator prompt and the evaluator LM output.
                logger.info(f"[STEP {self.global_step}] [Evaluator] prompt: {evaluator_prompt}")
                logger.info(f"[STEP {self.global_step}] [Evaluator] response: {evaluator_text}")

                # Tokenize evaluator response to compute token length.
                tokenized = self._tokenize(evaluator_text, self.cfg.generate_max_len, padding=False)
                response_tokens = tokenized["input_ids"]
                self.writer.add_scalar("evaluator/response_length", len(response_tokens), self.global_step)

                evaluator_text = "<think>" + evaluator_text.strip()
                regex_eval = r"^<think>.*<\/think>\n<answer>([\s\S]*)<\/answer>$"
                match_eval = re.search(regex_eval, evaluator_text, re.DOTALL)
                if match_eval:
                    evaluator_answer_text = match_eval.group(1).strip()
                    try:
                        evaluator_result = ast.literal_eval(evaluator_answer_text)
                    except Exception as e:
                        logger.error(f"Evaluator result parse error: {e}")
                        evaluator_result = None
                else:
                    evaluator_result = None

                # Log via tensorboard whether the evaluator result is correctly formatted.
                self.writer.add_scalar("evaluator/result_format_correct", 1 if evaluator_result is not None else 0, self.global_step)

            # Compute evaluator LM rewards for valid query ideas.
            valid_rewards = {}
            if evaluator_result is None or not isinstance(evaluator_result, dict):
                # If evaluator output is invalid, default all valid rewards to 1.0.
                for j in range(len(valid_responses)):
                    valid_rewards[j] = 0.0
            else:
                # Extract the three lists.
                not_harmful_list = evaluator_result.get("not_harmful", [])
                potentially_harmful_list = evaluator_result.get("potentially_harmful", [])
                harmful_list = evaluator_result.get("harmful", [])

                try:
                    # Combine them into a single list.
                    combined_list = not_harmful_list + potentially_harmful_list + harmful_list
                    # Expected list: every valid query idea, numbered 1 through N.
                    expected_list = list(range(1, len(valid_responses) + 1))
                    # Check that combined_list (sorted) matches expected_list exactly and has no duplicates.
                    if sorted(combined_list) == expected_list and len(combined_list) == len(expected_list):
                        # Every valid query idea is uniquely scored.
                        for j in range(len(valid_responses)):
                            idea_num = j + 1  # 1-indexed.
                            if idea_num in not_harmful_list:
                                # Use the order in the not_harmful_list for ranking.
                                rank = not_harmful_list.index(idea_num)  # 0-based rank; lower is better.
                                reward_val = (len(not_harmful_list) - rank) / len(not_harmful_list)
                                valid_rewards[j] = reward_val
                            else:
                                # If in potentially_harmful_list or harmful_list, assign a penalty.
                                valid_rewards[j] = -1.0

                        logger.info(f"[STEP {self.global_step}] [Evaluator] Results after valid ranking of {len(valid_rewards)} queries: {valid_rewards}") # CTRL+F for this to know it works
                    else:
                        # If any valid query idea is missing or duplicates exist, default all rewards to 1.0.
                        for j in range(len(valid_responses)):
                            valid_rewards[j] = 0.0
                except Exception as e:
                    logger.error(f"[STEP {self.global_step}] [Evaluator] Evaluator result processing error: {e}")
                    # If the lists are not all lists, default all valid rewards to 1.0.
                    for j in range(len(valid_responses)):
                        valid_rewards[j] = 0.0

                # Log a scalar to TensorBoard indicating if valid_rewards is not all ones.
                # It will be 1 if at least one reward is not 1.0, and 0 otherwise.
                valid_rewards_not_all_ones = 1.0 if valid_rewards and any(v != 0.0 for v in valid_rewards.values()) else 0.0
                self.writer.add_scalar("evaluator/valid_rewards_not_all_ones", valid_rewards_not_all_ones, self.global_step)

            # 3. Build the final evaluator_lm_rewards list for all responses (in the original order)
            evaluator_lm_rewards = []
            valid_iter = 0
            for i, fa in enumerate(final_answers):
                if fa == "":
                    # For responses with invalid format, assign -0.5
                    evaluator_lm_rewards.append(-0.5)
                else:
                    evaluator_lm_rewards.append(valid_rewards.get(valid_iter, 0.0))
                    valid_iter += 1
            logger.info(f"[STEP {self.global_step}] [Evaluator] evaluator_lm_rewards: {evaluator_lm_rewards}")

            # Add 'evaluator_lm_reward' to the outputs_query list.
            for i, out in enumerate(outputs_query):
                out["evaluator_lm_reward"] = evaluator_lm_rewards[i]


            ############################
            # Post-process query completions
            ############################
            def fix_multiline_string(dict_str):
                pattern = r"('prompt':\s*')(.*?)(')"
                def replacer(match):
                    start, content, end = match.groups()
                    fixed_content = content.replace('\n', '\\n')
                    return f"{start}{fixed_content}{end}"
                return re.sub(pattern, replacer, dict_str, flags=re.DOTALL)
            
            def extract_answer_input(query_item):
                system_prefix = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.\nThe reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, i.e. your response format should be: <think> reasoning process here </think>\n<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.\nThis is the problem:\n"
                assistant_suffix = " Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> answer here </answer>. Think step by step inside <think> tags.\nAssistant: <think>" # Give example final answer to help policy model understand output better

                # TODO: adjust these default values based on domain / dataset! Pull from example final answer from prompt
                if self.cfg.goal == "countdown":
                    default_prompt = "Using the numbers [1, 2, 3], create an equation that equals 6. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once." # dummy super easy problem, I KNOW too difficult of a problem makes policy learning more difficult
                    default_extras = {'target': 6, 'nums': [1, 2, 3], 'valid_prompt': False, 'answer': 6}
                    required_keys = {"target", "nums", "prompt"}
                elif self.cfg.goal == "math":
                    default_prompt = "What is 1 + 2?"
                    default_extras = {'target': 3, 'valid_prompt': False, 'answer': 3}
                    required_keys = {"target", "prompt"}
                
                default_full_prompt = system_prefix + default_prompt + assistant_suffix
                default_valid_answer_input = (
                    default_full_prompt,
                    default_extras
                )
                
                try:
                    dict_str = query_item            # Content inside <answer>...</answer>
                    dict_str_fixed = fix_multiline_string(dict_str)
                    answer_input = ast.literal_eval(dict_str_fixed)
                    if not isinstance(answer_input, dict) or not required_keys.issubset(answer_input.keys()):
                        return default_valid_answer_input
                    prompt_text = str(answer_input.get('prompt')).strip()
                    full_prompt = system_prefix + prompt_text + assistant_suffix
                    answer_input['prompt'] = full_prompt
                    answer_input['target'] = str(float(answer_input.get('target'))) if self.cfg.goal != "countdown" else int(answer_input.get('target')) # don't force to be an int unless for countdown
                    answer_input['answer'] = answer_input['target']
                    answer_input['nums'] = [int(x) for x in answer_input.get('nums', [])]

                    answer_input['valid_prompt'] = True
                    return (full_prompt, answer_input)
                except Exception as e:
                    return default_valid_answer_input
            
            # Post-process query responses and build a mapping for query completions.
            query_prompt_mapping = {}
            for idx, comp in enumerate(outputs_query):
                query_prompt_mapping[idx] = extract_answer_input(comp.get("final_answer", ""))
            
            # Log for inspection
            logger.debug("\n\nINSPECT QUERY BRANCH PROMPTS AND RESPONSES")
            logger.debug(f"(query branch) sample prompt, all_inputs[0]: {all_inputs[0]}\n")
            logger.debug(f"(query branch) sample response, outputs_query[0]: {outputs_query[0]}\n\n")
            logger.debug(f"(query branch) sample prompt, query_prompt_mapping[0]: {query_prompt_mapping[0]}\n")

            # Update all_inputs to be the list of processed (policy_prompt, policy_extras) tuples.
            all_inputs = [(query_prompt_mapping[idx][0], query_prompt_mapping[idx][1]) for idx in range(len(query_prompt_mapping))]

            # Also keep raw query responses for later inference.
            query_texts = outputs_query
            query_output_extras = [query_prompt_mapping[idx][1] for idx in range(len(query_prompt_mapping))]

            # skip when data is not enough
            if len(outputs_query) <= 0:
                return

            assert len(query_prompts) == len(outputs_query), "generate query objects number must be equal to all query inputs number"
            

        logger.debug("\n--- Policy Branch ---\n")
        logger.debug(f"(before policy branch) prompt batch size (bs): len(all_inputs): {len(all_inputs)}") # Note: when self.self_play = True, then we GENERATE bs prompts to run rollouts on. If self.self_play = False, then we just use those bs prompts to run rollouts on. Therefore, we will always have bs prompts in either case!
        logger.debug(f"(before policy branch) sample all_inputs[0]: {all_inputs[0]}")
        experiences = []
        all_prompts = sum([[prompt[0]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])
        all_extras = sum([[prompt[1]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])
        # shuffle all_prompts and all_extras together
        indices = list(range(len(all_prompts)))
        # rng = random.Random(42) # no shuffle to ensure that when self.self_play = True, the order of all_prompts is the same as query_prompts
        # rng.shuffle(indices)
        all_prompts = [all_prompts[i] for i in indices]
        all_extras = [all_extras[i] for i in indices]

        ############################
        # Policy Branch: Generate sequences via vLLM engines
        ############################
        # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence
        # 1.1 generate sequences via vllm engines
        outputs = []
        num_vllm_dp_gruops = len(self.vllm_engines)

        async with Timer("Generate sequences via vllm engines"):
            dp_prompt_size = (len(all_prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops
            dp_tasks = []
            for dp_rank in range(num_vllm_dp_gruops):
                dp_inputs = all_prompts[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                dp_extras = all_extras[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                # handle last batch has no enough data
                if len(dp_inputs) <= 0:
                    continue
                gen_func = self._get_generate_function(dp_rank)
                dp_tasks.append(self.generate_vllm(gen_func, dp_inputs, extras=dp_extras, exp_type="policy", **generate_kwargs))

            logger.info("start generation")
            local_responses = await asyncio.gather(*dp_tasks)
            outputs.extend(sum(local_responses, []))
            outputs_dicts = outputs
            logger.info("generate local rollout batch done")

            # offload vllm engines when colocate all models
            if self.cfg.colocate_all:
                async with Timer("Offload vllm engines to cpu"):
                    await self._offload_vllm_engines()

        # Log for inspection
        logger.debug("\n\nINSPECT POLICY BRANCH PROMPTS AND RESPONSES")
        logger.debug(f"(policy branch) sample prompt, all_inputs[0]: {all_inputs[0]}\n")
        logger.debug(f"(policy branch) sample response, outputs[0]: {outputs[0]}\n\n")

        # skip when data is not enough
        if len(outputs) <= 0:
            return

        assert len(all_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

        ############################
        # Custom Reward Calculation
        ############################
        # 1.2 calculate custom rewards if has custom reward function
        if self.cfg.use_compute_reward_fn:
            async with Timer("Calculate custom rewards"):
                dp_tasks = []
                reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                all_prompts, outputs, custom_rewards = await reward_fn(all_prompts, outputs, all_extras)
                assert len(all_prompts) == len(
                    outputs
                ), "generate objects number after custom reward function must be equal to all inputs number"
        else:
            all_prompts, outputs, custom_rewards = all_prompts, outputs, None

        # empty data
        if len(all_prompts) == 0:
            return

        ############################
        # Packing Samples
        ############################
        # 1.3 packing samples
        async with Timer("Packing samples"):
            (
                ret_sequences,
                ret_attention_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                all_prompts, outputs, custom_rewards, self.cfg.packing_max_len
            )
            action_masks = None

        ############################
        # Inference and PPO Computations (Policy Experiences)
        ############################
        # 1.4 inference and calculate values, log probs, rewards, kl divergence
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence"):
            experiences = await self.inference_and_calculates(
                ret_sequences,
                ret_attention_masks,
                action_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
                exp_type="policy",
            )
            logger.info(f"experiences size: {len(experiences)}")

        ############################
        # Visualization
        ############################
        # 2. visualization generated results example
        vis = self._detokenize(experiences[0].sequences[0][: int(experiences[0].info["total_length"].flatten()[0])])
        self.writer.add_text("generated_sequences", vis, self.global_step)
        self.writer.flush()

        ############################
        # Advantages and Returns Calculation and Logging
        ############################
        # 3. calculate advantages and returns / along with tensorboard logging
        reward_list = []
        kl_list = []
        kl_max_list = []
        response_length_list = []
        orm_score_list = []
        custom_rewards_list = []
        advantages_list = []
        advantages_abs_list = []

        async with Timer("Calculate advantages and returns"):
            adv_tasks = []
            for experience in experiences:
                adv_tasks.append(self._calc_advantages_and_returns(experience))

            for tsk in asyncio.as_completed(adv_tasks):
                experience, metrics = await tsk
                reward_list.append(metrics["avg_rewards"])
                kl_list.append(metrics["avg_kl"])
                kl_max_list.append(metrics["avg_kl_max"])
                response_length_list.append(metrics["avg_response_length"])
                orm_score_list.append(metrics["avg_orm_score"])
                custom_rewards_list.append(metrics["avg_custom_rewards"])
                advantages_list.append(metrics["avg_advantages"])
                advantages_abs_list.append(metrics["avg_advantages_abs"])
                self.replay_buffer.append(experience)

        def safe_mean(values):
            """Returns the mean of the list, or 0.0 if the list is empty."""
            return np.mean(values) if values else 0.0

        def safe_std(values):
            """Returns the standard deviation of the list, or 0.0 if the list is empty."""
            return np.std(values) if values else 0.0

        avg_rewards = safe_mean(reward_list)
        std_rewards = safe_std(reward_list)
        avg_kl = safe_mean(kl_list)
        std_kl = safe_std(kl_list)
        avg_kl_max = safe_mean(kl_max_list)
        std_kl_max = safe_std(kl_max_list)
        avg_response_length = safe_mean(response_length_list)
        std_response_length = safe_std(response_length_list)
        avg_orm_score = safe_mean(orm_score_list)
        std_orm_score = safe_std(orm_score_list)
        avg_custom_rewards = safe_mean(custom_rewards_list)
        std_custom_rewards = safe_std(custom_rewards_list)
        avg_advantages = safe_mean(advantages_list)
        std_advantages = safe_std(advantages_list)
        avg_advantages_abs = safe_mean(advantages_abs_list)
        std_advantages_abs = safe_std(advantages_abs_list)

        ############################
        # Logging Policy Metrics
        ############################
        # 4. tensorboard logging
        logger.info(
            f"avg_raw_rewards: {avg_rewards}, avg_kl: {avg_kl}, avg_response_length: {avg_response_length}, avg_orm_score: {avg_orm_score}, avg_custom_rewards: {avg_custom_rewards}"
        )
        self.writer.add_scalar("policy/avg_raw_rewards", avg_rewards, self.global_step)
        self.writer.add_scalar("policy/std_raw_rewards", std_rewards, self.global_step)
        self.writer.add_scalar("policy/avg_kl", avg_kl, self.global_step)
        self.writer.add_scalar("policy/std_kl", std_kl, self.global_step)
        self.writer.add_scalar("policy/avg_kl_max", avg_kl_max, self.global_step)
        self.writer.add_scalar("policy/std_kl_max", std_kl_max, self.global_step)
        self.writer.add_scalar("policy/avg_response_length", avg_response_length, self.global_step)
        self.writer.add_scalar("policy/std_response_length", std_response_length, self.global_step)
        self.writer.add_scalar("policy/avg_orm_score", avg_orm_score, self.global_step)
        self.writer.add_scalar("policy/std_orm_score", std_orm_score, self.global_step)
        self.writer.add_scalar("policy/avg_custom_rewards", avg_custom_rewards, self.global_step)
        self.writer.add_scalar("policy/std_custom_rewards", std_custom_rewards, self.global_step)
        self.writer.add_scalar("policy/avg_advantages", avg_advantages, self.global_step)
        self.writer.add_scalar("policy/std_advantages", std_advantages, self.global_step)
        self.writer.add_scalar("policy/avg_advantages_abs", avg_advantages_abs, self.global_step)
        self.writer.add_scalar("policy/std_advantages_abs", std_advantages_abs, self.global_step)
        self.writer.flush()


        ############################
        # Compute Query Rewards (Grouped by replication)
        ############################
        if self.self_play:
            # 0.2 calculate custom query rewards if has custom reward function
            R = self.cfg.n_samples_per_prompt  # Number of policy completions per query

            # --- Calculate custom query rewards ---
            async with Timer("Calculate custom query rewards"):
                # custom_query_reward_fn returns a list of reward tensors,
                # one per query output, with only the last token nonzero.
                query_prompts, query_outputs_responses, query_custom_rewards = await self.custom_query_reward_fn(
                    query_prompts,    # list of processed query prompt strings
                    query_texts,      # list of query output dictionaries (each with a "response" key)
                    query_output_extras,     # list of extras for each query (containing "valid_prompt", etc.)
                    custom_rewards,   # list of policy custom reward tensors (for grouping)
                    R
                )
                assert len(query_custom_rewards) == len(query_prompts), "query custom rewards number must be equal to all query inputs number"
            
            # empty data
            if len(query_custom_rewards) == 0:
                return

            # 0.3 Packing Query Samples
            async with Timer("Packing query samples"):
                (
                    query_ret_sequences,
                    query_ret_attention_masks,
                    query_ret_num_actions,
                    query_ret_packed_seq_lens,
                    query_ret_custom_rewards
                ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                    query_prompts, query_outputs_responses, query_custom_rewards, self.cfg.packing_max_len
                )
                action_masks = None

            # 0.4 Inference for Query Branch
            async with Timer("Inference and calculate values, log probs, rewards, kl divergence (Query)"):
                query_experiences = await self.inference_and_calculates(
                    query_ret_sequences,
                    query_ret_attention_masks,
                    action_masks,
                    query_ret_num_actions,
                    query_ret_packed_seq_lens,
                    query_ret_custom_rewards,
                    exp_type="query",
                )
                logger.info(f"Query experiences size: {len(query_experiences)}")

            # Visualization of a query response (optional)
            vis = self._detokenize(query_experiences[0].sequences[0][: int(query_experiences[0].info["total_length"].flatten()[0])])
            self.writer.add_text("query_generated_sequences", vis, self.global_step)
            self.writer.flush()

            # Calculate advantages and returns for the query branch.
            query_rewards_list = []
            query_kl_list = []
            query_kl_max_list = []
            query_response_length_list = []
            query_orm_score_list = []
            query_custom_rewards_list = []
            query_advantages_list = []
            query_advantages_abs_list = []

            async with Timer("Calculate query advantages and returns"):
                query_adv_tasks = []
                for exp in query_experiences:
                    query_adv_tasks.append(self._calc_advantages_and_returns(exp))
                for tsk in asyncio.as_completed(query_adv_tasks):
                    experience, metrics = await tsk
                    query_rewards_list.append(metrics["avg_rewards"])
                    query_kl_list.append(metrics["avg_kl"])
                    query_kl_max_list.append(metrics["avg_kl_max"])
                    query_response_length_list.append(metrics["avg_response_length"])
                    query_orm_score_list.append(metrics["avg_orm_score"])
                    query_custom_rewards_list.append(metrics["avg_custom_rewards"])
                    query_advantages_list.append(metrics["avg_advantages"])
                    query_advantages_abs_list.append(metrics["avg_advantages_abs"])

                    # have the same # of query and policy experiences # think the query overshoots basically and goes way off target?
                    # for _ in range(self.cfg.n_samples_per_prompt): # LEARNING: NOT GREAT
                    self.replay_buffer.append(experience)

            # 4. tensorboard logging
            num_query_exps = len(query_experiences)
            avg_query_rewards = safe_mean(query_rewards_list)
            std_query_rewards = safe_std(query_rewards_list)
            avg_query_kl = safe_mean(query_kl_list)
            std_query_kl = safe_std(query_kl_list)
            avg_query_kl_max = safe_mean(query_kl_max_list)
            std_query_kl_max = safe_std(query_kl_max_list)
            avg_query_response_length = safe_mean(query_response_length_list)
            std_query_response_length = safe_std(query_response_length_list)
            avg_query_orm_score = safe_mean(query_orm_score_list)
            std_query_orm_score = safe_std(query_orm_score_list)
            avg_query_custom_rewards = safe_mean(query_custom_rewards_list)
            std_query_custom_rewards = safe_std(query_custom_rewards_list)
            avg_query_advantages = safe_mean(query_advantages_list)
            std_query_advantages = safe_std(query_advantages_list)
            avg_query_advantages_abs = safe_mean(query_advantages_abs_list)
            std_query_advantages_abs = safe_std(query_advantages_abs_list)
            logger.info(
                f"Query metrics: avg_query_rewards: {avg_query_rewards:.4f}, "
                f"avg_query_kl: {avg_query_kl:.4f}, "
                f"avg_query_kl_max: {avg_query_kl_max:.4f}, "
                f"avg_query_response_length: {avg_query_response_length:.4f}, "
                f"avg_query_orm_score: {avg_query_orm_score:.4f}, "
                f"avg_query_custom_rewards: {avg_query_custom_rewards:.4f}, "
                f"avg_query_advantages: {avg_query_advantages:.4f}, "
                f"avg_query_advantages_abs: {avg_query_advantages_abs:.4f}"
            )
            self.writer.add_scalar("query/avg_rewards", avg_query_rewards, self.global_step)
            self.writer.add_scalar("query/std_rewards", std_query_rewards, self.global_step)
            self.writer.add_scalar("query/avg_kl", avg_query_kl, self.global_step)
            self.writer.add_scalar("query/std_kl", std_query_kl, self.global_step)
            self.writer.add_scalar("query/avg_kl_max", avg_query_kl_max, self.global_step)
            self.writer.add_scalar("query/std_kl_max", std_query_kl_max, self.global_step)
            self.writer.add_scalar("query/avg_response_length", avg_query_response_length, self.global_step)
            self.writer.add_scalar("query/std_response_length", std_query_response_length, self.global_step)
            self.writer.add_scalar("query/avg_orm_score", avg_query_orm_score, self.global_step)
            self.writer.add_scalar("query/std_orm_score", std_query_orm_score, self.global_step)
            self.writer.add_scalar("query/avg_custom_rewards", avg_query_custom_rewards, self.global_step)
            self.writer.add_scalar("query/std_custom_rewards", std_query_custom_rewards, self.global_step)
            self.writer.add_scalar("query/avg_advantages", avg_query_advantages, self.global_step)
            self.writer.add_scalar("query/std_advantages", std_query_advantages, self.global_step)
            self.writer.add_scalar("query/avg_advantages_abs", avg_query_advantages_abs, self.global_step)
            self.writer.add_scalar("query/std_advantages_abs", std_query_advantages_abs, self.global_step)
            self.writer.flush()


            # HELPER FUNCTIONS FOR ADDITIONAL METRICS
            query_completions = [out.get("response", "") for out in outputs_query if out.get("response", "").strip() != ""]

            def solvability_reward_func(completions, **kwargs):
                """
                For each completion, extracts the answer dictionary from the <answer> block and 
                checks whether the puzzle is solvable using basic arithmetic operations.
                Returns a list of rewards (1.0 if solvable, 0.0 otherwise).
                """
                def fix_multiline_string(dict_str):
                    pattern = r"('prompt':\s*')(.*?)(')"
                    def replacer(match):
                        start, content, end = match.groups()
                        fixed_content = content.replace('\n', '\\n')
                        return f"{start}{fixed_content}{end}"
                    return re.sub(pattern, replacer, dict_str, flags=re.DOTALL)
                
                def extract_answer_dict(query_item):
                    pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
                    match = re.search(pattern, query_item, re.DOTALL)
                    if not match:
                        return {'target': None, 'nums': None, 'prompt': None}
                    dict_str = match.group(1)
                    dict_str_fixed = fix_multiline_string(dict_str)
                    try:
                        answer_dict = ast.literal_eval(dict_str_fixed)
                        if not isinstance(answer_dict, dict):
                            return {'target': None, 'nums': None, 'prompt': None}
                    except Exception:
                        return {'target': None, 'nums': None, 'prompt': None}
                    try:
                        answer_dict['target'] = int(answer_dict.get('target'))
                    except Exception:
                        answer_dict['target'] = None
                    nums = answer_dict.get('nums')
                    if isinstance(nums, list):
                        try:
                            answer_dict['nums'] = [int(x) for x in nums]
                        except Exception:
                            answer_dict['nums'] = None
                    else:
                        answer_dict['nums'] = None
                    prompt = answer_dict.get('prompt')
                    answer_dict['prompt'] = prompt if isinstance(prompt, str) else (str(prompt) if prompt is not None else None)
                    return answer_dict

                def apply_operator(x, y, op):
                    if op == '+':
                        return x + y
                    if op == '-':
                        return x - y if x > y else None
                    if op == '*':
                        return x * y
                    if op == '/' and y != 0 and x % y == 0:
                        return x // y
                    return None

                def is_solvable(nums, target):
                    try:
                        ops = ['+', '-', '*', '/']
                        for num_perm in permutations(nums):
                            for op_set in product(ops, repeat=len(nums) - 1):
                                result = num_perm[0]
                                for i, op in enumerate(op_set):
                                    result = apply_operator(result, num_perm[i+1], op)
                                    if result is None:
                                        break
                                    if result == target:
                                        return True
                        return False
                    except Exception:
                        return False

                rewards = []
                for completion in completions:
                    try:
                        # Prepend <think> if needed for regex matching
                        completion = "<think>" + completion
                        answer_dict = extract_answer_dict(completion)
                        target = answer_dict.get('target')
                        nums = answer_dict.get('nums')
                        if target is None or nums is None:
                            rewards.append(0.0)
                        else:
                            rewards.append(1.0 if is_solvable(nums, target) else 0.0)
                    except Exception:
                        rewards.append(0.0)
                return rewards

            def diversity_reward_func(completions, **kwargs):
                """
                Computes a per-completion diversity reward.
                For each completion, extracts a key (target, tuple(nums)) from the answer dictionary.
                If a given key appears k times among N completions, the reward is:
                    reward = 1 - ((k - 1) / (N - 1))   if N > 1,
                    reward = 1                         if N == 1.
                """
                def fix_multiline_string(dict_str):
                    pattern = r"('prompt':\s*')(.*?)(')"
                    def replacer(match):
                        start, content, end = match.groups()
                        fixed_content = content.replace('\n', '\\n')
                        return f"{start}{fixed_content}{end}"
                    return re.sub(pattern, replacer, dict_str, flags=re.DOTALL)

                def extract_answer_dict(query_item):
                    pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
                    match = re.search(pattern, query_item, re.DOTALL)
                    if not match:
                        return {'target': None, 'nums': None, 'prompt': None}
                    dict_str = match.group(1)
                    dict_str_fixed = fix_multiline_string(dict_str)
                    try:
                        answer_dict = ast.literal_eval(dict_str_fixed)
                        if not isinstance(answer_dict, dict):
                            return {'target': None, 'nums': None, 'prompt': None}
                    except Exception:
                        return {'target': None, 'nums': None, 'prompt': None}
                    try:
                        answer_dict['target'] = int(answer_dict.get('target'))
                    except Exception:
                        answer_dict['target'] = None
                    nums = answer_dict.get('nums')
                    if isinstance(nums, list):
                        try:
                            answer_dict['nums'] = [int(x) for x in nums]
                        except Exception:
                            answer_dict['nums'] = None
                    else:
                        answer_dict['nums'] = None
                    prompt = answer_dict.get('prompt')
                    answer_dict['prompt'] = prompt if isinstance(prompt, str) else (str(prompt) if prompt is not None else None)
                    return answer_dict

                puzzle_keys = []
                for comp in completions:
                    comp = "<think>" + comp
                    answer_dict = extract_answer_dict(comp)
                    target = answer_dict.get('target')
                    nums = answer_dict.get('nums')
                    key = (target, tuple(nums)) if nums is not None else (target, None)
                    puzzle_keys.append(key)
                
                total = len(completions)
                freq = {}
                for key in puzzle_keys:
                    freq[key] = freq.get(key, 0) + 1

                rewards = []
                if total == 1:
                    return [1.0]
                
                for key in puzzle_keys:
                    k = freq.get(key, 0)
                    reward = 1 - ((k - 1) / (total - 1))
                    rewards.append(reward)
                return rewards

            def avg_num_solution_paths_reward_func(completions, **kwargs):
                """
                For each completion, extracts the answer dictionary and counts the number of valid solution paths.
                Returns a list of counts (floats) representing the number of valid solution paths for each puzzle.
                """
                def fix_multiline_string(dict_str):
                    pattern = r"('prompt':\s*')(.*?)(')"
                    def replacer(match):
                        start, content, end = match.groups()
                        fixed_content = content.replace('\n', '\\n')
                        return f"{start}{fixed_content}{end}"
                    return re.sub(pattern, replacer, dict_str, flags=re.DOTALL)

                def extract_answer_dict(query_item):
                    pattern = r"<answer>\s*(\{.*?\})\s*</answer>"
                    match = re.search(pattern, query_item, re.DOTALL)
                    if not match:
                        return {'target': None, 'nums': None, 'prompt': None}
                    dict_str = match.group(1)
                    dict_str_fixed = fix_multiline_string(dict_str)
                    try:
                        answer_dict = ast.literal_eval(dict_str_fixed)
                    except Exception:
                        return {'target': None, 'nums': None, 'prompt': None}
                    try:
                        answer_dict['target'] = int(answer_dict.get('target'))
                    except Exception:
                        answer_dict['target'] = None
                    nums = answer_dict.get('nums')
                    if isinstance(nums, list):
                        try:
                            answer_dict['nums'] = [int(x) for x in nums]
                        except Exception:
                            answer_dict['nums'] = None
                    else:
                        answer_dict['nums'] = None
                    prompt = answer_dict.get('prompt')
                    answer_dict['prompt'] = prompt if isinstance(prompt, str) else (str(prompt) if prompt is not None else None)
                    return answer_dict

                def apply_operator(x, y, op):
                    if op == '+':
                        return x + y
                    if op == '-':
                        return x - y if x > y else None
                    if op == '*':
                        return x * y
                    if op == '/' and y != 0 and x % y == 0:
                        return x // y
                    return None

                def count_solution_paths(nums, target, max_evaluations=1000):
                    try:
                        if not nums:
                            return 0
                        count = 0
                        eval_count = 0
                        ops = ['+', '-', '*', '/']
                        for num_perm in permutations(nums):
                            for op_set in product(ops, repeat=len(nums) - 1):
                                eval_count += 1
                                if eval_count >= max_evaluations:
                                    return count
                                result = num_perm[0]
                                for i, op in enumerate(op_set):
                                    result = apply_operator(result, num_perm[i+1], op)
                                    if result is None:
                                        break
                                    if result == target:
                                        count += 1
                                        break
                        return count
                    except Exception:
                        return 0

                rewards = []
                for completion in completions:
                    try:
                        comp_with_think = "<think>" + completion
                        answer_dict = extract_answer_dict(comp_with_think)
                        target = answer_dict.get('target')
                        nums = answer_dict.get('nums')
                        if target is None or nums is None:
                            rewards.append(0.0)
                        else:
                            paths = count_solution_paths(nums, target)
                            rewards.append(float(paths))
                    except Exception:
                        rewards.append(0.0)
                return rewards

            # --- KEY QUERY + POLICY LEARNING METRICS ---
            # 1. FORMAT (REWARD) FUNC CHECK
            # Check how often the policy and query branches generate valid completions in the right format where the final answer can be parsed out
            # Compute final answer existence for policy branch:
            policy_final_answer_exists = (
                np.mean([1.0 if out.get("final_answer", "").strip() != "" else 0.0 for out in outputs_dicts])
                if outputs_dicts else 0.0
            )
            self.writer.add_scalar("policy/format_valid_think_answer", policy_final_answer_exists, self.global_step)
            query_final_answer_exists = (
                np.mean([1.0 if out.get("final_answer", "").strip() != "" else 0.0 for out in outputs_query])
                if outputs_query else 0.0
            )
            self.writer.add_scalar("query/format_valid_think_answer", query_final_answer_exists, self.global_step)

            # POLICY AVERAGE ISCORRECT CHECK: Calculate average success rate for the policy branch
            # We assume each output dict has a key "iscorrect" which is either True/1 or False/0.
            policy_success_list = [1.0 if out.get("iscorrect", False) else 0.0 for out in outputs_dicts]
            policy_avg_success = np.mean(policy_success_list) if policy_success_list else 0.0
            self.writer.add_scalar("policy/avg_success", policy_avg_success, self.global_step)
            logger.info(f"Policy avg_success: {policy_avg_success:.4f}")

            # QUERY DIFFICULTY CHECK
            R = self.cfg.n_samples_per_prompt
            valid_query_difficulties = []

            for q_idx, extra in enumerate(query_output_extras):
                if extra.get("valid_prompt", False):
                    start_idx = q_idx * R
                    end_idx = (q_idx + 1) * R
                    query_policy_outputs = outputs_dicts[start_idx:end_idx]
                    policy_successes = [1.0 if out.get("iscorrect", False) else 0.0 for out in query_policy_outputs]
                    if policy_successes:
                        avg_success = np.mean(policy_successes)
                        query_difficulty = 1 - 2 * abs(0.5 - avg_success)
                        valid_query_difficulties.append(query_difficulty)
            if valid_query_difficulties:
                overall_query_difficulty = np.mean(valid_query_difficulties)
            else:
                overall_query_difficulty = 0.0
            self.writer.add_scalar("query/query_difficulty", overall_query_difficulty, self.global_step)
            logger.info(f"Query difficulty (avg over valid queries): {overall_query_difficulty:.4f}")

            # 2. POLICY + QUERY DENSE CORRECTNESS REWARD LEARNING CHECK
            # Check how often moderately difficult queries are given - a good measure of policy learning (some policies are successful, some policies are failures) AND good query learning (some queries more moderately difficult than otehrs)
            # Compute avg_success_rewards as the fraction of query experiences whose
            # info["custom_rewards"] (the per-query average reward) is neither 0.0 nor 1.0.
            valid_query_count = 0
            for exp in query_experiences:
                reward_tensor = exp.info.get("custom_rewards", None)
                if reward_tensor is not None and reward_tensor.numel() > 0:
                    reward_val = reward_tensor.item()
                    if reward_val > 0.0 and reward_val < 1.0:
                        valid_query_count += 1
            if len(query_experiences) > 0:
                query_avg_success_not_edge = valid_query_count / len(query_experiences)
            else:
                query_avg_success_not_edge = 0.0
            self.writer.add_scalar("query/avg_success_moderate_difficulty", query_avg_success_not_edge, self.global_step)

            # Compute query validity: fraction of queries that are valid (assuming query_output_extras has "valid_prompt")
            if query_output_extras:
                query_valid = np.mean([1.0 if d.get("valid_prompt", False) else 0.0 for d in query_output_extras])
            else:
                query_valid = 0.0
            self.writer.add_scalar("query/query_valid_dict", query_valid, self.global_step)

            # Compute query diversity:
            diversity_scores = diversity_reward_func(query_completions)
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
            self.writer.add_scalar("query/diversity", avg_diversity, self.global_step)

            # --------------------
            # Countdown-Specific Metrics:
            # --------------------
            # Countdown-specific target and nums metrics.
            all_targets = [d["target"] for d in query_output_extras if "target" in d and isinstance(d["target"], int)]
            avg_target = np.mean(all_targets) if all_targets else 0.0
            std_target = np.std(all_targets) if all_targets else 0.0
            all_nums_lists = [d["nums"] for d in query_output_extras if "nums" in d]
            nums_lengths = [len(nums) for nums in all_nums_lists if isinstance(nums, list)]
            avg_nums_length = np.mean(nums_lengths) if nums_lengths else 0.0
            std_nums_length = np.std(nums_lengths) if nums_lengths else 0.0
            all_nums_values = [num for nums in all_nums_lists if isinstance(nums, list) for num in nums if -1000 <= num <= 1000]
            avg_nums_value = np.mean(all_nums_values) if all_nums_values else 0.0
            std_nums_value = np.std(all_nums_values) if all_nums_values else 0.0

            self.writer.add_scalar("query/targets_avg", avg_target, self.global_step)
            self.writer.add_scalar("query/targets_std", std_target, self.global_step)
            self.writer.add_scalar("query/nums_length_avg", avg_nums_length, self.global_step)
            self.writer.add_scalar("query/nums_length_std", std_nums_length, self.global_step)
            self.writer.add_scalar("query/nums_value_avg", avg_nums_value, self.global_step)
            self.writer.add_scalar("query/nums_value_std", std_nums_value, self.global_step)

            # Compute average number of solution paths for query completions:
            avg_solution_paths_scores = avg_num_solution_paths_reward_func(query_completions)
            avg_num_solution_paths = np.mean(avg_solution_paths_scores) if avg_solution_paths_scores else 0.0
            self.writer.add_scalar("query/avg_num_solution_paths", avg_num_solution_paths, self.global_step)

            # Compute query solvability:
            solvability_scores = solvability_reward_func(query_completions)
            avg_solvability = np.mean(solvability_scores) if solvability_scores else 0.0
            self.writer.add_scalar("query/solvability", avg_solvability, self.global_step)

            self.writer.flush()


            # --- Logging to allow for inspecting query training data and policy training data --- # Purpose: ensure that the completion - reward calculations MAKE SENSE
            logger.debug(f"\n\n[STEP {self.global_step}] [INSPECT QUERY TRAINING DATA AND POLICY TRAINING DATA]")
            # Only inspect the first 4 queries.
            num_inspect = min(4, len(query_texts))
            R = self.cfg.n_samples_per_prompt  # number of policy completions per query
            for idx in range(num_inspect):
                # For the query branch: log the generated query text and the computed reward (last token value)
                query_text = query_texts[idx]
                # Assume query_custom_rewards[idx] is a tensor; we take its last element as the computed reward.
                query_reward_val = (
                    query_custom_rewards[idx][-1].item()
                    if hasattr(query_custom_rewards[idx], "numel") and query_custom_rewards[idx].numel() > 0
                    else "N/A"
                )
                logger.debug(f"\n[STEP {self.global_step}] Query {idx+1} Query Prompt: {str(query_prompts[idx])}")
                logger.debug(f"\n[STEP {self.global_step}] Query {idx+1} Query Text: {str(query_text)}")
                logger.debug(f"[STEP {self.global_step}] Query {idx+1} Query Extras: {str(query_output_extras[idx])}")
                logger.debug(f"[STEP {self.global_step}] Query {idx+1} Query Computed Reward (last token): {query_reward_val}") # Want to be able to search for high-scoring items

                # For the policy branch: each query was repeated R times.
                logger.debug(f"[STEP {self.global_step}] Policy Completions for Query {idx+1}:")
                for j in range(R):
                    policy_index = idx * R + j
                    comp_text = outputs[policy_index] if policy_index < len(outputs) else "N/A"
                    # For the custom reward tensor of the policy completion, take the last token.
                    policy_reward_val = (
                        custom_rewards[policy_index][-1].item()
                        if custom_rewards is not None and policy_index < len(custom_rewards) and custom_rewards[policy_index].numel() > 0
                        else "N/A"
                    )
                    logger.debug(f"  [STEP {self.global_step}] Policy Completion {j+1}: {comp_text}")
                    logger.debug(f"    [STEP {self.global_step}] Policy Generate VLLM Raw Output: {outputs_dicts[policy_index]}")
                    logger.debug(f"    [STEP {self.global_step}] Policy Extras: {all_extras[policy_index]}")
                    logger.debug(f"    [STEP {self.global_step}] Policy Computed Policy Reward (last token): {policy_reward_val}")


            # Update history.extend([(highest scoring query response, corresponding reward), (highest scoring policy response, corresponding reward)])
            # Update history with the best query and corresponding best policy response.
            # Find the highest scoring query response (using the last token reward from query_custom_rewards)
            # Update history with the best query and corresponding best policy response.
            # Find the highest scoring valid query response (using the last token reward from query_custom_rewards)
            best_query_idx = None
            best_query_reward = -float("inf")
            for i, reward_tensor in enumerate(query_custom_rewards):
                # Only consider queries with a valid prompt
                if not query_output_extras[i].get("valid_prompt", False):
                    continue
                if hasattr(reward_tensor, "numel") and reward_tensor.numel() > 0:
                    r_val = reward_tensor[-1].item()
                else:
                    r_val = -float("inf")
                if r_val > best_query_reward:
                    best_query_reward = r_val
                    best_query_idx = i

            # Only update history if a valid query is found.
            if best_query_idx is not None:
                # Get the corresponding query text and ensure it starts with <think>
                best_query_response = str(query_texts[best_query_idx]['response'])
                if not best_query_response.startswith("<think>"):
                    best_query_response = "<think>" + best_query_response

                # Now, for that query, choose the highest scoring policy response.
                # Each query has R = self.cfg.n_samples_per_prompt policy completions.
                R = self.cfg.n_samples_per_prompt
                start_idx = best_query_idx * R
                end_idx = start_idx + R
                best_policy_reward = -float("inf")
                best_policy_response = None
                for idx_policy in range(start_idx, min(end_idx, len(custom_rewards))):
                    if hasattr(custom_rewards[idx_policy], "numel") and custom_rewards[idx_policy].numel() > 0:
                        p_reward = custom_rewards[idx_policy][-1].item()
                    else:
                        p_reward = -float("inf")
                    if p_reward > best_policy_reward:
                        best_policy_reward = p_reward
                        best_policy_response = outputs[idx_policy]
                if best_policy_response is not None and not best_policy_response.startswith("<think>"):
                    best_policy_response = "<think>" + best_policy_response

                # Append both the best valid query and its best policy response to history.
                # Each element is a tuple: (response_text, reward)
                self.history.extend([
                    (best_query_response, best_query_reward),
                    (best_policy_response, best_policy_reward)
                ])

                logger.info(f"\n\n[STEP {self.global_step}] [HISTORY UPDATE] Best Query Reward: {best_query_reward}\n\nQuery Response: {best_query_response}")
                logger.info(f"\n\n[STEP {self.global_step}] [HISTORY UPDATE] Best Policy Reward: {best_policy_reward}\n\nPolicy Response: {best_policy_response}")


    @torch.no_grad()
    async def inference_and_calculates(
        self,
        sequences_all: List[torch.Tensor],
        attention_mask_all: List[torch.Tensor],
        action_mask_all: Optional[List[torch.Tensor]],
        num_actions_all: Optional[List[int]],
        packed_seq_lens_all: Optional[List[int]],
        custom_rewards_all: Optional[List[torch.Tensor]],
        exp_type: Optional[str] = "policy",
    ):
        num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
        num_critic_dp_groups = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
        num_ref_dp_groups = self.cfg.ref_num_nodes * self.cfg.ref_num_gpus_per_node
        num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node

        async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
            dp_iterator = self._split_dp_batch(
                (sequences, num_actions, attention_mask, packed_seq_lens),
                num_dps,
            )
            dp_tasks = []
            for dp_rank, (
                micro_sequences,
                micro_num_actions,
                micro_attention_mask,
                micro_packed_seq_lens,
            ) in enumerate(dp_iterator):
                model = self._get_dp_group_models(dp_rank, model_type)

                async def forward_fn(
                    local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                ):
                    return await local_model.forward.remote(
                        sequences=fwd_sequences,
                        num_actions=fwd_num_actions,
                        attention_mask=fwd_attention_mask,
                        packed_seq_lens=fwd_packed_seq_lens,
                    )

                dp_tasks.append(
                    self._split_and_run_micro_batch(
                        partial(forward_fn, model),
                        (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                        self.cfg.micro_forward_batch_size,
                    )
                )
            results = await asyncio.gather(*dp_tasks)
            results = sum(results, [])
            return results

        if action_mask_all is not None:
            num_actions_all = action_mask_all.size(1)

        # calculate critic values
        if self.cfg.colocate_all and self.critic_model is not None:
            await self.critic_model.backload_to_gpu()

        if self.critic_model is not None:
            value_ref = micro_infer_model(
                num_critic_dp_groups,
                "critic_model",
                sequences_all,
                num_actions_all,
                attention_mask_all,
                packed_seq_lens_all,
            )
            values = None
            if self.cfg.colocate_all:
                values = await value_ref
                await self.critic_model.offload_to_cpu()

        # calculate ref log probs
        base_action_log_probs_ref = micro_infer_model(
            num_ref_dp_groups, "ref_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
        )
        base_log_probs = None

        # handle colocate critic and reward model
        if self.cfg.colocate_critic_reward and not self.cfg.colocate_all and self.critic_model is not None:
            values = await value_ref
            await self.critic_model.async_run_method("empty_cache")

        # handle colocate actor and ref model
        if self.cfg.colocate_actor_ref or self.cfg.colocate_all:
            base_log_probs = await base_action_log_probs_ref
            await self.ref_model.async_run_method("empty_cache")

        # calculate rewards
        reward_refs = []
        if self.cfg.use_orm_score and self.reward_model:
            reward_refs.append(
                micro_infer_model(
                    num_reward_dp_groups,
                    "reward_model",
                    sequences_all,
                    num_actions_all,
                    attention_mask_all,
                    packed_seq_lens_all,
                )
            )

        if self.cfg.colocate_all:
            rewards = await asyncio.gather(*reward_refs)

        # calculate action log probs
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()

        action_log_probs_ref = micro_infer_model(
            num_policy_dp_groups,
            "policy_model",
            sequences_all,
            num_actions_all,
            attention_mask_all,
            packed_seq_lens_all,
        )
        action_log_probs = None
        if self.cfg.colocate_all:
            action_log_probs = await action_log_probs_ref
            await self.policy_model.offload_to_cpu()

        # wait all models done
        # if not colocate_actor_ref, then need to gather base_log_probs
        # if not colocate_critic_reward and self.critic_model is not None, then need to gather value
        # reward_refs is always handled at last
        if not self.cfg.colocate_all:
            if not self.cfg.colocate_actor_ref:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(
                        value_ref, base_action_log_probs_ref, action_log_probs_ref, *reward_refs
                    )
                    values, base_log_probs, action_log_probs, rewards = results[0], results[1], results[2], results[3:]
                else:
                    results = await asyncio.gather(base_action_log_probs_ref, action_log_probs_ref, *reward_refs)
                    base_log_probs, action_log_probs, rewards = results[0], results[1], results[2:]
            else:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(value_ref, action_log_probs_ref, *reward_refs)
                    values, action_log_probs, rewards = results[0], results[1], results[2:]
                else:
                    results = await asyncio.gather(action_log_probs_ref, *reward_refs)
                    action_log_probs, rewards = results[0], results[1:]

        r = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else None
        if not self.cfg.colocate_all:
            empty_cache_tasks = [
                self.policy_model.async_run_method("empty_cache"),
                self.ref_model.async_run_method("empty_cache"),
            ]
            if self.critic_model:
                empty_cache_tasks.append(self.critic_model.async_run_method("empty_cache"))
            if self.reward_model:
                empty_cache_tasks.extend([rm.async_run_method("empty_cache") for rm in self.reward_model])
            await asyncio.gather(*empty_cache_tasks)

        # 6. calculate kl divergence

        experiences = []
        if self.critic_model is not None:
            values = values[: len(sequences_all)]
        base_log_probs = base_log_probs[: len(sequences_all)]
        action_log_probs = action_log_probs[: len(sequences_all)]
        if r is not None:
            r = r[: len(sequences_all)]
        for i in range(len(action_log_probs)):
            response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
            total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
            kl = compute_approx_kl(
                action_log_probs[i],
                base_log_probs[i],
                action_mask=None,
                use_kl_estimator_k3=self.cfg.use_kl_estimator_k3,
                use_abs_kl=self.cfg.use_abs_kl,
            )
            kl_max = torch.max(kl.abs(), dim=-1)[0]
            kl_mean = masked_mean(kl, None, dim=-1)
            if r is not None:
                local_reward = r[i]
            else:
                local_reward = None
            info = {
                "kl": kl_mean,
                "kl_max": kl_max,
                "reward": local_reward,
                "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
                "response_length": response_length,
                "total_length": total_length,
                "num_actions": num_actions_all[i],
                "exp_type": torch.tensor([1.0]) if exp_type == "policy" else torch.tensor([0.0]), # must be tensors
            }
            experiences.append(
                Experience(
                    sequences_all[i],
                    action_log_probs[i],
                    base_log_probs[i],
                    values[i] if self.critic_model is not None else None,
                    None,
                    None,
                    attention_mask_all[i],
                    None,
                    response_length,
                    torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),
                    info,
                    kl,
                )
            )
        return experiences

    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        responses, _ = await gen_func(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        return responses

    def build_dataloader(self, dataset):
        # prepare dataloader
        prompts_dataloader = DataLoader(
            dataset, batch_size=self.cfg.rollout_batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8
        )
        self.num_update_steps_per_episodes = (
            len(dataset) * self.cfg.n_samples_per_prompt // self.cfg.train_batch_size * self.cfg.max_epochs
        )
        max_steps = math.ceil(self.cfg.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps
        return prompts_dataloader

    async def build_models(self, PolicyRayActor, CriticRayActor, RefRayActor, RewardRayActor=None):
        cfg = self.cfg
        pg = None

        if cfg.colocate_all:
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.2,
                        )
                    )
            else:
                reward_models = None

        else:
            if cfg.colocate_actor_ref:
                assert (
                    cfg.actor_num_nodes == cfg.ref_num_nodes
                    and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [
                    {"GPU": cfg.actor_num_gpus_per_node, "CPU": cfg.actor_num_gpus_per_node}
                    for _ in range(cfg.actor_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )

            # if colocated, create placement group for critic and reward model explicitly.
            pg = None
            if cfg.colocate_critic_reward:
                assert (
                    cfg.critic_num_nodes == cfg.reward_num_nodes
                    and cfg.critic_num_gpus_per_node == cfg.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

                bundles = [
                    {"GPU": cfg.critic_num_gpus_per_node, "CPU": cfg.critic_num_gpus_per_node}
                    for _ in range(cfg.critic_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.25 if pg else 1,
                        )
                    )
            else:
                reward_models = None

        if not cfg.colocate_all:
            refs = []
            refs.extend(ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            if cfg.critic_pretrain:
                refs.extend(critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    refs.extend(reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))
            await asyncio.gather(*refs)
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
        else:
            await asyncio.gather(*ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
            await policy_model.offload_to_cpu()
            if cfg.critic_pretrain:
                await asyncio.gather(*critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
                await critic_model.offload_to_cpu()
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    await asyncio.gather(*reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))

        self.policy_model = policy_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/ref/critic/reward models done")

    async def ppo_local_train_policy(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer("Policy model training"):
                status = await self.policy_model.async_ppo_train(global_steps, replay_buffers)
            self.writer.add_scalar("ppo_clip_count", status[0]["clip_ratio"], global_steps)
            self.writer.add_scalar("policy_update_steps", status[0]["policy_update_steps"], global_steps)
            self.writer.add_scalar("policy_entropy", status[0]["entropy"], global_steps)
            await self.policy_model.async_run_method("empty_cache")
        if self.cfg.colocate_all:
            async with Timer("Backload vllm engines to gpu"):
                await self._backload_vllm_engines()
            async with Timer("Broadcast actor weights to vllm engines"):
                await self._sync_policy_weights_to_vllm()

        if global_steps > self.cfg.freezing_actor_steps:
            return status[0]

    async def ppo_local_train_critic(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        async with Timer("Critic model training"):
            status = await self.critic_model.async_ppo_train(global_steps, replay_buffers)
        if critic_loss := status[0].get("critic_loss", None):
            self.writer.add_scalar("critic_loss", critic_loss, global_steps)
            self.writer.add_scalar("critic_update_steps", status[0]["critic_update_steps"], global_steps)
        return status[0]

    async def custom_query_reward_fn(
        self,
        query_prompts: List[str],
        query_outputs: List[Any],
        query_output_extras: List[dict],
        policy_custom_rewards: List[torch.Tensor],
        R: int,  # Number of policy completions per query
    ) -> List[torch.Tensor]:
        raise NotImplementedError("custom query reward function is not supported yet")

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        raise NotImplementedError("custom reward function is not supported yet")

    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            experience.action_mask,
            num_actions,
            self.cfg.gamma,
            self.cfg.lambd,
            packing=True,
        )
        return_sums = reward.sum(dim=-1)
        return_sums /= len(num_actions)
        experience.info["return"] = return_sums
        experience.kl = None

        avg_rewards = return_sums.mean().item()
        avg_kl = experience.info["kl"].mean().item()
        avg_kl_max = experience.info["kl_max"].mean().item()

        avg_response_length = experience.info["response_length"].mean().item()
        if experience.info["reward"] is not None:
            avg_orm_score = experience.info["reward"].mean().item()
        else:
            avg_orm_score = 0

        if experience.info["custom_rewards"] is not None:

            def func(x):
                return [r.sum() for r in x]

            avg_custom_rewards = torch.stack(func(experience.info["custom_rewards"])).mean().item()
            # experience.info["avg_custom_rewards"] = torch.stack(func(experience.info["custom_rewards"]))
        else:
            avg_custom_rewards = 0

        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        del experience.info["kl_max"]
        experience.to_device("cpu")

        # for replay buffer split batch
        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        metrics = {
            "avg_rewards": avg_rewards,
            "avg_kl": avg_kl,
            "avg_kl_max": avg_kl_max,
            "avg_response_length": avg_response_length,
            "avg_orm_score": avg_orm_score,
            "avg_custom_rewards": avg_custom_rewards,
            "avg_advantages": experience.advantages.mean().item(),
            "avg_advantages_abs": experience.advantages.abs().mean().item(),
        }

        return experience, metrics

    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        # This function is used when not packing samples
        # concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        prompt_token_lens, response_token_lens = [], []
        inputs_token_ids, outputs_token_ids = [], []
        for prompt, output in zip(prompts, outputs):
            input_token_ids = self._tokenize(prompt, self.cfg.prompt_max_len, padding=False)["input_ids"]
            response_token_ids = self._tokenize(output, self.cfg.generate_max_len, padding=False)["input_ids"]

            inputs_token_ids.append(input_token_ids)
            outputs_token_ids.append(response_token_ids)

            prompt_token_len = len(input_token_ids)
            response_token_len = len(response_token_ids)
            prompt_token_lens.append(prompt_token_len)
            response_token_lens.append(response_token_len)

            max_input_len = max(max_input_len, prompt_token_len)
            max_output_len = max(max_output_len, response_token_len)

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for i, prompt in enumerate(prompts):
            # left padding input
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # right padding output
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # replace last token with eos_token_id if it is not eos_token_id, keep the total length of output_ids
            # output_ids[output_len - 1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences, attention_mask, action_mask

    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]], packing_max_len: int
    ):
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            seq_offset = 0
            seq_index = 0
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
            )

        def _accumulate(
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
            sequence,
            attention_mask,
            num_action,
            total_len,
            custom_rewards,
            i,
        ):
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            return seq_offset + total_len, seq_index + 1

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # make packed sequences
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens)
        ):
            if seq_offset + total_len < packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
            elif seq_offset + total_len == packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
            elif seq_offset + total_len > packing_max_len:
                if seq_offset > 0:
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                    ) = _new_instance()
                    seq_offset, seq_index = _accumulate(
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                        sequence,
                        attention_mask,
                        num_action,
                        total_len,
                        custom_rewards,
                        i,
                    )

        if seq_offset > 0:
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)

        return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards

    def _get_dp_group_models(self, dp_rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[dp_rank]

    def _split_dp_batch(self, batch, num_dp, drop_last=False):
        # Convert batch tuple to list of lists, handling None values
        batch_lists = []
        batch_size = None
        for item in batch:
            if item is not None:
                if batch_size is None:
                    batch_size = len(item)
                batch_lists.append(item)
            else:
                batch_lists.append(None)

        if drop_last:
            dp_size = batch_size // num_dp
        else:
            dp_size = (batch_size + num_dp - 1) // num_dp
        valid_size = dp_size * num_dp

        if not drop_last:
            padding_index = None
            for i in range(len(batch_lists)):
                if batch_lists[i] is not None and (
                    isinstance(batch_lists[i], torch.Tensor) or isinstance(batch_lists[i], list)
                ):
                    padding_size = valid_size - len(batch_lists[i])
                    if padding_size > 0:
                        if padding_index is None:
                            if padding_size > len(batch_lists[i]):
                                padding_index = random.choices(range(len(batch_lists[i])), k=padding_size)
                            else:
                                padding_index = random.sample(range(len(batch_lists[i])), padding_size)
                        if isinstance(batch_lists[i], torch.Tensor):
                            batch_lists[i] = torch.cat([batch_lists[i], batch_lists[i][padding_index]], dim=0)
                        elif isinstance(batch_lists[i], list):
                            batch_lists[i] = batch_lists[i] + [batch_lists[i][j] for j in padding_index]

        for i in range(num_dp):
            # Extract micro batch for each input list
            micro_batch = []
            for batch_list in batch_lists:
                if batch_list is None:
                    micro_batch.append(None)
                elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                    micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
                else:
                    micro_batch.append(batch_list)
            yield tuple(micro_batch)

    def _split_dp_batch_dynamic_balance(self, batch, num_dp, balanced_values):
        batch = list(batch)
        assert len(batch) == len(balanced_values), "batch and balanced_values must have the same length"
        results = self._split_weighted_objects(zip(balanced_values, batch), num_dp)
        # re organize to the original format
        for i in range(num_dp):
            ret = [[] for _ in range(len(results[i][0]))]
            for sample in results[i]:
                for j, v in enumerate(sample):
                    ret[j].append(v)
            yield ret

    def _split_weighted_objects(self, items, n):
        result = [[] for _ in range(n)]

        heap = [(0, i) for i in range(n)]
        heapify(heap)

        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        for weight, obj in sorted_items:
            current_sum, index = heappop(heap)
            result[index].append(obj)
            heappush(heap, (current_sum + weight, index))

        return result

    async def _split_and_run_micro_batch(self, async_fn, batch_args, micro_size):
        # Ensure batch_args is a sequence of lists with equal length
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches
        for i in range(0, batch_size, micro_size):
            # Take slice i:i+micro_size from each argument
            micro_batch_args = []
            for arg in batch_args:
                if arg is not None:
                    if not isinstance(arg, torch.Tensor) and not isinstance(arg, list):
                        micro_batch_args.append(arg)
                    elif micro_size > 1 or isinstance(arg, torch.Tensor):
                        micro_batch_args.append(arg[i : i + micro_size])
                    else:
                        micro_batch_args.append(arg[i])
                else:
                    micro_batch_args.append(None)
            results.append(await async_fn(*micro_batch_args))
        return results

    def _get_generate_function(self, dp_rank: int):
        llm = self.vllm_engines[dp_rank % len(self.vllm_engines)]

        async def generate(prompts: List[str], truncate_prompt=True, **kwargs):
            if truncate_prompt:
                prompt_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
            else:
                prompt_token_ids = self._tokenize(prompts, padding=False)["input_ids"]
            outputs = await llm.generate.remote(prompt_token_ids=prompt_token_ids, **kwargs)
            responses = []
            prompt_logprobs = []
            finish_reasons = []
            for i, prompt in enumerate(prompts):
                content = outputs[i].outputs[0].text
                finish_reasons.append(outputs[i].outputs[0].finish_reason)
                responses.append(content)
                if outputs[i].prompt_logprobs:
                    prompt_logprobs.append(outputs[i].prompt_logprobs)
            if len(prompt_logprobs) > 0:
                return (
                    responses,
                    finish_reasons,
                    prompt_logprobs,
                )
            else:
                return responses, finish_reasons

        return generate

    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _warp_custom_reward_model_fn(self):
        if self.reward_model:
            # TODO: support multiple reward models]
            num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node

            async def warpped_reward_model_fn(prompts: List[str], outputs: List[str]):
                (
                    sequences,
                    attention_mask,
                    _,
                    packed_seq_lens,
                    _,
                ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                    prompts, outputs, None, self.cfg.packing_max_len
                )
                split_iterator = self._split_dp_batch(
                    (sequences, attention_mask, packed_seq_lens), num_policy_dp_groups
                )
                dp_tasks = []

                async def _rm_run(rm, seq, mask, lens):
                    return await rm.forward.remote(seq, mask, packed_seq_lens=lens)

                for dp_rank, args in enumerate(split_iterator):
                    rm = self._get_dp_group_models(dp_rank, "reward_model")
                    dp_tasks.append(
                        self._split_and_run_micro_batch(
                            partial(_rm_run, rm),
                            args,
                            self.cfg.micro_forward_batch_size,
                        )
                    )
                outputs = await asyncio.gather(*dp_tasks)
                outputs = sum(outputs, [])  # gather dp
                outputs = outputs[: len(sequences)]  # drop padding
                outputs = torch.hstack(outputs)

                assert outputs.size(0) == len(prompts), "reward outputs number must be equal to prompts number"
                return outputs

            return warpped_reward_model_fn
        else:
            return None

    async def _offload_vllm_engines(self):
        offload_tasks = []
        for engine in self.vllm_engines:
            offload_tasks.append(engine.offload_to_cpu.remote())
        await asyncio.gather(*offload_tasks)

    async def _backload_vllm_engines(self):
        backload_tasks = []
        for engine in self.vllm_engines:
            backload_tasks.append(engine.backload_to_gpu.remote())
        await asyncio.gather(*backload_tasks)

    async def _sync_policy_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.policy_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)
