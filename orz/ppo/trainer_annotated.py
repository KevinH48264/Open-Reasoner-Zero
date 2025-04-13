# Import standard libraries for async operations, JSON, math, OS operations, and random sampling.
import asyncio
import json
import math
import os
import random
from functools import partial  # Used for partial function application.
from heapq import heapify, heappop, heappush  # For maintaining heaps (priority queues).
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

# Import Ray for distributed computing.
import ray
# Import PyTorch for tensor computations.
import torch
# Import logger for logging.
from loguru import logger
# Import DictConfig for configuration dictionaries.
from omegaconf.dictconfig import DictConfig
# Import placement group utilities from Ray.
from ray.util.placement_group import PlacementGroup, placement_group
# Import DataLoader from PyTorch for batching datasets.
from torch.utils.data import DataLoader
# Import SummaryWriter to log TensorBoard metrics.
from torch.utils.tensorboard import SummaryWriter
# Import tqdm for progress bars.
from tqdm import tqdm

# Import custom modules for PPO actors, replay buffer, and various utilities.
from orz.ppo.actors import PPORayActorGroup
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from orz.ppo.utils import (
    Timer,  # Utility for timing code blocks.
    compute_approx_kl,  # Function to compute approximate KL divergence.
    compute_reward,  # Function to compute rewards.
    get_advantages_and_returns,  # Function to compute advantages and returns.
    masked_mean,  # Computes mean with a mask.
    normalize_advantages,  # Normalizes advantages in the replay buffer.
)

# Define the main trainer class for PPO using Ray.
class RayPPOTrainer:
    # Initialization of the trainer.
    def __init__(
        self,
        cfg: DictConfig,              # Configuration parameters.
        strategy: DeepspeedStrategy,  # Strategy for deepspeed optimization.
        tokenizer,                    # Tokenizer to process text.
        train_dataset,                # Training dataset.
        eval_dataset=None,            # Optional evaluation dataset.
        vllm_engines=None,            # Optional vLLM engines for generation.
        colocate_pg: Optional[PlacementGroup] = None,  # Optional placement group for resource colocation.
    ):
        self.cfg = cfg                         # Store configuration.
        self.strategy = strategy               # Store deepspeed strategy.
        self.tokenizer = tokenizer             # Store tokenizer.
        self.train_dataset = train_dataset     # Store training dataset.
        self.eval_dataset = eval_dataset       # Store evaluation dataset.
        self.vllm_engines = vllm_engines       # Store vLLM engines.
        # Build the dataloader for training prompts.
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg         # Placement group for colocating resources.

        # Create a TensorBoard writer for logging metrics.
        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)
        # Initialize a replay buffer to store experiences for PPO training.
        self.replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,               # No limit on stored samples.
            cpu_offload=True,      # Offload to CPU memory if needed.
            packing_samples=True,  # Enable packing samples to maximize utilization.
        )

    # Destructor to ensure the TensorBoard writer is closed.
    def __del__(self):
        self.writer.close()

    # Evaluation function (must be implemented in the experiment subclass).
    async def eval(self):
        raise NotImplementedError("Eval function should be implemented in user's exp")

    # Main training loop.
    async def train(self):
        # 1. Initialize the policy model and vLLM engine groups, then broadcast weights to vLLM engines.
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()  # Move policy model weights to GPU.
            await self._backload_vllm_engines()           # Move vLLM engines back to GPU.

        # Initialize the vLLM actor group for the policy model.
        await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)
        logger.info("Create vllm engine gourps done.")

        # Synchronize policy weights to vLLM engines.
        async with Timer("Sync actor weights to vllm engines"):
            await self._sync_policy_weights_to_vllm()

        # If models are colocated, offload the policy model back to CPU to save GPU memory.
        if self.cfg.colocate_all:
            async with Timer("Offload policy model to cpu"):
                await self.policy_model.offload_to_cpu()

        # 2. Main training loop: iterate over episodes.
        consumed_samples = 0
        # Compute how many rollouts are performed per episode.
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
        # Loop over episodes.
        for episode in range(start_episode, self.cfg.num_episodes):
            # Create a progress bar over dataloader iterations.
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()), desc=f"Episode [{episode + 1}/{self.cfg.num_episodes}]"
            )
            # Iterate over minibatches of prompts.
            for iter, rand_prompts in enumerate(self.prompts_dataloader):

                # 1. Evaluate if evaluation is enabled and if conditions are met.
                if self.cfg.enable_eval and (
                    self.global_step % self.cfg.eval_interval == 0 or iter == len(self.prompts_dataloader) - 1
                ):
                    await self.eval()

                # 3. Generate experiences by running the model, then compute advantages and returns.
                await self.make_experience(rand_prompts)

                # If the replay buffer is empty, re-synchronize weights and continue.
                if len(self.replay_buffer) <= 0:
                    if self.cfg.colocate_all:
                        await self.policy_model.backload_to_gpu()
                        await self._backload_vllm_engines()
                        await self._sync_policy_weights_to_vllm()
                        await self.policy_model.offload_to_cpu()
                    continue

                # Normalize advantages in the replay buffer if enabled.
                if self.cfg.advantage_normalize:
                    self.replay_buffer = normalize_advantages(self.replay_buffer)

                # Serialize and dump the replay buffer to a JSONL file for debugging or logging.
                async with Timer("Dumping replay buffer"):
                    all_replay_buffer_save_path = os.path.join(self.cfg.save_path, "dumped_replay_buffer")
                    os.makedirs(all_replay_buffer_save_path, exist_ok=True)
                    dump_path = os.path.join(all_replay_buffer_save_path, f"iter{self.global_step}_replay_buffer.jsonl")
                    with open(dump_path, "a") as f:
                        logger.info(f"dumping replay buffer to {dump_path}")
                        for item in self.replay_buffer:
                            f.write(json.dumps(item.to_json()) + "\n")

                # Split replay buffer into batches for policy and critic training based on device parallelism.
                num_policy_dp_nodes = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
                num_critic_dp_nodes = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
                policy_buffers = self.replay_buffer.split_to_n_batches(num_policy_dp_nodes)
                if num_policy_dp_nodes != num_critic_dp_nodes:
                    critic_buffers = self.replay_buffer.split_to_n_batches(num_critic_dp_nodes)
                else:
                    critic_buffers = policy_buffers

                # 4. Train the policy (actor) and critic models.
                if self.cfg.colocate_all:
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
                    # If not colocated, train actor and critic concurrently.
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

                # Clear the replay buffer after training.
                self.replay_buffer.clear()

                # 5. Log training status and update global step counters.
                logger.info(status)
                pbar.update()
                self.writer.add_scalar("episode_idx", episode, self.global_step)
                self.global_step += 1
                # Save model checkpoints at regular intervals.
                if self.global_step % self.cfg.save_interval == 0:
                    await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                    if self.critic_model is not None:
                        await self.critic_model.async_save_model(self.tokenizer, self.global_step)
                    logger.info("Successfully save model weights, training continue.")

            # After each episode, update the reference model if configured.
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

        # Final save after completing all episodes.
        await self.policy_model.async_save_model(self.tokenizer, self.cfg.num_episodes * len(self.prompts_dataloader))
        logger.info("Successfully save model weights, training done.")

    # Function to generate experiences by sampling from the model and computing rewards/advantages.
    @torch.no_grad()
    async def make_experience(self, all_inputs: Union[Tuple[str, dict], List[Tuple[str, dict]]], **generate_kwargs):
        experiences = []
        # Duplicate each prompt n_samples_per_prompt times.
        all_prompts = sum([[prompt[0]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])
        all_extras = sum([[prompt[1]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])
        # Shuffle prompts and corresponding extras with a fixed seed.
        indices = list(range(len(all_prompts)))
        rng = random.Random(42)
        rng.shuffle(indices)
        all_prompts = [all_prompts[i] for i in indices]
        all_extras = [all_extras[i] for i in indices]

        # 1. Generate sequences via vLLM engines.
        outputs = []
        num_vllm_dp_gruops = len(self.vllm_engines)

        async with Timer("Generate sequences via vllm engines"):
            # Split prompts across vLLM engines.
            dp_prompt_size = (len(all_prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops
            dp_tasks = []
            for dp_rank in range(num_vllm_dp_gruops):
                dp_inputs = all_prompts[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                dp_extras = all_extras[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                # Skip if no prompts for this engine.
                if len(dp_inputs) <= 0:
                    continue
                # Get the generation function for this group.
                gen_func = self._get_generate_function(dp_rank)
                dp_tasks.append(self.generate_vllm(gen_func, dp_inputs, extras=dp_extras, **generate_kwargs))

            logger.info("start generation")
            local_responses = await asyncio.gather(*dp_tasks)
            # Flatten responses from all groups.
            outputs.extend(sum(local_responses, []))
            logger.info("generate local rollout batch done")

            # If models are colocated, offload vLLM engines to CPU.
            if self.cfg.colocate_all:
                async with Timer("Offload vllm engines to cpu"):
                    await self._offload_vllm_engines()

        # Skip if no outputs were generated.
        if len(outputs) <= 0:
            return

        # Ensure that the number of prompts matches the number of outputs.
        assert len(all_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

        # 1.2 Optionally calculate custom rewards using a provided custom reward function.
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

        # If there is no data, exit.
        if len(all_prompts) == 0:
            return

        # 1.3 Pack the samples into batch tensors.
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

        # 1.4 Perform inference and calculate values, log probabilities, rewards, and KL divergence.
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence"):
            experiences = await self.inference_and_calculates(
                ret_sequences,
                ret_attention_masks,
                action_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            )
            logger.info(f"experiences size: {len(experiences)}")

        # 2. Visualize one example of generated sequences and log to TensorBoard.
        vis = self._detokenize(experiences[0].sequences[0][: int(experiences[0].info["total_length"].flatten()[0])])
        self.writer.add_text("generated_sequences", vis, self.global_step)
        self.writer.flush()

        # 3. Compute advantages and returns for each experience, accumulate metrics.
        avg_rewards = 0
        avg_kl = 0
        avg_kl_max = 0
        avg_response_length = 0
        avg_orm_score = 0
        avg_custom_rewards = 0
        avg_advantages = 0
        avg_advantages_abs = 0

        async with Timer("Calculate advantages and returns"):
            adv_tasks = []
            for experience in experiences:
                adv_tasks.append(self._calc_advantages_and_returns(experience))

            # As each advantage computation completes, aggregate the metrics and append experience to the replay buffer.
            for tsk in asyncio.as_completed(adv_tasks):
                experience, metrics = await tsk
                avg_rewards += metrics["avg_rewards"]
                avg_kl += metrics["avg_kl"]
                avg_kl_max += metrics["avg_kl_max"]
                avg_response_length += metrics["avg_response_length"]
                avg_orm_score += metrics["avg_orm_score"]
                avg_custom_rewards += metrics["avg_custom_rewards"]
                avg_advantages += metrics["avg_advantages"]
                avg_advantages_abs += metrics["avg_advantages_abs"]
                self.replay_buffer.append(experience)

        # 4. Log averaged metrics to TensorBoard.
        logger.info(
            f"avg_raw_rewards: {avg_rewards / len(experiences)}, avg_kl: {avg_kl / len(experiences)}, avg_response_length: {avg_response_length / len(experiences)}, avg_orm_score: {avg_orm_score / len(experiences)}, avg_custom_rewards: {avg_custom_rewards / len(experiences)}"
        )
        self.writer.add_scalar("avg_raw_rewards", avg_rewards / len(experiences), self.global_step)
        self.writer.add_scalar("avg_kl", avg_kl / len(experiences), self.global_step)
        self.writer.add_scalar("avg_kl_max", avg_kl_max / len(experiences), self.global_step)
        self.writer.add_scalar("avg_response_length", avg_response_length / len(experiences), self.global_step)
        self.writer.add_scalar("avg_orm_score", avg_orm_score / len(experiences), self.global_step)
        self.writer.add_scalar("avg_custom_rewards", avg_custom_rewards / len(experiences), self.global_step)
        self.writer.add_scalar("avg_raw_advantages", avg_advantages / len(experiences), self.global_step)
        self.writer.add_scalar("avg_raw_advantages_abs", avg_advantages_abs / len(experiences), self.global_step)
        self.writer.flush()

    # Function to perform inference and calculate values, log probabilities, rewards, and KL divergence.
    @torch.no_grad()
    async def inference_and_calculates(
        self,
        sequences_all: List[torch.Tensor],
        attention_mask_all: List[torch.Tensor],
        action_mask_all: Optional[List[torch.Tensor]],
        num_actions_all: Optional[List[int]],
        packed_seq_lens_all: Optional[List[int]],
        custom_rewards_all: Optional[List[torch.Tensor]],
    ):
        # Compute the number of data-parallel groups for various models.
        num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
        num_critic_dp_groups = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
        num_ref_dp_groups = self.cfg.ref_num_nodes * self.cfg.ref_num_gpus_per_node
        num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node

        # Define an inner async function to perform micro-batch inference for a given model type.
        async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
            # Split the batch for data parallelism.
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
                # Get the corresponding model for this DP group.
                model = self._get_dp_group_models(dp_rank, model_type)

                # Define a forward function for the micro-batch.
                async def forward_fn(
                    local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                ):
                    return await local_model.forward.remote(
                        sequences=fwd_sequences,
                        num_actions=fwd_num_actions,
                        attention_mask=fwd_attention_mask,
                        packed_seq_lens=fwd_packed_seq_lens,
                    )

                # Schedule micro-batch inference using the forward function.
                dp_tasks.append(
                    self._split_and_run_micro_batch(
                        partial(forward_fn, model),
                        (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                        self.cfg.micro_forward_batch_size,
                    )
                )
            # Await all DP group tasks and flatten the results.
            results = await asyncio.gather(*dp_tasks)
            results = sum(results, [])
            return results

        # If action masks are provided, set the number of actions based on their size.
        if action_mask_all is not None:
            num_actions_all = action_mask_all.size(1)

        # Calculate critic values if a critic model exists.
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

        # Calculate log probabilities from the reference model.
        base_action_log_probs_ref = micro_infer_model(
            num_ref_dp_groups, "ref_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
        )
        base_log_probs = None

        # If critic and reward models are not colocated, gather critic values.
        if self.cfg.colocate_critic_reward and not self.cfg.colocate_all and self.critic_model is not None:
            values = await value_ref
            await self.critic_model.async_run_method("empty_cache")

        # If models are colocated or actor/ref are colocated, gather base log probabilities.
        if self.cfg.colocate_actor_ref or self.cfg.colocate_all:
            base_log_probs = await base_action_log_probs_ref
            await self.ref_model.async_run_method("empty_cache")

        # Calculate rewards if using an ORM score and if a reward model is available.
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

        # Calculate action log probabilities from the policy model.
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

        # If not all models are colocated, gather missing results.
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

        # Sum rewards across DP groups if available.
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

        # 6. Calculate KL divergence and package all results into Experience objects.
        experiences = []
        if self.critic_model is not None:
            values = values[: len(sequences_all)]
        base_log_probs = base_log_probs[: len(sequences_all)]
        action_log_probs = action_log_probs[: len(sequences_all)]
        if r is not None:
            r = r[: len(sequences_all)]
        for i in range(len(action_log_probs)):
            # Convert number of actions and total sequence lengths to tensors.
            response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
            total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
            # Compute approximate KL divergence.
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
            # Gather info for the experience.
            info = {
                "kl": kl_mean,
                "kl_max": kl_max,
                "reward": local_reward,
                "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
                "response_length": response_length,
                "total_length": total_length,
                "num_actions": num_actions_all[i],
            }
            # Create an Experience object with all the computed fields.
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

    # Asynchronous function to generate outputs using a provided generation function.
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        # Set up sampling parameters for generation from kwargs (or default values).
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # Generate responses using the provided generation function.
        responses, _ = await gen_func(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        return responses

    # Build a DataLoader for the given dataset.
    def build_dataloader(self, dataset):
        # Create DataLoader with specified batch size, shuffling, and number of workers.
        prompts_dataloader = DataLoader(
            dataset, batch_size=self.cfg.rollout_batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8
        )
        # Calculate number of update steps per episode.
        self.num_update_steps_per_episodes = (
            len(dataset) * self.cfg.n_samples_per_prompt // self.cfg.train_batch_size * self.cfg.max_epochs
        )
        max_steps = math.ceil(self.cfg.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps
        return prompts_dataloader

    # Asynchronously build all necessary models (policy, critic, reference, reward) based on the configuration.
    async def build_models(self, PolicyRayActor, CriticRayActor, RefRayActor, RewardRayActor=None):
        cfg = self.cfg
        pg = None  # Placement group variable.

        if cfg.colocate_all:
            # When all models are colocated, enforce that various resource counts are equal.
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg

            # Build policy, reference, and optionally critic actor groups with specified GPU fraction.
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

            # Build multiple reward models if applicable.
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
            # When models are not fully colocated, set up actor and reference groups.
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

            # Create a placement group for critic and reward models if they are to be colocated.
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

            # Build multiple reward models if specified.
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

        # If models are not colocated, initialize them from pretrained checkpoints.
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
            # If models are colocated, initialize and offload them as needed.
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

        # Assign the created models to instance variables.
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/ref/critic/reward models done")

    # Asynchronously train the policy (actor) model using PPO.
    async def ppo_local_train_policy(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer("Policy model training"):
                status = await self.policy_model.async_ppo_train(global_steps, replay_buffers)
            # Log various metrics from the policy training.
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

    # Asynchronously train the critic model using PPO.
    async def ppo_local_train_critic(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        async with Timer("Critic model training"):
            status = await self.critic_model.async_ppo_train(global_steps, replay_buffers)
        if critic_loss := status[0].get("critic_loss", None):
            self.writer.add_scalar("critic_loss", critic_loss, global_steps)
            self.writer.add_scalar("critic_update_steps", status[0]["critic_update_steps"], global_steps)
        return status[0]

    # Placeholder for custom reward function; to be implemented in a subclass if needed.
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        raise NotImplementedError("custom reward function is not supported yet")

    # Asynchronously calculate advantages and returns for an experience.
    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        # Compute reward via remote function.
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
        # Compute advantages and returns using a remote function.
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

        # Aggregate various metrics.
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
        else:
            avg_custom_rewards = 0

        # Clean up temporary info from the experience.
        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        del experience.info["kl_max"]
        # Move the experience to CPU.
        experience.to_device("cpu")

        # Compute average return over the packed samples.
        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        # Collect metrics to be logged.
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

    # Convert prompts and outputs to batch tensors (non-packing version).
    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        # This function concatenates prompts and outputs into sequences with left-padding for prompts and right-padding for outputs.
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
            # Left-pad the prompt tokens.
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # Right-pad the output tokens.
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # Concatenate prompt and output tokens.
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        # Process sequences to generate attention and action masks.
        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences, attention_mask, action_mask

    # Convert prompts and outputs to batch tensors using a packing strategy.
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

        # Ensure the input lists have equal and nonzero lengths.
        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        # Helper function to initialize a new packing instance.
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

        # Helper function to accumulate a sequence into the current packing instance.
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

        # Tokenize prompts and outputs.
        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # Create packed sequences.
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

    # Helper function to get the data-parallel group model for a given rank and model type.
    def _get_dp_group_models(self, dp_rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[dp_rank]

    # Split a batch into DP groups.
    def _split_dp_batch(self, batch, num_dp, drop_last=False):
        # Convert each element in the batch tuple into a list (or leave as None).
        batch_lists = []
        batch_size = None
        for item in batch:
            if item is not None:
                if batch_size is None:
                    batch_size = len(item)
                batch_lists.append(item)
            else:
                batch_lists.append(None)

        # Calculate the size of each DP group.
        if drop_last:
            dp_size = batch_size // num_dp
        else:
            dp_size = (batch_size + num_dp - 1) // num_dp
        valid_size = dp_size * num_dp

        # If not dropping last, pad the batch lists to have equal sizes.
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

        # Yield slices corresponding to each DP group.
        for i in range(num_dp):
            micro_batch = []
            for batch_list in batch_lists:
                if batch_list is None:
                    micro_batch.append(None)
                elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                    micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
                else:
                    micro_batch.append(batch_list)
            yield tuple(micro_batch)

    # Split a batch into DP groups with dynamic balance based on provided weights.
    def _split_dp_batch_dynamic_balance(self, batch, num_dp, balanced_values):
        batch = list(batch)
        assert len(batch) == len(balanced_values), "batch and balanced_values must have the same length"
        results = self._split_weighted_objects(zip(balanced_values, batch), num_dp)
        # Reorganize the results into the original format.
        for i in range(num_dp):
            ret = [[] for _ in range(len(results[i][0]))]
            for sample in results[i]:
                for j, v in enumerate(sample):
                    ret[j].append(v)
            yield ret

    # Helper function to split weighted objects across groups.
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

    # Asynchronously split a batch into micro batches and run the provided async function.
    async def _split_and_run_micro_batch(self, async_fn, batch_args, micro_size):
        # Ensure all batch arguments are sequences of equal length.
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches.
        for i in range(0, batch_size, micro_size):
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

    # Get the generation function for a given DP rank using the corresponding vLLM engine.
    def _get_generate_function(self, dp_rank: int):
        llm = self.vllm_engines[dp_rank % len(self.vllm_engines)]

        async def generate(prompts: List[str], truncate_prompt=True, **kwargs):
            if truncate_prompt:
                prompt_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
            else:
                prompt_token_ids = self._tokenize(prompts, padding=False)["input_ids"]
            # Call the remote generate function of the vLLM engine.
            outputs = await llm.generate.remote(prompt_token_ids=prompt_token_ids, **kwargs)
            responses = []
            prompt_logprobs = []
            finish_reasons = []
            # Process outputs to extract text responses, finish reasons, and optionally log probabilities.
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

    # Process sequences to generate attention masks and action masks.
    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        # Create an attention mask where tokens are not EOS or PAD.
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find the index of the last non-padding token for each sequence.
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For some models (e.g., Llama3, Qwen2), there may be EOS tokens in the middle; recompute attention mask.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # Compute action mask: actions are the tokens that are not EOS or PAD.
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    # Tokenize texts using the provided tokenizer.
    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        if not padding:
            # Return a list of token ids if no padding is requested.
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

    # Convert token ids back to text.
    def _detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    # Wrap the custom reward model function for use in training.
    def _warp_custom_reward_model_fn(self):
        if self.reward_model:
            # TODO: support multiple reward models.
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
                outputs = sum(outputs, [])  # Gather results from all DP groups.
                outputs = outputs[: len(sequences)]  # Discard any extra padding.
                outputs = torch.hstack(outputs)

                assert outputs.size(0) == len(prompts), "reward outputs number must be equal to prompts number"
                return outputs

            return warpped_reward_model_fn
        else:
            return None

    # Asynchronously offload vLLM engines to CPU.
    async def _offload_vllm_engines(self):
        offload_tasks = []
        for engine in self.vllm_engines:
            offload_tasks.append(engine.offload_to_cpu.remote())
        await asyncio.gather(*offload_tasks)

    # Asynchronously backload vLLM engines to GPU.
    async def _backload_vllm_engines(self):
        backload_tasks = []
        for engine in self.vllm_engines:
            backload_tasks.append(engine.backload_to_gpu.remote())
        await asyncio.gather(*backload_tasks)

    # Synchronize policy weights to the vLLM engines.
    async def _sync_policy_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.policy_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)
