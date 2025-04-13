# Import asyncio for asynchronous programming
import asyncio
# Import copy for deep-copying objects
import copy
# Import json for JSON serialization/deserialization
import json
# Import os for operating system interactions (e.g. file paths, environment variables)
import os
# Import re for regular expression operations
import re
# Import defaultdict from collections for dictionaries with default values
from collections import defaultdict
# Import ThreadPoolExecutor for concurrent thread execution
from concurrent.futures import ThreadPoolExecutor
# Import dataclass decorator for data classes
from dataclasses import dataclass
# Import cached_property to cache property values in classes
from functools import cached_property
# Import islice and zip_longest for iterator slicing and pairing with different lengths
from itertools import islice, zip_longest
# Import type hints for more explicit code annotations
from typing import Any, Awaitable, Callable, List, Optional, Tuple

# Import numpy for numerical operations
import numpy as np
# Import ray for distributed computing and remote execution
import ray
# Import torch for PyTorch deep learning framework
import torch
# Import logger from loguru for logging purposes
from loguru import logger
# Import ListConfig from omegaconf for configuration lists
from omegaconf.listconfig import ListConfig
# Import override decorator from typing_extensions to indicate method overriding
from typing_extensions import override

# Import base experiment classes and configuration for PPO experiments
from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
# Import RayPPOTrainer for PPO training using Ray
from orz.ppo import RayPPOTrainer
# Import utility functions for math utilities, such as checking equality and converting solutions to answers
from orz.ppo.tools.math_utils import is_equal, solution2answer
# Import a utility to check reflection patterns in text
from orz.ppo.utils import check_reflection_pattern
# Import custom dataset classes for the zero setting experiments
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

# Set a global debug flag based on the DEBUG_MODE environment variable.
DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

# Create a file name for saving logs and checkpoints; prefix with "debug_" if in debug mode.
file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
# Create a ThreadPoolExecutor with a maximum of 64 worker threads for concurrent tasks.
executor = ThreadPoolExecutor(max_workers=64)


def repeatness(s: str):
    """
    Computes a measure of 'repeatness' in a string by using suffix arrays and LCP (longest common prefix).
    """

    # Nested helper function: get ranking indices for list elements.
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    # Nested helper function: construct suffix array from string represented as list of integers.
    def suffixArray(s):
        line = ranks(s)  # Get initial ranking for characters.
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        # Iteratively update rankings doubling k each time until the entire string is processed.
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        # Construct suffix array from final ranking.
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    # Nested helper function: compute longest common prefix (LCP) between adjacent suffixes.
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

    # Convert the string to a list of ordinal values.
    arr = [ord(i) for i in s]
    n = len(arr)
    # If string is too short, return 0.
    if n <= 1:
        return 0
    # Get suffix array and ranking.
    c, sa = suffixArray(arr)
    # Compute the sum of LCP values.
    cnt = sum(lcp(arr, sa, c))

    # Normalize the repeatness value.
    return cnt * 2 / (n * (n + 1))


# Define a configuration data class for PPO experiments, extending a base configuration.
@dataclass
class PPOExpConfig(BasePPOExpConfig):
    # Whether to compute the reward function.
    use_compute_reward_fn: bool = True
    # Whether to use ORM (optional reward mechanism) score.
    use_orm_score: bool = False

    # Conditional setting: total number of nodes (fewer in debug mode).
    total_num_nodes: int = 32 if not DEBUG_MODE else 8

    # Resource related settings: number of nodes and GPUs for different roles (ref, actor, critic)
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    # Whether to colocate all components on the same nodes.
    colocate_all: bool = True
    # Whether to colocate critic and reward calculation.
    colocate_critic_reward: bool = True
    # Whether to colocate actor and reference model.
    colocate_actor_ref: bool = True
    # Number of vLLM engines equals the number of total nodes.
    vllm_num_engines: int = total_num_nodes
    # Tensor parallel size for vLLM.
    vllm_tensor_parallel_size: int = 1
    # Whether to offload Adam optimizer states.
    adam_offload: bool = False
    # ZeRO optimization stage.
    zero_stage: int = 3

    # Path related settings: paths for pretraining model, reward pretraining and where to save checkpoints.
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B"  # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/{file_name}"
    save_path: str = f"/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # Data related settings: paths for training and evaluation prompt datasets.
    prompt_data: ListConfig = ListConfig(
        [
            "data/orz_math_57k_collected.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ]
    )
    # Probability weights for sampling from prompt data.
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # PPO-related hyperparameters.
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    # Rollout and episode settings.
    num_episodes: int = 20
    rollout_batch_size: int = 128 if not DEBUG_MODE else 16
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 128

    # Update steps for policy and critic networks.
    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # KL loss settings.
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    # Evaluation settings.
    enable_eval: bool = True
    eval_interval: int = 10

    # Generation settings: maximum lengths and sampling parameters.
    packing_max_len: int = 16384
    generate_max_len: int = 8000  # TODO: change to larger later
    max_len: int = 8192  # TODO: change to larger later
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    # List of stop tokens to signal generation stop.
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # GRPO related settings.
    use_grpo: bool = False

    # GPU memory utilization settings, dependent on GRPO and debug mode.
    gpu_memory_utilization: float = 0.75 if use_grpo else 0.7 if not DEBUG_MODE else 0.5
    # Critic pretraining: if using GRPO, leave blank; otherwise use pretrain model.
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    # Discount factors for reward calculation.
    gamma: float = 1.0
    lambd: float = 1.0


# CustomRewardTrainer extends the RayPPOTrainer to override reward computation and generation logic.
class CustomRewardTrainer(RayPPOTrainer):
    # Override the custom reward function which is asynchronous.
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # Initialize lists for logging various metrics.
        scores = []
        responses = []
        avg_non_stop_count = 0
        # Dictionary to track pass@n for each prompt.
        pass_at_n_dict = defaultdict(list)
        num_tokens: List[int] = []

        # Define a Ray remote function to compute the repeat score.
        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            return repeatness(res)

        # Define a Ray remote function to compute a reflection pattern score.
        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        # Prepare tasks for computing repeat and reflection scores.
        rep_tasks = []
        for output in outputs:
            response = output["response"]
            # Add remote tasks for each response.
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        # Execute the remote tasks and collect results.
        rep_task_results = ray.get(rep_tasks)

        # Separate the results into repeat and reflection pattern scores.
        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        # Collect responses from outputs.
        for output in outputs:
            responses.append(output["response"])
        # Tokenize the responses with maximum generation length.
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        # Log a sample of generated outputs to TensorBoard.
        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )
        # Iterate over each output to compute scores and log metrics.
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            # Record scores in the output dictionary.
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score
            # Only assign a reward of 1.0 if the response stopped properly and is correct.
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
            else:
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)

            # Compute pass@n metric per prompt.
            pass_at_n_dict[prompt].append(scores[-1])
            # Record the number of tokens.
            num_tokens.append(response_token)

        # Convert token counts and scores into numpy arrays for statistics.
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        # If using GRPO, adjust the scores by normalizing them.
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            # For each prompt, subtract the mean and divide by the standard deviation.
            for i, prompt in enumerate(prompts):
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std

        # Helper function to dump the generation results to a JSON file.
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

        # Use the global executor to dump results asynchronously.
        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        # Compute various logging metrics and add them to TensorBoard.
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
        # Create a logging string with formatted metrics.
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # Log histograms of response lengths for correct and incorrect responses.
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # Prepare a score tensor for each response, marking the last token with the score.
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # Remove any outputs with empty responses.
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        # Return the filtered prompts, responses, and score tensors.
        return res_prompts, res_responses, res_score_tensors

    # Override generate_vllm to generate outputs using vLLM and perform post-processing.
    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        # Import SamplingParams from vllm for controlling generation parameters.
        from vllm import SamplingParams

        # Create sampling parameters using configuration settings.
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        # Generate responses and stop reasons using the provided generation function.
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        # Define a remote function to extract the final answer from a batch of responses.
        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            # Use a regex pattern to capture the final answer within <answer> tags.
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            results = []
            for response in responses:
                matches = re.findall(pattern, response)
                results.append(matches[-1] if matches else "")
            return results

        # Set batch size for extraction.
        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        # Create tasks to extract final answers from batches.
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))
        # Run extraction tasks concurrently.
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        # Flatten the list of final answers.
        final_answers = [answer for batch in batched_results for answer in batch]

        # Use the global executor to determine if the final answer matches the expected solution.
        global executor
        equal_tasks = []
        for extra, final_answer in zip(extras, final_answers):
            equal_tasks.append(is_equal(solution2answer(extra["answer"]), solution2answer(final_answer), executor))
        # Await all equality-check tasks.
        equal_results = await asyncio.gather(*equal_tasks)

        # Combine the results into a structured list of dictionaries.
        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, responses, final_answers, stop_reasons, equal_results
        ):
            results.append(
                dict(
                    response=response,
                    iscorrect=iscorrect,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                )
            )

        # Return the list of result dictionaries.
        return results

    # Override the evaluation function.
    @override
    async def eval(self):
        logger.info("Start evaluating on val set")
        from vllm import SamplingParams

        # Set sampling parameters for evaluation generation.
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        # Import DataLoader for creating evaluation batches.
        from torch.utils.data import DataLoader

        # Get the evaluation dataset and create a DataLoader.
        dataset = self.eval_dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        # Calculate how many prompts to assign per vLLM engine.
        prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

        output_for_save = []
        log_dict = defaultdict(float)
        # Loop through the evaluation batches.
        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])
            outputs = []
            # Distribute generation across vLLM engines.
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                    )
                )
            # Await all generation outputs.
            outputs = await asyncio.gather(*outputs)
            # Flatten the list of outputs.
            outputs = sum(outputs, [])

            final_answers = []
            # Regex to extract final answer from generated text.
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            # For each prompt, compare the generated answer to the expected answer.
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, executor)
                # Save the evaluation output for dumping.
                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )
                # Accumulate metrics per file.
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # Get file names (without suffix) from evaluation prompt data.
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
        ]
        # Compute accuracy and average response length for each file.
        for file_name in all_file_names:
            log_dict[f"{file_name}/response_len_in_char"] = (
                log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            )
            log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            # Remove intermediate metrics.
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/correct")
            log_dict.pop(f"{file_name}/total")
        # Calculate the overall evaluation accuracy.
        log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            all_file_names
        )

        # Prepare a filename for dumping evaluation outputs.
        dump_file_name = f"eval_output_iter{self.global_step}"
        # Append accuracy info for each file.
        for file_name in all_file_names:
            dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
        dump_file_name += ".jsonl"
        # Dump the evaluation outputs as a JSONL file.
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

        # Log evaluation metrics.
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)
        for k, v in log_dict.items():
            self.writer.add_scalar(f"evals/{k}", v, self.global_step)


# Define the main PPO experiment class that extends the base PPO experiment.
class PPOExp(BasePPOExp):
    # Cached property to create and return a trainer.
    @cached_property
    def trainer(self):
        # Create inference engines using vLLM.
        vllm_engines = self.create_inference_engine()
        # Instantiate the custom reward trainer with configuration, strategy, tokenizer, datasets, and engines.
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    # Cached property to prepare the training dataset.
    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        # Load dialogues from each file in the prompt_data configuration.
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        # Create a custom dataset from the dialogues.
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    # Cached property to prepare the evaluation dataset.
    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        # Load dialogues from each file in the eval_prompt_data configuration.
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                # Annotate each item with the file name (without extension).
                for loaded_data_item in loaded_data:
                    loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")
        # Create an evaluation custom dataset.
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


# Main entry point for running the experiment.
if __name__ == "__main__":
    # Create an instance of the PPO experiment and set its configuration.
    exp = PPOExp().set_cfg(PPOExpConfig())
    # Log the configuration details.
    logger.info(exp.get_cfg_as_str(exp.cfg))
    # Ensure that the save_path, tensorboard_log_dir, and ckpt_path directories exist.
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    # Run the experiment using asyncio.
    asyncio.run(exp.run())
