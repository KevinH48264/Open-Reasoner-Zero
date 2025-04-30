import multiprocessing


class PromptDataset:
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    # eot_text = EOT
    # bot_text = BOT

    # currently hack for llama 3.1
    start_header_text = "<|start_header_id|>"
    end_header_text = "<|end_header_id|>"
    eot_text = "<|eot_id|>"

    def __init__(
        self,
        dialogues,
        tokenizer: callable,
        max_length: int,
        strategy,
        pretrain_mode: bool = False,
        num_processors: int = 8,
        remove_half_GT_answers_from_train_dataset: bool = False,
        allow_do_not_know: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        # Preprocess dialogues
        if num_processors < 2:
            self.dialogues = [self.process_dialogue(x, remove_half_GT_answers_from_train_dataset, allow_do_not_know) for x in dialogues]
        else:
            pool = multiprocessing.Pool(processes=num_processors)
            self.dialogues = pool.map(self.process_dialogue, dialogues, remove_half_GT_answers_from_train_dataset, allow_do_not_know)
            pool.close()
            pool.join()
        
        # ──────────────────────────────────────────────────────────────
        # Sort by pass_rate_72b_tir (highest first), missing/invalid → 0.0
        def _get_pass_rate(item):
            # item is (prompt, extra)
            extra = item[1]
            raw = extra.get("pass_rate_72b_tir", 0.0)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        # now sort in-place, highest rates first; missing/invalid become 0.0 and end up last
        self.dialogues.sort(key=_get_pass_rate, reverse=True)
        # ──────────────────────────────────────────────────────────────

        # Calculate and print average pass_rate_72b_tir
        pass_rates = [_get_pass_rate(item) for item in self.dialogues]
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
        print(f"[PromptDataset] Average pass_rate_72b_tir after sorting {len(self.dialogues)} items: {avg_pass_rate:.4f}")

        # ──────────────────────────────────────────────────────────────
        # Calculate and print % of entries with no GT answer
        total = len(self.dialogues)
        no_gt_count = sum(
            1 for _, extra in self.dialogues
            if extra.get("answer", "") == "[NO GT ANSWER]"
        )
        pct_no_gt = (no_gt_count / total * 100) if total else 0.0
        print(f"[PromptDataset] % [NO GT ANSWER] entries out of {len(self.dialogues)} items: {pct_no_gt:.2f}%")
        # ──────────────────────────────────────────────────────────────


    def process_dialogue(self, dialogue: dict, remove_half_GT_answers_from_train_dataset=False, allow_do_not_know=False):
        prompt_template = ""
        if self.tokenizer.bos_token_id is not None:
            prompt_template += f"{self.tokenizer.decode([self.tokenizer.bos_token_id])}"

        prompts = dialogue["prompt"]
        if prompts[-1]["role"] == "assistant":
            prompts = prompts[:-1]
        for message in prompts:
            prompt_template += f"{self.start_header_text}{message['role']}{self.end_header_text}\n{message['content']}{self.eot_text}\n"
        # append bot token
        prompt_template += f"{self.start_header_text}assistant{self.end_header_text}\n"

        extra = {key: value for key, value in dialogue.items() if key != "prompt"}

        return prompt_template, extra

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, extra in item_list:
            all_inputs.append((prompt, extra))
        return all_inputs

    def __getitem__(self, idx):
        inputs = self.dialogues[idx]
        return inputs

    def __len__(self):
        return len(self.dialogues)
