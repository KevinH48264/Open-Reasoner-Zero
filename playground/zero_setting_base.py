from typing import List

from jinja2 import Template

from orz.ppo import PromptDataset

import random
random.seed(42)


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List, remove_half_GT_answers_from_train_dataset=False, allow_do_not_know=False):
#         prompt_template_jinja = """\
# {{bos_token}}You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. User: {{prompt}}
# Assistant:\nLet me solve this step by step.\n<think>\
# """ # prompt here = prompt_instruction_template_jinja
# #         prompt_instruction_template_jinja = """\
# # You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# # This is the problem:
# # {{prompt}}
# # """
#         prompt_instruction_template_jinja = """{{prompt}}""" # remove the boxed requirement

    # --- v1310 start ---
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n\n<answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
""" # prompt here = prompt_instruction_template_jinja
        if not allow_do_not_know:
            prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
This is the problem:
{{prompt}}
"""
    # --- v1310 end ---

#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """ # prompt here = prompt_instruction_template_jinja
#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """

        # v1315
#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer (solution + final answer). \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n\n<answer> solution + final answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """ # prompt here = prompt_instruction_template_jinja
#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> solution + final answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """

        # --- v1405 start ---
        if allow_do_not_know:
            prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. If you do not know the answer, you can write "I do not know" inside the <answer> tags. 
This is the problem:
{{prompt}}
"""

        assert len(dialogue) == 2, "dialogue must contain 2 items"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        if remove_half_GT_answers_from_train_dataset and random.random() < 0.5:
            extra = {
                "prompt": prompt,
                "answer": "[NO GT ANSWER]",
                "target": dialogue[1]["ground_truth"]["value"],
            } # 50% chance to remove the GT answer for training dataset
        else:
            extra = {
                "prompt": prompt,
                "answer": dialogue[1]["ground_truth"]["value"],
                "target": dialogue[1]["ground_truth"]["value"],
            }

        # Countdown specific extra
        if "nums" in dialogue[1]:
            extra["nums"] = dialogue[1]["nums"]["value"]

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict, remove_half_GT_answers_from_train_dataset=False, allow_do_not_know=False):
#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """

        # --- v1310 start ---
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n\n<answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
""" # prompt here = prompt_instruction_template_jinja

        if not allow_do_not_know:
            prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
This is the problem:
{{prompt}}
"""
        # --- v1310 end --- #

#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """


#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer (solution + final answer). \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. your response format should be: <think> reasoning process here </think>\n\n<answer> solution + final answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """ # prompt here = prompt_instruction_template_jinja
#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> solution + final answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """

        # --- v1405 start ---
        if allow_do_not_know:
            prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. If you do not know the answer, you can write "I do not know" inside the <answer> tags. 
This is the problem:
{{prompt}}
"""

        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue["prompt"][0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        # Custom for '/data/users/kevihuang/projects/Open-Reasoner-Zero/data/eval_data/prm800k_100_correct_100_incorrect_rm_eval.json'
        if "Math Teacher Response: <think>\n" in dialogue["prompt"][0]["value"]:
            prompt = dialogue["prompt"][0]["value"] # No extra stuff, just normal RF Eval Prompt

        extra = {"prompt": prompt, "answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
