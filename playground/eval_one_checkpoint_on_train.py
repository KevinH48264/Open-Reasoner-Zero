#!/usr/bin/env python
# python -m playground.eval_one_checkpoint_on_train

import os, re, json, math, random, gc, asyncio, torch, time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from playground.zero_setting_base import CustomDataset
from orz.ppo.tools.math_utils import is_equal, solution2answer

# ───────────────────── helpers ─────────────────────
def last_boxed_only_string(s: str):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0: return None
    i, lvl = idx, 0
    while i < len(s):
        if s[i] == "{": lvl += 1
        elif s[i] == "}":
            lvl -= 1
            if lvl == 0: return s[idx:i+1]
        i += 1
    return None

def remove_boxed(s):
    return s[7:-1] if s and s.startswith("\\boxed{") and s.endswith("}") else None

def extract_final_answer(txt: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", txt, re.DOTALL)
    if m: return m.group(1).strip()
    boxed = last_boxed_only_string(txt)
    if boxed:
        inner = remove_boxed(boxed)
        if inner: return inner
    return txt.strip()

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

# ───────────────────── configuration ─────────────────────
CKPT  = "/vc_data_blob/users/kevihuang/models/orz/orz_ckpt/debug_orz_7b_ppo_self_play__math__v1481__self_play_False/iter450/policy"
FILES = ["/vc_data_blob/users/kevihuang/data/orz/openmathreasoning__cot__has_answer_extracted_only__easiest_half.json"]
TEMP  = 1.0
MAX_GEN_TOK = 8000
PROMPT_MAXLEN = 8000
BATCH = 1024
EVAL_N, ROLLOUTS, SEED = 100, 4, 42
OUT_JSON = "playground/eval_one_checkpoint_on_train.json"
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

# ───────────────────── evaluation ─────────────────────
async def main():
    random.seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
    llm = LLM(model=CKPT, tensor_parallel_size=4)

    gen_params = SamplingParams(
        temperature=TEMP, top_p=1.0, top_k=-1,
        max_tokens=MAX_GEN_TOK,
        stop=["</answer>"], skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    judge_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_GEN_TOK,
        stop=["</answer>"], skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 32)

    for f in FILES:
        print(f"\nEvaluating {f}")
        dialogs = json.load(open(f))
        ds = CustomDataset(dialogs, tokenizer, PROMPT_MAXLEN,
                           strategy=None, pretrain_mode=False, num_processors=1)
        samples = random.sample(list(ds), EVAL_N)

        # Build rollout prompts
        prompts, expected = [], []
        for s in samples:
            if isinstance(s, tuple):
                p, extra = s
                exp = str(extra.get("answer","")).strip()
            else:
                p, exp = s.get("prompt",""), str(s.get("answer","")).strip()
            for _ in range(ROLLOUTS):
                prompts.append(p); expected.append(exp)

        # Run policy rollouts
        responses = []
        for i in tqdm(range(0,len(prompts),BATCH), desc="rollouts",
                      total=math.ceil(len(prompts)/BATCH)):
            responses += llm.generate(prompts[i:i+BATCH], gen_params)
        responses = [r.outputs[0].text for r in responses]

        # Build and run judge prompts
        judge_prompts = []
        for ptxt, rtxt, exp in zip(prompts, responses, expected):
            q = ptxt.split("This is the problem:",1)[-1].rsplit("Assistant:",1)[0].strip()
            judge_prompts.append(build_judge_prompt(q, extract_final_answer(rtxt), exp))

        judge_replies = []
        for i in tqdm(range(0,len(judge_prompts),BATCH), desc="judge",
                      total=math.ceil(len(judge_prompts)/BATCH)):
            judge_replies += llm.generate(judge_prompts[i:i+BATCH], judge_params)
        judge_replies = [j.outputs[0].text for j in judge_replies]
        semantic_ok   = [parse_judge_reply(r) for r in judge_replies]

        # Exact match
        exact_tasks=[]
        for rtxt, exp in zip(responses, expected):
            gt  = solution2answer(exp).strip().lower()
            gen = solution2answer(extract_final_answer(rtxt)).strip().lower()
            exact_tasks.append(is_equal(gt, gen, executor))
        exact_ok = await asyncio.gather(*exact_tasks)

        # Collate & save 10 examples
        records = []
        for gp, exp, resp, jp, jr, sok, eok in zip(
                prompts, expected, responses,
                judge_prompts, judge_replies, semantic_ok, exact_ok):
            records.append({
                "generation_prompt": gp,
                "expected": exp,
                "model_response": resp,
                "judge_prompt": jp,
                "judge_reply": jr,
                "semantic_ok": sok,
                "exact_ok": eok,
            })

        json.dump(random.sample(records, 10), open(OUT_JSON,"w"), indent=2)
        print(f"Wrote 10 records → {OUT_JSON}")

        # Metrics
        tot = len(records)
        sem_ok_count   = sum(1 for r in records if r["semantic_ok"] is True)
        sem_none_count = sum(1 for r in records if r["semantic_ok"] is None)
        sem_acc        = sem_ok_count / tot
        none_pct       = sem_none_count / tot
        ex_acc         = sum(1 for r in records if r["exact_ok"]) / tot

        print(f"Semantic-judge accuracy : {sem_acc:.2%} ({sem_ok_count}/{tot})")
        print(f"Judge-reply none rate : {none_pct:.2%} ({sem_none_count}/{tot})")
        print(f"Exact is_equal accuracy : {ex_acc:.2%}")

    # Cleanup vLLM
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel(); destroy_distributed_environment()
    except Exception: pass
    del llm; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())
