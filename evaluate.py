import os, json, argparse

import json
import os
import re
import torch


FEWSHOT_PROMPTS = \
"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""

SYSTEM_PROMPT = """Your task is to answer a math question. You must give the final answer with the format "The answer is [number]." in the end of the output. 
\nHere are some examples:"""


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("./grade_school_math/data", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer_from_completion(model_completion) == gt_answer


def construct_prompt(question, model_name):
    model_name = model_name.lower()
    return f"{FEWSHOT_PROMPTS}\nQ: {question}\nA: "
    # if "llama" in model_name:
    #     prompt = f"<>"

def construct_prompt_with_template(question, model_name):
    model_name = model_name.lower()
    if "llama" in model_name:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}\n{FEWSHOT_PROMPTS}<|eot_id|>"
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQ: {question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nA: "
        )
        return prompt

def extract_answer_from_completion(completion):
    completion = completion.strip().replace("\n", " ")
    regex = r"The answer is ([0-9\.\,]+)."
    match = re.search(regex, completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def generate_answers(model_path, test_questions, temperature=0.7, max_tokens=1024, top_p=1, stop_token_ids=None, max_model_len=4096, gpu_memory_utilization=0.9, dtype=torch.float16, num_gpus=2):
        from vllm import LLM, SamplingParams
        print("start generating, test question length: ", len(test_questions))
        model_name = model_path.split("/")[-1]
        # test_questions = [construct_prompt_with_template(question, model_name) for question in test_questions]
        test_questions = [construct_prompt(question, model_name) for question in test_questions]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )
        llm = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            disable_custom_all_reduce=True,
            max_model_len=max_model_len,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization
        )
        outputs = llm.generate(test_questions, sampling_params)

        final_ans_jsons = []
        for output in outputs:
            text = output.outputs[0].text
            final_ans_jsons.append(text)

        return final_ans_jsons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True, help="path to the model")
    parser.add_argument("--split", type=str, default="test", help="split to evaluate on")
    parser.add_argument("-n", "--num-examples", type=int, default=-1, help="number of examples to evaluate on (-1 for all)")
    parser.add_argument("--generate", type=int, choices=[0,1], default=1, help="whether to generate completions")
    parser.add_argument("--temperature", type=float, default=0.001, help="temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="max tokens for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="top-p for sampling")
    parser.add_argument("--stop_token_ids", type=int, nargs="+", default=None, help="stop token ids for sampling")
    parser.add_argument("--max_model_len", type=int, default=4096, help="max model length for generation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.99, help="gpu memory utilization for generation")
    parser.add_argument("--dtype", type=str, default="float16", help="data type for generation")
    args = parser.parse_args()

    examples = get_examples(args.split)
    examples = examples[:args.num_examples] if args.num_examples > 0 else examples
    test_questions = [ex["question"] for ex in examples]

    model_name = args.model_path.split("/")[-1]
    save_path = f"./completetions/model_completions_{model_name}_{args.split}.jsonl"

    # get the number of gpus
    num_gpus = torch.cuda.device_count()

    if args.generate:
        model_completions = generate_answers(
            model_path=args.model_path,
            test_questions=test_questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            stop_token_ids=args.stop_token_ids,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=getattr(torch, args.dtype),
            num_gpus=num_gpus
        )
    else:
        save_data = read_jsonl(save_path)
        model_completions = [ex["model_completion"] for ex in save_data]

    assert len(model_completions) == len(test_questions)

    correct_count = 0
    save_jsons = []
    for i, (model_completion, gt_example) in enumerate(zip(model_completions, examples)):
        if is_correct(model_completion, gt_example):
            correct_count += 1
        # else:
        #     print(f"Example {i} failed: {model_completion} vs {gt_example['answer']}") 
        # save the model completions and the ground truth answers for further analysis
        save_jsons.append({"question": gt_example["question"], "model_completion": model_completion, "gt_answer": gt_example["answer"], "correct": is_correct(model_completion, gt_example)})
    
    with open(save_path, "w") as f:
        f.write(json.dumps({"correct num": correct_count, "total num": len(examples), "accuracy": correct_count / len(examples)}))
        f.write("\n")
        for json_obj in save_jsons:
            f.write(json.dumps(json_obj) + "\n")

    print(f"Accuracy for {model_name} on {args.split} set: {correct_count / len(examples)}")


if __name__ == "__main__":
    main()