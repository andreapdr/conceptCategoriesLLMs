import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')

from tqdm import tqdm
from PIL import Image
import os
import json
import pandas as pd
import numpy as np
import re

import torch

from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForPreTraining
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

DEVICE          = "cuda:2"

LLAVA_ID        = "llava-hf/llava-v1.6-mistral-7b-hf"
MISTRAL_ID      = "mistralai/Mistral-7B-Instruct-v0.2"
NEMO_ID         = "mistralai/Mistral-Nemo-Instruct-2407"
LLAMA31_ID      = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA32_ID      = "meta-llama/Llama-3.2-3B-Instruct"
IDEFICS2_ID     = "HuggingFaceM4/idefics2-8b"
MIXTRAL_ID      = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLAMA_31_70B    = "meta-llama/Llama-3.1-70B-Instruct"


def get_model(model_name="llava"):
    if model_name == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(LLAVA_ID, torch_dtype=torch.bfloat16).to(DEVICE)
        processor = AutoProcessor.from_pretrained(LLAVA_ID)
        tokenizer = processor.tokenizer
        # start_token = "[INST]"
        # end_token = "[/INST]"
        start_token = "<s>"
        end_token = "</s>"
    elif model_name == "mistral":
        model = AutoModelForCausalLM.from_pretrained(MISTRAL_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN")).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_ID, token=os.getenv("MY_HF_TOKEN"))
        tokenizer.pad_token = tokenizer.eos_token
        processor = None
        start_token = "<s>"
        end_token = "</s>"
    elif model_name == "nemo":
        model = AutoModelForCausalLM.from_pretrained(NEMO_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN")).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(NEMO_ID, token=os.getenv("MY_HF_TOKEN"))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        processor = None
        start_token = "<s>"
        end_token = "</s>"
    elif model_name == "llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA31_ID, token=os.getenv("MY_HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(LLAMA31_ID, token=os.getenv("MY_HF_TOKEN")).to(DEVICE)
        processor = None
    elif model_name == "llama3.2":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA32_ID, token=os.getenv("MY_HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(LLAMA32_ID, token=os.getenv("MY_HF_TOKEN")).to(DEVICE)
        processor = None
        start_token = "<|begin_of_text|>"
        end_token = "<|end_of_text|>"
    elif model_name == "idefics2":
        processor = AutoProcessor.from_pretrained(IDEFICS2_ID, token=os.getenv("MY_HF_TOKEN"))
        model = AutoModelForPreTraining.from_pretrained(IDEFICS2_ID, token=os.getenv("MY_HF_TOKEN"), torch_dtype=torch.bfloat16).to(DEVICE)
        tokenizer = processor.tokenizer
    elif model_name == "mixtral":
        from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
        model = AutoModelForCausalLM.from_pretrained(MIXTRAL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_ID)
        processor = None
    elif model_name.lower() == "llama3.1-70b":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_31_70B, token=os.getenv("MY_HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(LLAMA_31_70B, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        processor = None
    
    return model, tokenizer, processor, start_token, end_token


def run_dataset_experiment(model, tokenizer, dataset, mode="easy", verbose=False, is_visual=False, enumerate_prompt=False, start_token="", end_token=""):
    # TASK_PROMPT = "Data una parola che denota un concetto, elenca tutti i \"tipi di\" quel concetto. <concept>:"
    # TASK_PROMPT = "Given an Italian word denoting a concept, list all of the possible Italian \"kinds of\" for that concept. <concept>:"
    # TASK_PROMPT = "Ecco una lista di parole ordinate secondo il loro grado di familiarità. <concept>:"
    # TASK_PROMPT = "The following are the most prominent elements associated with the concept of <concept>:"
    TASK_PROMPT = "Questi sono gli esemplari principali che appartengono al concetto di <concept>:"

    idx_to_char = {0: "A", 1: "B", 2: "C", 3: "D"}
    accuracy = 0
    all_ppl = []

    is_llava = isinstance(model, LlavaNextForConditionalGeneration)
    # TODO append [INST] [/INST] tokens?

    for i, data in enumerate(tqdm(dataset)):
        concept = data["concept"]
        candidates = data["data"]["candidates"][mode]
        target = idx_to_char[data["data"]["answers"][mode]]
        img_name = data["img"]

        if verbose: print(f"{concept=}")
        if verbose: print(f"{candidates=}")
        if verbose: print(f"{target=}")

        with torch.no_grad():
            losses = []
            for candidate in candidates:

                if is_visual:
                    if enumerate_prompt:
                        prompt = f"<image>\n{TASK_PROMPT.replace('<concept>', concept)} N) " + ", N) ".join(candidate)
                        prompt = replace_with_numbers(prompt)
                    else:
                        prompt = f"<image>\n{TASK_PROMPT.replace('<concept>', concept)} " + ", ".join(candidate)

                    img = f"dataset/concept-images/stable-diffusion-xl-base-1.0/0319_1724/{img_name}"
                    model_inputs = tokenizer(images=Image.open(img), text=prompt, return_tensors="pt").to(DEVICE)
                    to_mask = tokenizer(images=Image.open(img), text=prompt.split(":")[0], return_tensors="pt").input_ids
                    labels = model_inputs["input_ids"].clone()
                else:
                    if enumerate_prompt:
                        prompt = f"{start_token} {TASK_PROMPT.replace('<concept>', concept)} N) " + ", N) ".join(candidate) + end_token
                        prompt = replace_with_numbers(prompt)
                    else:
                        prompt = f"{TASK_PROMPT.replace('<concept>', concept)} " + ", ".join(candidate)
                    model_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                    to_mask = tokenizer(prompt.split(":")[0], return_tensors="pt").input_ids
                    labels = model_inputs["input_ids"].clone()

                if verbose: print(f"{prompt=}")

                labels[0, :to_mask.shape[-1] + 1] = -100 # masking non-candidate tokens
                loss = model(**model_inputs, labels=labels).loss
                losses.append(torch.exp(loss).item())

        losses = [round(elem, 3) for elem in losses]
        answer = idx_to_char[losses.index(min(losses))]
        correct = answer == target
        if verbose: print(f"{concept:<10} {correct:>5} - {answer=}, {target=}, {losses=}")
        if correct:
            accuracy += 1
        
        all_ppl.append(losses)
        # all_preds.append(correct)
    
    accuracy = (accuracy / len(dataset))*100
    return accuracy, all_ppl


def replace_with_numbers(text):
    count = 1
    def replacer(match):
        nonlocal count
        replacement = f"{count})"
        count += 1
        return replacement

    # Use re.sub with a custom replacer function
    new_text = re.sub(r'N\)', replacer, text)
    return new_text


def eval(args):
    if args.name == "availability":
        if args.selected:
            dataset_name = "dataset/dataset_avail_subset.json" 
        else:
            dataset_name = "dataset/avail/bydiff/binary_dataset_avail.json"
    elif args.name == "rank":
        if args.selected:
            dataset_name = "dataset/dataset_avgrank_subset.json"
        else:
            dataset_name = "dataset/avgrank/bydiff/binary_dataset_avgrank.json"
    else:
        raise NameError

    do_enumerate = args.enumerate
    source = args.source
    
    import os
    output_dir = f"results-detection/bydiff/{args.name}/"
    os.makedirs(output_dir, exist_ok=True)

    # output_name = dataset_name.split("/")[-1].replace(".json", "")
    output_name = "results"

    if do_enumerate:
        output_name += "_enum"

    dataset = json.load(open(dataset_name))
    print(f"dataset: {dataset_name}") 

    models = ["llama3.1", "llama3.2", "nemo", "mistral", "llava", "idefics2"]
    models_visual = ["idefics2", "llava"]
    models = ["llama3.1-70B", "mixtral"]
    models_visual = []

    results = {"model": [], "easy": [], "medium": [], "hard": [], "llm": []}
    for model_name in models:
        print(f"- testing {model_name}")
        model, tokenizer, processor, bos_token, eos_token = get_model(model_name=model_name)

        acc_easy, easy_ppl = run_dataset_experiment(model, tokenizer, dataset,      mode="easy",     verbose=False, enumerate_prompt=do_enumerate, start_token=bos_token, end_token=eos_token)
        acc_medium, medium_ppl = run_dataset_experiment(model, tokenizer, dataset,  mode="medium",   verbose=False, enumerate_prompt=do_enumerate, start_token=bos_token, end_token=eos_token)
        acc_hard, hard_ppl = run_dataset_experiment(model, tokenizer, dataset,      mode="hard",     verbose=False, enumerate_prompt=do_enumerate, start_token=bos_token, end_token=eos_token)
        acc_llm, llm_ppl = run_dataset_experiment(model, tokenizer, dataset,        mode="llm",      verbose=False, enumerate_prompt=do_enumerate, start_token=bos_token, end_token=eos_token)

        results["model"].append(model_name)
        results["easy"].append(acc_easy)
        results["medium"].append(acc_medium)
        results["hard"].append(acc_hard)
        results["llm"].append(acc_llm)

        for i, elem in enumerate(dataset):
            output_key = "outputs-human" if source == "human" else "outputs-llm"
            if output_key not in elem:
                elem[output_key] = {}
            elem[output_key][model_name] =  {"easy": easy_ppl[i], "medium": medium_ppl[i], "hard": hard_ppl[i], "llm": llm_ppl[i]}
        
        del model, tokenizer, processor
        torch.cuda.empty_cache()

    for model_name in models_visual:
        print(f"- testing {model_name} (with visual)")
        model, _ , tokenizer = get_model(model_name=model_name)

        acc_easy, easy_ppl = run_dataset_experiment(model, tokenizer, dataset,      mode="easy",     verbose=False, is_visual=True, enumerate_prompt=do_enumerate)
        acc_medium, medium_ppl = run_dataset_experiment(model, tokenizer, dataset,  mode="medium",   verbose=False, is_visual=True, enumerate_prompt=do_enumerate)
        acc_hard, hard_ppl = run_dataset_experiment(model, tokenizer, dataset,      mode="hard",     verbose=False, is_visual=True, enumerate_prompt=do_enumerate)
        acc_llm, llm_ppl = run_dataset_experiment(model, tokenizer, dataset,        mode="llm",      verbose=False, is_visual=True, enumerate_prompt=do_enumerate)

        results["model"].append(model_name + "_img")
        results["easy"].append(acc_easy)
        results["medium"].append(acc_medium)
        results["hard"].append(acc_hard)
        results["llm"].append(acc_llm)

        for i, elem in enumerate(dataset):
            output_key = "outputs-human" if source == "human" else "outputs-llm"
            if output_key not in elem:
                elem[output_key] = {}
            elem[output_key][model_name] =  {"easy": easy_ppl[i], "medium": medium_ppl[i], "hard": hard_ppl[i], "llm": llm_ppl[i]}
    
    res_df = pd.DataFrame.from_dict(data=results, orient="columns")
    res_df["avg"] = res_df[["easy", "medium", "hard", "llm"]].mean(axis=1)
    print(res_df.round(2))
    
    if not args.nosave:
        res_df.to_csv(f"{output_dir + output_name}.diff.bigmodels.csv", index=False)
        with open(f"{output_dir + output_name}.diff.annotated.bigmodels.json", "w") as f:
            json.dump(dataset, f, ensure_ascii=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--name", default="availability", type=str, help="Sorting modality of the human exemplars (either 'availability' or 'rank')")
    parser.add_argument("--source", default="human", type=str, help="Source of the candidates (either 'human' or 'llm')")
    parser.add_argument("--selected", default=False, action="store_true")
    parser.add_argument("--enumerate", default=False, action="store_true")
    parser.add_argument("--nosave", default=False, action="store_true")

    args = parser.parse_args()
    eval(args)