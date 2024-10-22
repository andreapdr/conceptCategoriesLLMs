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

DEVICE          = "cuda:3"
MODEL_NAME      = "nemo"

LLAVA_ID        = "llava-hf/llava-v1.6-mistral-7b-hf"
MISTRAL_ID      = "mistralai/Mistral-7B-Instruct-v0.2"
NEMO_ID         = "mistralai/Mistral-Nemo-Instruct-2407"
LLAMA31_ID      = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA32_ID      = "meta-llama/Llama-3.2-3B-Instruct"
IDEFICS2_ID     = "HuggingFaceM4/idefics2-8b"
MIXTRAL_ID      = "mistralai/Mixtral-8x7B-Instruct-v0.1"

CATEGORY_MAPPER_ITA = {
    'animals': 'animali',
    'body parts': 'parti del corpo',
    'clothes': 'vestiti',
    'foods': 'cibi',
    'furnishings/fittings': 'arredamenti/accessori',
    'furniture': 'mobili',
    'hobbies': 'passatempi',
    'housing buildings': 'edifici residenziali',
    'kitchenware': 'utensili da cucina',
    'plants': 'piante',
    'stationery': 'cancelleria',
    'vehicles': 'veicoli'
}

def get_model(model_name="llava"):
    if model_name == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(LLAVA_ID, torch_dtype=torch.bfloat16).to(DEVICE)
        processor = AutoProcessor.from_pretrained(LLAVA_ID)
        tokenizer = processor.tokenizer
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
    
    return model, tokenizer, processor


def eval():
    dataset_path = "dataset/avail/bydiff/binary_dataset_avail.json"
    dataset = json.load(open(dataset_path))
    print(f"{dataset_path=}")
    print(f"{MODEL_NAME=}")

    categories  = sorted(list(set([elem["category"] for elem in dataset])))
    concepts    = sorted(list(set([elem["concept"] for elem in dataset])))

    model, tokenizer, processor = get_model(model_name=MODEL_NAME)

    all_res = []
    for sample in tqdm(dataset):
        human_exemplars = sample["data"]["candidates"]["easy"][sample["data"]["answers"]["easy"]]
        gt_category = CATEGORY_MAPPER_ITA[sample["category"]]
        gt_concept = sample["concept"]

        cat_ppls = {}
        concept_ppls = {}

        for cat in categories:
            prompt = ", ".join(human_exemplars)
            prompt = f"{prompt}. Questi sono elementi che appartengono alla categoria: {CATEGORY_MAPPER_ITA[cat]}"

            model_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            to_mask = tokenizer(prompt.split(":")[0] + ":", return_tensors="pt").input_ids
            labels = model_inputs["input_ids"].clone()
            labels[0, :to_mask.shape[-1]] = -100 # masking non-candidate tokens

            loss = model(**model_inputs, labels=labels).loss
            cat_ppls[cat] = torch.exp(loss).item()
        p_cat = CATEGORY_MAPPER_ITA[min(cat_ppls, key=cat_ppls.get)]

        for concept in concepts:
            prompt = ", ".join(human_exemplars)
            prompt = f"{prompt}. Questi sono elementi che appartengono al concetto: {concept}"

            model_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            to_mask = tokenizer(prompt.split(":")[0] + ":", return_tensors="pt").input_ids
            labels = model_inputs["input_ids"].clone()
            labels[0, :to_mask.shape[-1]] = -100 # masking non-candidate tokens

            loss = model(**model_inputs, labels=labels).loss
            concept_ppls[concept] = torch.exp(loss).item()

        p_concept = min(concept_ppls, key=concept_ppls.get)
        
        all_res.append({"ppl_cat": cat_ppls, "gt_cat": gt_category, "pred_cat": p_cat, "ppl_concept": concept_ppls, "gt_concept": gt_concept, "pred_concept": p_concept, "human_exemplars": human_exemplars})
        
    # print(all_res)

    with open (f"results-ex2cat/{MODEL_NAME}_results.json", "w") as f:
        json.dump(all_res, f, ensure_ascii=False)
    
    acc_cat = 0
    acc_concept = 0
    for elem in all_res:
        if elem["pred_cat"] == elem["gt_cat"]:
            acc_cat += 1
        
        if elem["pred_concept"] == elem["gt_concept"]:
            acc_concept += 1
    
    print(f"Acc concept: {acc_concept / len(all_res):.2f}")
    print(f"Acc category: {acc_cat / len(all_res):.2f}")
        

if __name__ == "__main__":
    eval()
