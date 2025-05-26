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

# disable_progress_bar()

DEVICE          = "cuda"
# MODEL_NAME      = "idefics2"

LLAVA_ID        = "llava-hf/llava-v1.6-mistral-7b-hf"
MISTRAL_ID      = "mistralai/Mistral-7B-Instruct-v0.2"
NEMO_ID         = "mistralai/Mistral-Nemo-Instruct-2407"
LLAMA31_ID      = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA32_ID      = "meta-llama/Llama-3.2-3B-Instruct"
IDEFICS2_ID     = "HuggingFaceM4/idefics2-8b"
MIXTRAL_ID      = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLAMA_31_70B    = "meta-llama/Llama-3.1-70B-Instruct"


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
        processor.end_of_utterance_token = "[/INST]"
        tokenizer = processor.tokenizer
        tokenizer.end_of_utterance_token = "[/INST]"
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
        end_token = "<end_of_utterance>"
    elif model_name == "mixtral":
        from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
        model = AutoModelForCausalLM.from_pretrained(MIXTRAL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_ID)
        processor = None
    elif model_name.lower() == "llama3.1-70b":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_31_70B, token=os.getenv("MY_HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(LLAMA_31_70B, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        processor = None
    
    model.eval()
    return model, tokenizer, processor


def prepare_input(tokenizer, prompt, image=None, is_visual=False, device="cuda"):
    if is_visual:
        conv = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
                ]
            },
            ]
    else:
        conv = [
            {"role": "user", "content": prompt}
        ]

    conv = tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)

    if is_visual:
        image = Image.open(image)
        inputs = tokenizer(images=image, text=conv, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(text=conv, return_tensors="pt").to(device)

    # masking
    _mask_prompt = prompt.split(":")[0] + ": "
    if is_visual:
        _mask_conv = [
            {"role": "user", "content": [
                {"type": "text", "text": _mask_prompt},
                {"type": "image"},
                ]
            },
            ]
    else:
        _mask_conv = [
            {"role": "user", "content": _mask_prompt}
        ]
    
    _end_token = str(tokenizer.end_of_utterance_token)

    _mask_conv = tokenizer.apply_chat_template(_mask_conv, add_generation_prompt=False, tokenize=False)
    _mask_conv = _mask_conv.replace(_end_token, "")

    if is_visual:
        mask_inputs = tokenizer(images=image, text=_mask_conv, return_tensors="pt").to(device).input_ids
    else:
        mask_inputs = tokenizer(text=_mask_conv, return_tensors="pt").to(device).input_ids

    labels = inputs["input_ids"].clone()
    # labels[0, :mask_inputs.shape[-1] - 1] = -100 # masking non-candidate tokens
    inputs.update({"labels": labels})
    return inputs


def eval(args):
    dataset_path = "dataset/dataset.avail.json"
    dataset = json.load(open(dataset_path))

    results_outdir = os.path.join("results-ex2cat", "final-results-080225")    
    _model_name = args.model + "-visual.json" if args.visual else + ".json"
    output_fn = os.path.join(results_outdir, _model_name)

    os.makedirs(results_outdir, exist_ok=True)

    print(f"{dataset_path=}")
    print(f"{args.visual=}")
    print(f"{args.model=}")

    categories  = sorted(list(set([elem["category"] for elem in dataset])))
    concepts    = sorted(list(set([elem["concept"] for elem in dataset])))

    model, tokenizer, processor = get_model(model_name=args.model)
    if args.visual:
        tokenizer = processor

    model.eval()

    all_res = []
    for sample in tqdm(dataset[:10]):
        human_exemplars = sample["data"]["candidates"]["easy"][sample["data"]["answers"]["easy"]]
        img = os.path.join(args.imgdir, sample["img"]) if args.visual else None
        gt_category = CATEGORY_MAPPER_ITA[sample["category"]]
        gt_concept = sample["concept"]

        cat_ppls = {}
        concept_ppls = {}

        for cat in tqdm(categories):
            prompt = ", ".join(human_exemplars)
            prompt = f"''{prompt}''. Questi sono elementi che appartengono alla categoria: {CATEGORY_MAPPER_ITA[cat]}"

            model_inputs = prepare_input(tokenizer=tokenizer, prompt=prompt, image=img, is_visual=args.visual, device=DEVICE)
            
            with torch.no_grad():
                loss = model(**model_inputs).loss

            cat_ppls[cat] = torch.exp(loss).item()

        p_cat = CATEGORY_MAPPER_ITA[min(cat_ppls, key=cat_ppls.get)]

        for concept in tqdm(concepts):
            prompt = ", ".join(human_exemplars)
            prompt = f"''{prompt}''. Questi sono elementi che appartengono al concetto: {concept}"

            model_inputs = prepare_input(tokenizer=tokenizer, prompt=prompt, image=img, is_visual=args.visual, device=DEVICE)

            with torch.no_grad():
                loss = model(**model_inputs).loss

            concept_ppls[concept] = torch.exp(loss).item()

        p_concept = min(concept_ppls, key=concept_ppls.get)
        
        all_res.append({"ppl_cat": cat_ppls, "gt_cat": gt_category, "pred_cat": p_cat, "ppl_concept": concept_ppls, "gt_concept": gt_concept, "pred_concept": p_concept, "human_exemplars": human_exemplars})
        

    with open (output_fn, "w") as f:
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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--imgdir", type=str, default="dataset/concept-images/stable-diffusion-xl-base-1.0/0319_1724")
    parser.add_argument("--visual", action="store_true")

    args = parser.parse_args()
    eval(args)
