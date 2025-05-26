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


def get_model(model_name="llava", device="cuda:0", padding_side="left"):
    if model_name == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(LLAVA_ID, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(LLAVA_ID, padding_side=padding_side)
        tokenizer = processor.tokenizer
    elif model_name == "mistral":
        model = AutoModelForCausalLM.from_pretrained(MISTRAL_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN")).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_ID, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        processor = None
    elif model_name == "nemo":
        model = AutoModelForCausalLM.from_pretrained(NEMO_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN")).to(device)
        tokenizer = AutoTokenizer.from_pretrained(NEMO_ID, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        processor = None
    elif model_name == "llama3.1":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA31_ID, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(LLAMA31_ID, token=os.getenv("MY_HF_TOKEN")).to(device)
        processor = None
    elif model_name == "llama3.2":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA32_ID, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(LLAMA32_ID, token=os.getenv("MY_HF_TOKEN")).to(device)
        processor = None
    elif model_name == "idefics2":
        processor = AutoProcessor.from_pretrained(IDEFICS2_ID, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        model = AutoModelForPreTraining.from_pretrained(IDEFICS2_ID, token=os.getenv("MY_HF_TOKEN"), torch_dtype=torch.bfloat16, device_map="auto").to(device)
        tokenizer = processor.tokenizer
        tokenizer.chat_template = processor.chat_template
        tokenizer.eos_token = processor.end_of_utterance_token
    elif model_name == "mixtral":
        model = AutoModelForCausalLM.from_pretrained(MIXTRAL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_ID, padding_side=padding_side)
        processor = None
    elif model_name.lower() == "llama3.1-70b":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_31_70B, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(LLAMA_31_70B, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        processor = None
    
    return model, tokenizer, processor


def eval(args):
    data = pd.read_csv("data/human_exemplars.andrea.csv")
    concepts = data.concept.unique()

    annotated_df_dict = {}
    models = ["llama3.1", "llama3.2", "mistral", "llava", "idefics2", "nemo"]

    for model_name in models:
        model, tokenizer, processor = get_model(model_name=model_name, device=DEVICE)

        
        for concept in concepts:
            concept_results = []

            selected = data[data.concept == concept].sort_values(by="availability", ascending=False)
            category = selected.iloc[0].category

            if concept not in annotated_df_dict:
                annotated_df_dict[concept] = selected

            for i, row in selected.iterrows():
                # prompt = f"{row.exemplar.split('_')[-1].title()} è un tipo di {concept.lower()}"
                prompt = f"{row.exemplar_string.title()} è un tipo di {concept.lower()}"

                if model_name == "idefics2":
                    conv = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                else:
                    conv = [{"role": "user", "content": prompt}]

                text = tokenizer.apply_chat_template(conv, tokenize=False)
                model_inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    loss = model(**model_inputs, labels=model_inputs.input_ids).loss.detach().item()
                concept_results.append(loss)
            
            annotated_df_dict[concept][model_name] = concept_results
        
        del model, tokenizer, processor
        torch.cuda.empty_cache()
    
    merged_df = pd.concat(annotated_df_dict.values())
    merged_df.to_csv("prototipicality.small.andrea.csv")
                


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--name", default="availability", type=str, help="Sorting modality of the human exemplars (either 'availability' or 'rank')")
    # parser.add_argument("--source", default="human", type=str, help="Source of the candidates (either 'human' or 'llm')")
    # parser.add_argument("--selected", default=False, action="store_true")
    # parser.add_argument("--enumerate", default=False, action="store_true")
    # parser.add_argument("--semantic", action="store_true")
    # parser.add_argument("--nosave", default=False, action="store_true")

    args = parser.parse_args()
    eval(args)
