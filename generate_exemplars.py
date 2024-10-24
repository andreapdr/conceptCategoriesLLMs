import torch
import argparse
import json
import pandas as pd
import pickle as pkl
import os

from datetime import datetime
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForPreTraining

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
        model = AutoModelForPreTraining.from_pretrained(IDEFICS2_ID, token=os.getenv("MY_HF_TOKEN"), torch_dtype=torch.bfloat16).to(device)
        tokenizer = processor.tokenizer
    elif model_name == "mixtral":
        model = AutoModelForCausalLM.from_pretrained(MIXTRAL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_ID, padding_side=padding_side)
        processor = None
    elif model_name.lower() == "llama3.1-70b":
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_31_70B, token=os.getenv("MY_HF_TOKEN"), padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(LLAMA_31_70B, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", token=os.getenv("MY_HF_TOKEN"))
        processor = None
    
    return model, tokenizer, processor


class ConceptDataset(Dataset):
    def __init__(self, tokenizer, datapath="data/unibo-concepts-it.csv", prompt="", multimodal=False):
        df = pd.read_csv(datapath)

        self.concepts = df.concept.tolist()
        self.images = df.img.tolist() if multimodal else None
        self.multimodal = multimodal
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.concepts)
    
    def __getitem__(self, index):
        concept = self.concepts[index]
        instruction_prompt, started_generation = self.prompt.split("<ROLE_ASSISTANT>")
        instruction_prompt = instruction_prompt.replace("<CONCEPT>", concept).rstrip()
        started_generation = started_generation.replace("<CONCEPT>", concept)

        conv = [
            {"role": "user", "content": instruction_prompt},
            {"role": "assistant", "content": started_generation}
        ]

        text = self.tokenizer.apply_chat_template(conv, tokenize=False)
        text = text.replace(self.tokenizer.eos_token, "")

        if self.multimodal:
            image = Image.open(f"data/concept-images/stable-diffusion-xl-base-1.0/0319_1724/{self.images[index]}").convert("RGB")
            text = "<s>[INST] <image>\n" + text
        else:
            image = None
            # text = "<s>[INST] " + text
        return {"text": text, "image": image, "concept": concept}


class LlavaCustomCollate:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        if examples[0]["image"] is not None:
            images = [e["image"] for e in examples]
        else:
            images = None
        inputs = self.processor(texts, return_tensors="pt", padding=True)
        return inputs 


def main(args):
    device = args.device
    language = args.language
    short_lang = language[:2]
    modality = args.modality
    temp = args.temp
    prompt_style = args.promptstyle
    batch_size = args.batchsize
    iter_name = args.iter
    model_name = args.model_name

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    base_outdir = "results-generation"

    raw_outdir = f"{base_outdir}/raw/{language}"
    outdir = f"{base_outdir}/{language}"

    for datadir in [raw_outdir, outdir]:
        os.makedirs(datadir, exist_ok=True)

    filename = f"unibo-concepts-{short_lang}{'-imgs' if modality == 'visual' else ''}"
    out_filename = f"{model_name}-{short_lang}-{modality}-t{str(temp).replace('.', '')}_iter{iter_name}"
    
    prompt_list = json.load(open(f"data/unibo-convs.json", "r"))
    prompt = prompt_list[language][modality][prompt_style]["messages"][0][-1]

    model, tokenizer, processor = get_model(model_name=model_name, device=device)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = ConceptDataset(tokenizer, datapath=f"data/{filename}.csv", prompt=prompt, multimodal=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=LlavaCustomCollate(processor=tokenizer))

    generation_config = {
        "timestamp": timestamp,
        "model": model.name_or_path,
        "modality": modality,
        "max_length": 300,
        "prompt": prompt,
        "modality": modality,
        "language": language,
        "prompt_style": prompt_style,
        "do_sample": True if temp != 0.0 else False,
        "temperature": temp,
        "top_k": 50,
        "top_p": 0.72,
        "repetition_penalty": 1.15,
        }

    output = []    
    for _, batch in enumerate(tqdm(dataloader)):
        batch.to(device)
        generate_ids = model.generate(
            **batch,
            max_length=generation_config["max_length"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
            pad_token_id=tokenizer.eos_token_id,
            )
        output.append(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))

    clean_output = [out.split("al concetto " if language == "italian" else "for the concept")[-1].lstrip() for out in sum(output, [])]

    # with open(f"{pickled_outdir}/{out_filename}.pkl", mode="wb") as file:
    #     pkl.dump(clean_output, file)

    with open(f"{raw_outdir}/{out_filename}.txt", mode="w") as file:
        for line in clean_output:
            file.write(line + "\n$$$$\n")
    
    formatted = {"_exp_config": generation_config}
    for line in clean_output:
        try:
            c, elements = line.split(":", maxsplit=1)
            elements = elements.lower().split("\n*")
            clean_elements = list(dict.fromkeys(elements[1:]))
            c = c.replace("tipi di` ", "")
            c = c.replace("`", "")
            formatted[c.replace("`", "")] = {i + 1: clean_exemplar(k) for i, k in enumerate(clean_elements)}
        except:
            formatted[c.replace("`", "")] = {}

    with open(f"{outdir}/{out_filename}.json", mode="w") as file:
        print(f"- storing results in {outdir}/{out_filename}.json")
        json.dump(formatted, file, ensure_ascii=False)


def clean_exemplar(exemplar):
    exemplar = exemplar.lstrip()
    exemplar = exemplar.replace("`", "")
    exemplar = exemplar.split("\n")[0]
    exemplar = exemplar.split("(")[0]
    exemplar = exemplar.rstrip()
    return exemplar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract subordinate categories from LLaVA-NeXT.')
    
    parser.add_argument("--model", dest="model_name", type=str, default="llama3.1", help="model name")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use")
    parser.add_argument("--language", type=str, default="italian", help="Language to use")
    parser.add_argument("--modality", type=str, default="textual", help="Modality to use")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature value")
    parser.add_argument("--promptstyle", type=str, default="guided-example", help="prompt style to use")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--iter", type=int, default=1)

    args = parser.parse_args()

    main(args)