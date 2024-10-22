from tqdm import tqdm

import os
import json
import pandas as pd
import numpy as np

import torch

from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from pprint import pprint as pp

import matplotlib as mpl
import matplotlib.pyplot as plt

HUMAN_DATAPATH = "data/best_human_exemplars.xlsx"

BPE_START_TOKEN = "▁"

IMAGE_BASEPATH  = "../unibo-subordinate/data/concept-images/stable-diffusion-xl-base-1.0/0319_1724"
DEVICE          = "cuda:0"

LLAVA_ID        = "llava-hf/llava-v1.6-mistral-7b-hf"
MISTRAL_ID      = "mistralai/Mistral-7B-Instruct-v0.2"
NEMO_ID         = "mistralai/Mistral-Nemo-Instruct-2407"

BASE_PROMPT = """<s>[INST] Data una parola che denota una concetto, elenca tutta i 'tipi di' quel concetto. Elenca solo i nomi delle entità. 
Per esempio per il concetto 'elettrodomestico' elenca:
* frullatore
* aspirapolvere
* tostapane
* lavatrice

Ora fai lo stesso per il concetto '<CONCEPT>' [/INST] Questa è una lista 'tipi di' che appartengono al concetto `<CONCEPT>`: \n*"""

BASE_PROMPT_IMG = """<s>[INST] <image>
Data una parola che denota una concetto, elenca tutta i 'tipi di' quel concetto. Elenca solo i nomi delle entità. 
Per esempio per il concetto 'elettrodomestico' elenca:
* frullatore
* aspirapolvere
* tostapane
* lavatrice

Ora fai lo stesso per il concetto '<CONCEPT>' [/INST] Questa è una lista 'tipi di' che appartengono al concetto `<CONCEPT>`: \n*"""

ITALIAN_STOPWORDS = ["a","abbastanza","abbia","abbiamo","abbiano","abbiate","accidenti","ad","adesso","affinché","agl","agli","ahime","ahimè","ai","al","alcuna","alcuni","alcuno","all","alla","alle","allo","allora","altre","altri","altrimenti","altro","altrove","altrui","anche","ancora","anni","anno","ansa","anticipo","assai","attesa","attraverso","avanti","avemmo","avendo","avente","aver","avere","averlo","avesse","avessero","avessi","avessimo","aveste","avesti","avete","aveva","avevamo","avevano","avevate","avevi","avevo","avrai","avranno","avrebbe","avrebbero","avrei","avremmo","avremo","avreste","avresti","avrete","avrà","avrò","avuta","avute","avuti","avuto","basta","ben","bene","benissimo","brava","bravo","buono","c","caso","cento","certa","certe","certi","certo","che","chi","chicchessia","chiunque","ci","ciascuna","ciascuno","cima","cinque","cio","cioe","cioè","circa","ciò","co","codesta","codesti","codesto","cogli","coi","col","colei","coll","coloro","colui","come","cominci","comprare","comunque","con","concernente","conclusione","consecutivi","consecutivo","consiglio","contro","cortesia","cos","cosa","cosi","così","cui","d","da","dagl","dagli","dai","dal","dall","dalla","dalle","dallo","dappertutto","davanti","degl","degli","dei","del","dell","della","delle","dello","dentro","detto","deve","devo","di","dice","dietro","dire","dirimpetto","diventa","diventare","diventato","dopo","doppio","dov","dove","dovra","dovrà","dovunque","due","dunque","durante","e","ebbe","ebbero","ebbi","ecc","ecco","ed","effettivamente","egli","ella","entrambi","eppure","era","erano","eravamo","eravate","eri","ero","esempio","esse","essendo","esser","essere","essi","ex","fa","faccia","facciamo","facciano","facciate","faccio","facemmo","facendo","facesse","facessero","facessi","facessimo","faceste","facesti","faceva","facevamo","facevano","facevate","facevi","facevo","fai","fanno","farai","faranno","fare","farebbe","farebbero","farei","faremmo","faremo","fareste","faresti","farete","farà","farò","fatto","favore","fece","fecero","feci","fin","finalmente","finche","fine","fino","forse","forza","fosse","fossero","fossi","fossimo","foste","fosti","fra","frattempo","fu","fui","fummo","fuori","furono","futuro","generale","gente","gia","giacche","giorni","giorno","giu","già","gli","gliela","gliele","glieli","glielo","gliene","grazie","gruppo","ha","haha","hai","hanno","ho","i","ie","ieri","il","improvviso","in","inc","indietro","infatti","inoltre","insieme","intanto","intorno","invece","io","l","la","lasciato","lato","le","lei","li","lo","lontano","loro","lui","là","ma","macche","magari","maggior","mai","male","malgrado","malissimo","me","medesimo","mediante","meglio","meno","mentre","mesi","mezzo","mi","mia","mie","miei","mila","miliardi","milioni","minimi","mio","modo","molta","molti","moltissimo","molto","momento","mondo","ne","negl","negli","nei","nel","nell","nella","nelle","nello","nemmeno","neppure","nessun","nessuna","nessuno","niente","no","noi","nome","non","nondimeno","nonostante","nonsia","nostra","nostre","nostri","nostro","novanta","nove","nulla","nuovi","nuovo","o","od","oggi","ogni","ognuna","ognuno","oltre","oppure","ora","ore","osi","ossia","ottanta","otto","paese","parecchi","parecchie","parecchio","parte","partendo","peccato","peggio","per","perche","perchè","perché","percio","perciò","perfino","pero","persino","persone","però","piedi","pieno","piglia","piu","piuttosto","più","po","pochissimo","poco","poi","poiche","possa","possedere","posteriore","posto","potrebbe","preferibilmente","presa","press","prima","primo","principalmente","probabilmente","promesso","proprio","puo","pure","purtroppo","può","qua","qualche","qualcosa","qualcuna","qualcuno","quale","quali","qualunque","quando","quanta","quante","quanti","quanto","quantunque","quarto","quasi","quattro","quel","quella","quelle","quelli","quello","quest","questa","queste","questi","questo","qui","quindi","quinto","realmente","recente","recentemente","registrazione","relativo","riecco","rispetto","salvo","sara","sarai","saranno","sarebbe","sarebbero","sarei","saremmo","saremo","sareste","saresti","sarete","sarà","sarò","scola","scopo","scorso","se","secondo","seguente","seguito","sei","sembra","sembrare","sembrato","sembrava","sembri","sempre","senza","sette","si","sia","siamo","siano","siate","siete","sig","solito","solo","soltanto","sono","sopra","soprattutto","sotto","spesso","sta","stai","stando","stanno","starai","staranno","starebbe","starebbero","starei","staremmo","staremo","stareste","staresti","starete","starà","starò","stata","state","stati","stato","stava","stavamo","stavano","stavate","stavi","stavo","stemmo","stessa","stesse","stessero","stessi","stessimo","stesso","steste","stesti","stette","stettero","stetti","stia","stiamo","stiano","stiate","sto","su","sua","subito","successivamente","successivo","sue","sugl","sugli","sui","sul","sull","sulla","sulle","sullo","suo","suoi","tale","tali","talvolta","tanto","te","tempo","terzo","th","ti","titolo","tra","tranne","tre","trenta","triplo","troppo","trovato","tu","tua","tue","tuo","tuoi","tutta","tuttavia","tutte","tutti","tutto","uguali","ulteriore","ultimo","un","una","uno","uomo","va","vai","vale","vari","varia","varie","vario","verso","vi","vicino","visto","vita","voi","volta","volte","vostra","vostre","vostri","vostro","è"]


def remove_stopwords(text, stop_words):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def get_model(model_name="llava"):
    if model_name == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(LLAVA_ID, torch_dtype=torch.bfloat16).to(DEVICE)
        processor = AutoProcessor.from_pretrained(LLAVA_ID)
        tokenizer = processor.tokenizer
    elif model_name == "mistral":
        model = AutoModelForCausalLM.from_pretrained(MISTRAL_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN")).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_ID, token=os.getenv("MY_HF_TOKEN"))
        tokenizer.pad_token = tokenizer.eos_token
        processor = None
    elif model_name == "nemo":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(NEMO_ID, torch_dtype=torch.bfloat16, token=os.getenv("MY_HF_TOKEN"), quantization_config=quantization_config) #.to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(NEMO_ID, token=os.getenv("MY_HF_TOKEN"))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        processor = None
    
    return model, tokenizer, processor


def complete_generation(model, prompt, new_token, tokenizer, max_length=100, device="cuda:0", stop_string="\n*"):
    prompt += tokenizer.decode(new_token)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if "nemo" in model.name_or_path.lower():
        inputs.pop("token_type_ids")
    
    generated_ids = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        temperature=0.0,
        top_k=50,
        stop_strings=stop_string,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
        )
    generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    selected = generated_string.split(" \n*")[-1].strip("*\n")
    return selected, generated_ids


def batch_complete_generation(model, prompt, next_tokens, tokenizer, max_length=100, padding=True, device="cuda:0", stop_string="\n*"):
    _p = tokenizer([prompt + tokenizer.decode(tok) for tok in next_tokens], padding=padding, return_tensors="pt").to(device)

    if "nemo" in model.name_or_path.lower():
        _p.pop("token_type_ids")

    generated_ids = model.generate(
        **_p,
        # max_length=max_length,
        max_new_tokens=5,
        do_sample=False,
        temperature=0.0,
        top_k=50,
        stop_strings=stop_string,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
        )
    generated_strings = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    selected = [s.split(":", 2)[-1].strip() for s in generated_strings]
    return selected, generated_ids


def search_exemplars(top_tokens, output_distribution, clean_output, human_list):
    tok_to_human_exemplar = {}
    missing_exemplars = human_list.copy()
    found_pos = []
    human_to_model = {exemplar: [] for exemplar in human_list}
    output_distribution = sorted(output_distribution.tolist(), reverse=True)

    for i, k in enumerate(top_tokens):
        llm_exemplar = clean_output[i]

        for human_exemplar in human_list:
            if human_exemplar in llm_exemplar.lower():
                tok_to_human_exemplar[int(k)] = f"{llm_exemplar.rstrip()[:20]} (pos={i+1})"
                human_to_model[human_exemplar].append((llm_exemplar, i, output_distribution[i]))
                if human_exemplar in missing_exemplars:
                    missing_exemplars.remove(human_exemplar)
                found_pos.append(i)
                continue
            if len(set(human_exemplar.split(" ")).intersection(set(llm_exemplar.split(" ")))) > 0:
                tok_to_human_exemplar[int(k)] = f"{llm_exemplar.rstrip()[:20]} (pos={i+1})"
                human_to_model[human_exemplar].append((llm_exemplar, i, output_distribution[i]))
                if human_exemplar in missing_exemplars:
                    missing_exemplars.remove(human_exemplar)
                found_pos.append(i)
    
    return tok_to_human_exemplar, missing_exemplars, found_pos, human_to_model


def get_blacklisted(tokenizer, min_len=2):
    blacklisted_tok_indices = []
    for k, v  in tokenizer.vocab.items():
        # if not k.startswith(BPE_START_TOKEN) or len(k) <= 2:
        # if not k.startswith(BPE_START_TOKEN) or len(k) <= 2 or k[1].isupper():
        # if len(k) <= min_len or k[0].isupper():
        if len(k) <= min_len:
            blacklisted_tok_indices.append(v)
            continue
    return blacklisted_tok_indices


def extract_from_model(
    prompt,
    target_list,
    model,
    processor,
    tokenizer,
    top_k=100,
    out_max_length=150,
    concept_image=None,
    blacklisted_indices=None
    ):
    human_targets   = " ".join(target_list)
    
    if processor is not None:
        inputs = processor(text=prompt, images=concept_image, return_tensors="pt").to(DEVICE)
        # targets = processor(text=human_targets, images=None, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # targets = tokenizer(human_targets, return_tensors="pt")
    
    if "nemo" in model.name_or_path.lower():
        inputs.pop("token_type_ids")

    with torch.no_grad():    
        logits = model(**inputs).logits
        stored_logits = logits[:, -1, :].clone().flatten()

    if blacklisted_indices is not None:
        stored_logits[blacklisted_indices] = 0.0                                      # NOTE  here we blacklist output tokens!

    output_distribution = torch.nn.functional.softmax(stored_logits, dim=-1).cpu().detach()
    top_k_tokens = output_distribution.argsort(descending=True)[:top_k].numpy()

    # # NON-BATCHED EXTRACTION ----------------------------------------------
    # completed_labels = []
    # completed_toks = []
    # for tok in top_k_tokens:
    #     _c_labels, _c_toks = complete_generation(model=model, prompt=prompt, new_token=tok, tokenizer=tokenizer, device=DEVICE, max_length=out_max_length)
    #     completed_labels.append(_c_labels)
    #     completed_toks.append(_c_toks.cpu())

    ## BATCHED EXTRACTION ----------------------------------------------
    completed_labels, completed_toks = batch_complete_generation(
        model=model,
        prompt=prompt,
        next_tokens=top_k_tokens,
        tokenizer=tokenizer,
        padding=True,
        device=DEVICE,
        max_length=out_max_length
        )
    
    clean_completed_labels = [c.strip().replace("\n", "").lstrip(" *").split("*")[0][:30] for i, c in enumerate(completed_labels)]
    completed_labels = [(tokenizer.decode(top_k_tokens[i]), c.lstrip(tokenizer.decode(top_k_tokens[i]))) for i, c in enumerate(clean_completed_labels)]

    formatted_completed_labels = [f"({tokenizer.decode(top_k_tokens[i])}) {c}" for i, c in enumerate(clean_completed_labels)]

    tok_to_human_exemplar, missing_exemplars, found_pos, human_to_model = search_exemplars(top_tokens=top_k_tokens, output_distribution=output_distribution, clean_output=clean_completed_labels, human_list=target_list)
    if len(tok_to_human_exemplar) == 0:
        human_probs = torch.zeros(len(missing_exemplars)).numpy().flatten()        
    else:
        human_probs = output_distribution[torch.tensor(list(tok_to_human_exemplar.keys()))]
        human_probs = torch.cat((human_probs, torch.zeros(len(missing_exemplars))), dim=-1).cpu().detach().numpy().flatten()
    
    return {
        "output_distribution": output_distribution[top_k_tokens].tolist(),
        "output_exemplars": completed_labels,
        "human_exemplars": target_list,
        # "human_distribution": human_probs.tolist(),
        "matches": human_to_model,
        # "missing_exemplars": missing_exemplars,
    }
    

def main():
    MODEL_NAME          = "nemo"
    MULTIMODAL_INPUT    = False 
    TOP_K               = 250
    REMOVE_BLACKLISTED  = False

    human_df = pd.read_excel(HUMAN_DATAPATH)[["concept", "category", "concept_eng", "exemplar", "count", "dominance"]]
    human_df.concept = human_df.concept.str.lower().str.strip()
    human_df["clean_exemplar"] = [row.exemplar.replace(f"{row.concept.upper()}_", "") for i, row in human_df.iterrows()]

    concept_to_exemplars = {c: human_df[human_df.concept == c].clean_exemplar.tolist() for c in human_df.concept.unique()}
    concept_to_image = {row.concept: f"{IMAGE_BASEPATH}/{row.img}" for i, row in pd.read_csv("../unibo-subordinate/data/unibo-concepts-it-imgs.csv").iterrows()}

    concepts = sorted(concept_to_exemplars.keys())

    model, tokenizer, processor = get_model(model_name=MODEL_NAME)

    if REMOVE_BLACKLISTED:
        blacklisted_tok_indices = get_blacklisted(tokenizer, min_len=2)
        print(f"Number of blacklisted tokens: {len(blacklisted_tok_indices)}")
    else:
        blacklisted_tok_indices = None

    results = {}
    for concept in tqdm(concepts):
        target_list = [remove_stopwords(w, stop_words=ITALIAN_STOPWORDS) for w in concept_to_exemplars[concept]]

        if MODEL_NAME == "llava":
            if MULTIMODAL_INPUT:
                prompt = BASE_PROMPT_IMG.replace("<CONCEPT>", concept)
                concept_image = Image.open(concept_to_image[concept]).convert("RGB")
            else:
                prompt = BASE_PROMPT.replace("<CONCEPT>", concept)
                concept_image = None
        elif MODEL_NAME == "mistral" or MODEL_NAME == "nemo":
            prompt = BASE_PROMPT.replace("<CONCEPT>", concept)
            concept_image = None
               
        results[concept] = extract_from_model(
            model=model,
            prompt=prompt,
            target_list=target_list,
            tokenizer=tokenizer,
            processor=processor,
            top_k=TOP_K,
            out_max_length=150,
            concept_image=concept_image,
            blacklisted_indices=blacklisted_tok_indices
            )
    
    with open(f"{'blacklisted_' if REMOVE_BLACKLISTED else ''}p_{MODEL_NAME}{'_multimodal' if MULTIMODAL_INPUT else ''}_topk{TOP_K}.json", "w") as outfile:
        json.dump(results, outfile, ensure_ascii=False)


if __name__ == "__main__":
    main()
