import os
import pandas as pd

def load_datadict(temp=0.5, basedir="results-generation/italian/stats_wfreqs"):
    _temp = f"temp_{str(temp).replace('.', '')}"

    human_data =        pd.read_csv("data/best_human_exemplars_withstrings.csv", delimiter="\t")

    data_llama31_70b =  pd.read_csv(os.path.join(basedir, f"llama3.1-70b-it-textual-{_temp}_alliters.stats.csv"))
    data_llama31 =      pd.read_csv(os.path.join(basedir, f"llama3.1-it-textual-{_temp}_alliters.stats.csv"))
    data_llama32 =      pd.read_csv(os.path.join(basedir, f"llama3.2-it-textual-{_temp}_alliters.stats.csv"))
    data_mixtral =      pd.read_csv(os.path.join(basedir, f"mixtral-it-textual-{_temp}_alliters.stats.csv"))
    data_mistral =      pd.read_csv(os.path.join(basedir, f"mistral-it-textual-{_temp}_alliters.stats.csv"))
    data_nemo =         pd.read_csv(os.path.join(basedir, f"nemo-it-textual-{_temp}_alliters.stats.csv"))
    data_llava =        pd.read_csv(os.path.join(basedir, f"llava-it-textual-{_temp}_alliters.stats.csv"))
    # data_llava_img =    pd.read_csv(os.path.join(basedir, f"llava-it-visual-{_temp}_alliters.stats.csv"))
    data_idefics2 =     pd.read_csv(os.path.join(basedir, f"idefics2-it-textual-{_temp}_alliters.stats.csv"))
    # data_idefics2_img = pd.read_csv(os.path.join(basedir, f"idefics2-it-visual-{_temp}_alliters.stats.csv"))
    
    data_dict = {
        "human": human_data,
        "llama-3.1-70b": data_llama31_70b,
        "llama-3.1-8b": data_llama31,
        "llama-3.2-3b": data_llama32,
        "mixtral-8x7b": data_mixtral,
        "mistral-7b": data_mistral,
        "nemo-12b": data_nemo,
        "llava-v1.6-mistral-7b": data_llava,
        # "llava-v1.6-mistral-7b-img": data_llava_img,
        "idefics2-8b": data_idefics2,
        # "idefics2-8b-img": data_idefics2_img,
        }

    return data_dict
