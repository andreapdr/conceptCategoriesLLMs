# compute the correlation between the number of exemplars produced by human subjects and those produced by LLMs

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
    data_llava_img =    pd.read_csv(os.path.join(basedir, f"llava-it-visual-{_temp}_alliters.stats.csv"))
    data_idefics2 =     pd.read_csv(os.path.join(basedir, f"idefics2-it-textual-{_temp}_alliters.stats.csv"))
    data_idefics2_img = pd.read_csv(os.path.join(basedir, f"idefics2-it-visual-{_temp}_alliters.stats.csv"))
    
    data_dict = {
        "human": human_data,
        "llama-3.1-70b": data_llama31_70b,
        "llama-3.1-8b": data_llama31,
        "llama-3.2-3b": data_llama32,
        "mixtral-8x7b": data_mixtral,
        "mistral-7b": data_mistral,
        "nemo-12b": data_nemo,
        "llava-v1.6-mistral-7b": data_llava,
        "llava-v1.6-mistral-7b-img": data_llava_img,
        "idefics2-8b": data_idefics2,
        "idefics2-8b-img": data_idefics2_img,
        }

    return data_dict


def get_mangitudes(human_data, llm_data, concept, filter):
    selected_h = human_data[human_data.concept == concept.upper()]
    selected_llm = llm_data[llm_data.concept == concept]
    
    if filter:
        selected_llm = selected_llm[selected_llm["abs_freq"] > 0]
    
    len_h = len(selected_h)
    len_llm = len(selected_llm)

    return len_h, len_llm


def main(args):
    data_dict = load_datadict(args.temp)

    if args.filter:
        print("- filtering results according to exemplars freqs from SketchEngine!")

    concepts = data_dict["human"].concept.str.lower().unique()
    concept2cat = {row.concept.lower(): row.category.lower() for i, row in data_dict["human"][["concept", "category"]].drop_duplicates().iterrows()}
    models = [m for m in list(data_dict.keys()) if m != "human"]

    magnitude_df = pd.DataFrame(columns=["category", "concept", "human"] +  models)
    for concept in concepts:
        row_results = {}
        for model in models:
            len_h, len_llm = get_mangitudes(data_dict["human"], data_dict[model], concept, filter=args.filter)
            row_results[model] = len_llm
        row_results["category"] = concept2cat[concept]
        row_results["concept"] = concept
        row_results["human"] = len_h
        
        magnitude_df.loc[len(magnitude_df)] = row_results
    
    corr_df = magnitude_df.corr(numeric_only=True)
    corr_cat_dict = {cat: v.corr(numeric_only=True) for cat, v in magnitude_df.groupby("category")} 

    corr_df_outname = f"res_all_t{str(args.temp).replace('.', '')}.csv"
    if args.filter:
        corr_df_outname = corr_df_outname.replace(".csv", ".filter.csv")
    corr_df.to_csv(f"results-magnitude/{corr_df_outname}", index=False)

    os.makedirs("results-magnitude/category", exist_ok=True)
    for cat, res in corr_cat_dict.items():
        cat = cat.replace("/", "-")
        cat_df_outname = f"res_{cat}_t{str(args.temp).replace('.', '')}.csv"
        if args.filter:
            cat_df_outname = cat_df_outname.replace(".csv", ".filter.csv")
        cat_df_outname = cat_df_outname.replace(" ", "_")
        res.to_csv(f"results-magnitude/category/{cat_df_outname}", index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--filter", action="store_true")
    args = parser.parse_args()
    main(args)