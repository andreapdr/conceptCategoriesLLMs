import pandas as pd
import json
import os


def load_results(basedir="results-ex2cat"):
    res_name_llama31 = "llama3.1_results.json"
    res_name_llama32 = "llama3.2_results.json"
    res_name_llava = "llava_results.json"
    res_name_mistral = "mistral_results.json"
    res_name_mixtral = "mixtral_results.json"
    res_name_nemo = "nemo_results.json"
    res_name_idefics2 = "idefics2_results.json"
    res_name_llama31_70b = "llama3.1-70b_results.json"

    llama31_results =       json.load(open(os.path.join(basedir, res_name_llama31)))
    llama32_results =       json.load(open(os.path.join(basedir, res_name_llama32)))
    llama31_70b_results =   json.load(open(os.path.join(basedir, res_name_llama31_70b)))
    nemo_results =          json.load(open(os.path.join(basedir, res_name_nemo)))
    mistral_results =       json.load(open(os.path.join(basedir, res_name_mistral)))
    mixtral_results =       json.load(open(os.path.join(basedir, res_name_mixtral)))
    llava_results =         json.load(open(os.path.join(basedir, res_name_llava)))
    idefics2_results =      json.load(open(os.path.join(basedir, res_name_idefics2)))

    data_dict = {
        "llama-3.1-70b": llama31_70b_results,
        "llama-3.1-8b": llama31_results,
        "llama-3.2-3b": llama32_results,
        "mixtral-8x7b": mixtral_results,
        "mistral-7b": mistral_results,
        "nemo-12b": nemo_results,
        "llava-v1.6-mistral-7b": llava_results,
        # "llava-v1.6-mistral-7b-img": None,
        "idefics2-8b": idefics2_results,
        # "idefics2-8b-img": data_idefics2_img,
    }

    return data_dict 


def main():
    results_dict = load_results(basedir="results-ex2cat/results-final")
    models = ['llama-3.1-70b', 'llama-3.1-8b', 'llama-3.2-3b', 'mixtral-8x7b', 'mistral-7b', 'nemo-12b', 'llava-v1.6-mistral-7b', 'idefics2-8b']

    df_cat = pd.DataFrame(columns=["concept", "category"] + models)
    df_concept = pd.DataFrame(columns=["concept", "category"] + models)
    for i in range(len(results_dict["llama-3.1-70b"])):
        row_cat = {}
        row_concept = {}
        concept = results_dict["llama-3.1-70b"][i]["gt_concept"]
        category = results_dict["llama-3.1-70b"][i]["gt_cat"]
        for model in models:
            results_model = results_dict[model][i]
            assert concept == results_model["gt_concept"]
            assert category == results_model["gt_cat"]

            row_cat[model] = results_model["pred_cat"]
            row_concept[model] = results_model["pred_concept"]

        row_cat["concept"] = concept
        row_cat["category"] = category 
        row_concept["concept"] = concept
        row_concept["category"] = category 

        df_concept.loc[len(df_concept)] = row_concept
        df_cat.loc[len(df_cat)] = row_cat
    
    outdir = "results-ex2cat"
    df_concept_outname = "concept_preds.giulia.csv"
    df_cat_outname = "category_preds.giulia.csv"
    
    df_concept.to_csv(f"{outdir}/{df_concept_outname}", index=False)
    df_cat.to_csv(f"{outdir}/{df_cat_outname}", index=False)
    
    for model in models:
        df_cat[model] = df_cat.category == df_cat[model]
        df_concept[model] = df_concept.concept == df_concept[model]

    df_cat = df_cat[models].sum() / len(df_cat)
    df_concept = df_concept[models].sum() / len(df_concept)

    df_res_concept = "concept_res.giulia.csv"
    df_res_cat = "category_res.giulia.csv"

    df_concept.to_csv(f"{outdir}/{df_res_concept}", index=True)
    df_cat.to_csv(f"{outdir}/{df_res_cat}", index=True)


if __name__ == "__main__":
    main()