import os
import pandas as pd


def main():
    data = pd.read_csv("results-prototipicality/prototypicality.all.andrea.csv")
    data.concept = data.concept.str.lower()
    concepts = data.concept.str.lower().unique()
    concept2cat = {row.concept.lower(): row.category.lower() for i, row in data[["concept", "category"]].drop_duplicates().iterrows()}
    models = ['llama3.1', 'llama3.2', 'mistral', 'llava', 'idefics2', 'nemo', 'llama3.1-70b', 'mixtral']


    df_acc = pd.DataFrame(columns=["concept", "category", "classif", "abs_diff_avail"] + models)
    for concept in concepts:
        selected = data[data.concept == concept]
        upper = selected.iloc[0][models]
        lower = selected.iloc[-1][models]
        res = (upper <= lower).astype(int).to_dict()
        res["concept"] = concept
        res["category"] = concept2cat[concept]
        res["classif"] = selected.iloc[0].classif
        res["abs_diff_avail"] = abs(selected.iloc[0].availability - selected.iloc[-1].availability)

        df_acc.loc[len(df_acc)] = res
    
    # print((df_acc[models].sum() / len(df_acc)).round(2))
    
    df_acc_outname ="acc_results.all.andrea.absdiff_avail.csv" 
    
    df_acc.to_csv(f"results-prototipicality/{df_acc_outname}", index=False)
    
    # os.makedirs("results-prototipicality/concept_corr/", exist_ok=True)
    # for concept in concepts:
    #     df_corr_outname = f"corr_{concept}.all.andrea.csv"

    #     selected = data[data.concept == concept][["availability"] + models]
    #     df_corr = selected.corr()
    #     df_corr.to_csv(f"results-prototipicality/concept_corr/{df_corr_outname}", index=False)
    

    for k, mode in data.groupby("classif"):
        df_corr_outname = f"corr_{k}.csv"
        
        selected = mode[["availability"] + models]
        df_corr = selected.corr()
        df_corr.to_csv(f"results-prototipicality/{df_corr_outname}", index=False)


def compute_prototipicality(upper, lower):
    if upper <= lower:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()