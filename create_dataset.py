import json
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

import itertools


def my_shuffle(selected, llm_selected, mode="easy"):
    """
    easy:       reverse order list
    medium:     reverse first/last
    hard:       reverse mid elements
    """
    if mode == "easy":
        return selected[::-1]
    
    elif mode == "medium":
        if len(selected) > 1:
            return [selected[-1]] + selected[1:-1] + [selected[0]]
        else:
            return selected  # No swap needed if list has 1 or 0 elements

    elif mode == "hard":
        if len(selected) > 2:
            return [selected[0]] + selected[1:-1][::-1] + [selected[-1]]
        else:
            return selected  # No middle to reverse if list has less than 3 elements
    
    elif mode == "llm":
        return llm_selected
    
    else:
        raise ValueError("Mode must be 'easy', 'medium', or 'hard'")


def create_candidates(concept, data, concept2img_map, data_llm=None, remove_concept=True, verbose=False, n_alternatives=3, threshold=10, sorting_mode="avail"):
    concept_data = {
        "concept":      concept.lower(),
        "category":     None,
        "llm_name":     None,
        "data": {
            "candidates":   {"easy": None, "medium": None, "hard": None, "llm": None},
            "answers":      {"easy": None, "medium": None, "hard": None, "llm": None}
            },
    }

    if sorting_mode == "avgrank":
        selected = data[data.concept == concept].sort_values("avg_rank_order")
    elif sorting_mode == "avail":
        selected = data[data.concept == concept].sort_values("availability", ascending=False)
    concept_data["category"] = selected.iloc[0].category.lower()
    concept_data["img"] = concept2img_map[concept.lower()] 
    concept_data["sorting_mode"] = sorting_mode
    concept_data["llm_name"] = "gpt-3.5"
    selected = selected.exemplar.to_list()
    if verbose: print(f"{selected=}")

    # remove concept
    if remove_concept:
        selected = [elem.split("_")[-1] for elem in selected]
        if verbose: print(f"{selected=}")
    
    for diff in ["easy", "medium", "hard", "llm"]:
        ground_truth = selected[:threshold]    
        selected_llm = data_llm[data_llm.concept == concept.lower()].sort_values("normed_rank_order", ascending=False)
        selected_llm = selected_llm.exemplar.to_list()
        selected_llm = [str(elem).replace(concept.lower(), "").lstrip() for elem in selected_llm][:len(ground_truth)]

        foil = my_shuffle(ground_truth, selected_llm, mode=diff)
        candidates = [ground_truth, foil]
        random.shuffle(candidates)
        gt_index = candidates.index(ground_truth)

        concept_data["data"]["candidates"][diff] = candidates 
        concept_data["data"]["answers"][diff] = gt_index

    return concept_data


def main():
    from tqdm import tqdm
    
    SORTING_MODE = "avgrank"
    
    file_human_exemplar = "data/best_human_exemplars.xlsx"
    file_gpt_exemplar = "data/gpt/temperature00/data_exemplars_gpt3.5_temp0_withfreqs.csv"
    # file_human_exemplar = "data/subset_human_exemplars.xlsx" 

    human_data = pd.read_excel(file_human_exemplar)
    llm_data = pd.read_csv(file_gpt_exemplar)
    # human_data = pd.read_excel(file_human_exemplar_subset)
    concepts = human_data.concept.unique()
    
    # concept2img_map = 
    concept2img_map = pd.read_csv("unibo-concepts-it-imgs.csv")
    concept2img_map = {r.concept: r.img for _, r in concept2img_map.iterrows()}

    thresholds = [("short", 5), ("medium", 8), ("long", 10)]
    sample_short_dataset = []

    BLACKLISTED_CONCEPTS = ["ficus", "girasole"]
    for concept in tqdm(concepts):
        if concept.lower() in BLACKLISTED_CONCEPTS:
            continue
        if concept == "giubotto":
            concept == "giubbotto"
        sample_short_dataset.append(create_candidates(concept, human_data, concept2img_map, n_alternatives=1, data_llm=llm_data, verbose=False, threshold=10,sorting_mode=SORTING_MODE))

    # with open("dataset/binary_dataset_avgrank_and_llm_upto10.json", "w") as f:
    with open(f"dataset/{SORTING_MODE}/bydiff/binary_dataset_avgrank.json", "w") as f:
        json.dump(sample_short_dataset, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()

# TODO add image name automatically ...