import json
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

import itertools


def my_shuffle(selected, llm_selected, mode="easy", concept=None, human_data=None, category=None):
    """
    easy:       reverse order list
    medium:     reverse first/last
    hard:       reverse mid elements
    """
    if mode == "easy":
        return selected[::-1], None
    
    elif mode == "medium":
        if len(selected) > 1:
            return [selected[-1]] + selected[1:-1] + [selected[0]], None
        else:
            return selected, None  # No swap needed if list has 1 or 0 elements

    elif mode == "hard":
        if len(selected) > 2:
            return [selected[0]] + selected[1:-1][::-1] + [selected[-1]], None
        else:
            return selected, None  # No middle to reverse if list has less than 3 elements
    
    elif mode == "llm":
        return llm_selected, None

    elif mode == "semantic":
        alternative_concepts = human_data[human_data.category == category.upper()].concept.unique().tolist()
        alternative_concepts.remove(concept.upper())
        fake_concept = random.sample(alternative_concepts, 1)[0].lower()
        fake_exemplar = human_data[human_data.concept == fake_concept.upper()].sort_values(by="availability", ascending=False).iloc[0].exemplar
        fake_exemplar = fake_exemplar.split("_")[-1]

        # replace first occurrence of ground truth list with fake exemplar
        # selected[0] = fake_exemplar
        selected = [fake_exemplar] + selected[1:]
        return selected, fake_concept
    
    elif mode == "order":
        return my_shuffle(selected, llm_selected, mode="easy")
    else:
        raise ValueError("Mode must be 'easy', 'medium', or 'hard'")


def create_candidates(concept, data, concept2img_map, data_llm=None, remove_concept=True, verbose=False, n_alternatives=3, threshold=10, sorting_mode="avail"):
    concept_data = {
        "concept":      concept.lower(),
        "alt_concept":  None,
        "category":     None,
        "llm_name":     None,
        "data": {
            "candidates":   {"order": None, "semantic": None, "llm": None},
            "answers":      {"order": None, "semantic": None, "llm": None}
            },
    }

    if sorting_mode == "avgrank":
        selected = data[data.concept == concept].sort_values("avg_rank_order")
    elif sorting_mode == "avail":
        selected = data[data.concept == concept].sort_values("availability", ascending=False)
    category = selected.iloc[0].category.lower()
    concept_data["category"] = category
    concept_data["img"] = concept2img_map[concept.lower()] 
    concept_data["sorting_mode"] = sorting_mode
    concept_data["llm_name"] = "gpt-3.5"
    selected = selected.exemplar.to_list()
    if verbose: print(f"{selected=}")

    # remove concept
    if remove_concept:
        selected = [elem.split("_")[-1] for elem in selected]
        if verbose: print(f"{selected=}")
    
    for diff in ["order", "semantic", "llm"]:
        ground_truth = selected[:threshold]    
        selected_llm = data_llm[data_llm.concept == concept.lower()].sort_values("normed_rank_order", ascending=False)
        selected_llm = selected_llm.exemplar.to_list()
        selected_llm = [str(elem).replace(concept.lower(), "").lstrip() for elem in selected_llm][:len(ground_truth)]

        foil, alt_concept = my_shuffle(ground_truth, selected_llm, mode=diff, concept=concept, category=category, human_data=data)

        if diff == "semantic":
            concept_data["alt_concept"] = alt_concept

        candidates = [ground_truth, foil]
        random.shuffle(candidates)
        gt_index = candidates.index(ground_truth)

        concept_data["data"]["candidates"][diff] = candidates 
        concept_data["data"]["answers"][diff] = gt_index

    return concept_data


def main():
    from tqdm import tqdm
    
    SORTING_MODE = "avail"
    
    file_human_exemplar = "data/best_human_exemplars.xlsx"
    file_gpt_exemplar = "data/gpt/temperature00/data_exemplars_gpt3.5_temp0_withfreqs.csv"
    # file_human_exemplar = "data/subset_human_exemplars.xlsx" 

    human_data = pd.read_excel(file_human_exemplar)
    llm_data = pd.read_csv(file_gpt_exemplar)
    # human_data = pd.read_excel(file_human_exemplar_subset)
    concepts = human_data.concept.unique()
    
    concept2img_map = pd.read_csv("data/unibo-concepts-it-imgs.csv")
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

    with open(f"dataset/semantic/binary_dataset.json", "w") as f:
        json.dump(sample_short_dataset, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()

# TODO add image name automatically ...