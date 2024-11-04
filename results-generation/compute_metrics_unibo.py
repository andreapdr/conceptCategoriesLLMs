"""ported from GPT_exemplars.rmd"""
import math
import pandas as pd

from collections import Counter

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


def exemplar_dominance(data, concept="abete"):
    """
    Exemplar dominance: the proportion of the participants who produced 
    a determinate exemplar in response to a given concept;
    """
    concept_data = data[data.concept == concept]

    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])

    n_participants = len(concept_data.participant.unique())
    exemplar_count = Counter(concept_data.concept_exemplar)

    concept_dominance = dict(sorted({k: round(v/n_participants, 4) for k,v in exemplar_count.items()}.items(), key=lambda x: x[-1], reverse=True))
    
    return concept_dominance


def mean_rank_order(data, concept="abete"):
    """
    Mean rank order: mean output position of each exemplar for the corresponding concept; 
    """
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])
    
    concept_avg_ranks = {exemplar: round(concept_data[concept_data.concept_exemplar == exemplar].rank_order.mean(), 4) for exemplar in concept_data.concept_exemplar}

    return dict(sorted(concept_avg_ranks.items(), key=lambda x: x[-1], reverse=False))


def first_occurrence_value(data, concept="abete"):
    """
    First occurrence value: the proportion of participants who produced a given exemplar as their first response; 
    """
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])

    n_participants = len(concept_data.participant.unique())
    
    concept_avg_first_rank = {exemplar: round(len(concept_data[(concept_data.concept_exemplar == exemplar) & (concept_data.rank_order == 1.0)]) / n_participants, 4) for exemplar in concept_data.concept_exemplar.unique()}

    return concept_avg_first_rank


def exemplar_availability(data, ranks, concept="abete"):
    """
    Exemplar availability: a measure that represents the readiness with which an exemplar is produced as a member
    of a given concept, by taking into account its position in a given concept for each participant, its production
    frequency within a concept, the lowest position in which it was produced across participants, and the total
    number of participants who responded with it. 
    """

    def availability(p, n, fpi, N):
        """      
        p:      rank within an exemplar list
        n:      max_rank (i.e., index of its latest production)
        fpi:    freq_rank (number of occurrences across different productions)
        N:      # participant (number of lists)
        """      
        try:     
            c = math.exp(-2.3*((p-1)/(n-1)))
        except ZeroDivisionError:
            c = math.exp(-2.3*(0))
        z = (fpi/N*1.0) * c
        return z 
    
    concept_data = data[data.concept == concept]
    # concept_ranks = ranks[ranks.concept_exemplar == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])

    exemplars =  concept_data.concept_exemplar.unique()

    availabilities = {}
    for e in exemplars:
        # fpi_dict = Counter(ranks[ranks.concept_exemplar == e].rank_order.tolist())
        fpi_dict = Counter(ranks[ranks.concept_exemplar == e].rank_order.tolist())
        N = concept_data[concept_data.concept_exemplar == e].participant_concept.item()
        n = concept_data[concept_data.concept_exemplar == e].max_rank.item()
        availabilities[e] = round(sum([availability(p=p, n=n, fpi=fpi, N=N) for p, fpi in fpi_dict.items()]), 5)


    return availabilities


def exemplar_min_rank(data, concept="abete"):
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])
    
    min_ranks = {}
    for concept in concept_data.concept_exemplar.unique():
        ranks = concept_data[concept_data.concept_exemplar == concept].rank_order
        min_ranks[concept] = min(ranks)

    return min_ranks


def exemplar_max_rank(data, concept="abete"):
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])
    
    max_ranks = {}
    for concept in concept_data.concept_exemplar.unique():
        ranks = concept_data[concept_data.concept_exemplar == concept].rank_order
        max_ranks[concept] = max(ranks)

    return max_ranks


def exemplar_count(data, concept="abete"):
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])

    return dict(Counter(concept_data.concept_exemplar))


def participant_per_concept_count(data, concept="abete"):
    concept_data = data[data.concept == concept]
    
    # drop rows where concept_exemplar is NaN
    concept_data = concept_data.dropna(subset=['concept_exemplar'])

    return len(concept_data.participant.unique())


def main(args):
    data = pd.read_csv(args.datapath)
    # data = pd.read_csv("results-generation/italian/formatted/temp_05/llava-it-textual-temp_05_alliters.csv")
    out_f = "italian/stats/"  + args.datapath.split("/")[-1].replace(".csv", ".stats.csv")
    data.rename(columns={"rank": "rank_order"}, inplace=True)
    data["concept_exemplar"]  = data.concept.str.upper() + "_" + data.exemplar
    # data["concept_exemplar"] = data.exemplar
    concept2cat = {row.concept: row.category for _, row in data[["category", "concept"]].drop_duplicates().iterrows()}
    raw_data = data

    # drop rows where we have NaN concepts
    raw_data = raw_data.dropna(subset=['concept'])

    all_concepts = sorted(raw_data.concept.unique())
    
    h_min_rank = {c: exemplar_min_rank(raw_data, concept=c) for c in all_concepts}
    h_max_rank = {c: exemplar_max_rank(raw_data, concept=c) for c in all_concepts}
    h_count = {c: exemplar_count(raw_data, concept=c) for c in all_concepts} 
    h_participant_per_concept = {c: participant_per_concept_count(raw_data, concept=c) for c in all_concepts}

    h_exemplar_dominance = {c: exemplar_dominance(raw_data, concept=c) for c in all_concepts}
    h_mean_rank_order = {c: mean_rank_order(raw_data, concept=c) for c in all_concepts}
    h_first_occurrence_value = {c: first_occurrence_value(raw_data, concept=c) for c in all_concepts}
    
    data = []
    for concept, exemplars_availability in h_count.items():
        for exemplar, count in exemplars_availability.items():
            data.append({
                "category":             concept2cat[concept],
                "category_ita":         CATEGORY_MAPPER_ITA[concept2cat[concept]],
                "concept":              concept,                            # e.g., 'nave'
                "concept_exemplar":     exemplar,                           # e.g., 'NAVE_da crociera' -> to avoid collision bwtween same exemplars for different concepts (e.g., "PIZZA margherita" e "FIORE margherita")
                "exemplar":             exemplar.split("_")[-1].replace(concept + " ", "").lstrip().rstrip() if exemplar.lower() != f"{concept.lower()}_{concept.lower()}" else exemplar.split("_")[-1],            # e.g., 'da crociera'
                "count":                count,
                "min_rank":             h_min_rank[concept][exemplar],
                "max_rank":             h_max_rank[concept][exemplar],
                "mean_rank":            h_mean_rank_order[concept][exemplar],
                "first_occurrence":     h_first_occurrence_value[concept][exemplar],
                "dominance":            h_exemplar_dominance[concept][exemplar],
                "participant_concept":  h_participant_per_concept[concept],
                })

    df = pd.DataFrame(data)

    h_availability = {c: exemplar_availability(df, ranks=raw_data[["concept_exemplar", "rank_order"]], concept=c) for c in all_concepts}

    availability_data = []
    for concept, exemplars_availability in h_availability.items():
        for concept_exemplar, availabilty in exemplars_availability.items():
            availability_data.append({
                "concept":  concept,
                "concept_exemplar": concept_exemplar,
                "availability":     availabilty
            })
    
    df_availability = pd.DataFrame(availability_data)
    df = df.merge(df_availability, on=["concept", "concept_exemplar"])
    # df.to_csv("llava-it-visual-t10-statistics.csv", index=False)
    df.to_csv(out_f, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", required=True, type=str)
    args = parser.parse_args()
    main(args)
