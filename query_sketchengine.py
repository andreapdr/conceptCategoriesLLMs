import os
import csv
import pandas as pd
import requests
import time
from dotenv import load_dotenv
load_dotenv(".secrets.env")

from tqdm import tqdm

"""
The FUP limit applies to the number of requests (not to the volume of data transferred) per one minute, one hour and one day and defaulting to 100, 900 and 2,000 requests, respectivel
"""


API_DAY_LIMIT = 2000

base_url = 'https://api.sketchengine.eu/bonito/run.cgi'
corpus = 'preloaded/ittenten20_fl1'


def get_freqs(expression, cname, username, api_key):
    page = 'first'
    databnc = {
            'corpname': cname,
            'format': 'json',
            'iquery': expression,
            'tab': 'basic',
    }
    url = base_url + '/%s?corpname=%s' % (page, databnc['corpname'])

    response = requests.get(url, params=databnc, auth=(username, api_key))
    
    if response.status_code == 200:
        data = response.json()
        abs_freq = data.get('fullsize', 0)
        rel_freq = data.get('relsize', 0)
        return abs_freq, rel_freq
    else:
        response.raise_for_status()


def get_user_data(user):
    USERNAME = os.getenv(f"USERNAME_{user.upper()}")
    API_KEY = os.getenv(f"API_KEY_{user.upper()}")
    return USERNAME, API_KEY


def main(args):
    _temp = "temp_" + str(args.temp).replace(".", "")

    assert args.modality in ["visual", "textual"]

    USERNAME, API_KEY = get_user_data(args.user)

    exemplars_f = f"results-generation/italian/formatted/{_temp}/{args.model}-it-{args.modality}-{_temp}_alliters.csv"
    all_freqs_f = "results-generation/freqs/all/all-freqs-sketchengine.csv"
    out_f = f"results-generation/freqs/{args.model}-{args.modality}/{_temp}"
    out_all_f = f"results-generation/freqs/all/all-freqs-sketchengine-{args.user}.csv"

    os.makedirs(out_f, exist_ok=True)

    exemplars = pd.read_csv(exemplars_f)
    all_freqs = pd.read_csv(all_freqs_f)
    all_freqs_exemplars = all_freqs.exemplar.tolist()

    if args.resume:
        print(f"- resuming from {out_f}/freqs.csv !")
        model_freqs = pd.read_csv(f"{out_f}/freqs.csv") 
    else:
        model_freqs = pd.DataFrame(columns=["exemplar", "abs_freq", "rel_freq"])

    # remove multiple occurrences of the same generated exemplar
    exemplars = exemplars.drop_duplicates(subset="exemplar")
    # print(exemplars.head())

    api_hits = 0
    for i, row in tqdm(exemplars.iterrows(), total=exemplars.shape[0]):
        exemplar = row.exemplar
        if exemplar in all_freqs_exemplars:
            match_row = all_freqs[all_freqs.exemplar == exemplar].iloc[0]
            abs_freq = match_row.abs_freq
            rel_freq = match_row.rel_freq
            model_freqs.loc[len(model_freqs)] = [exemplar, abs_freq, rel_freq]
        else:
            time.sleep(0.25)
            abs_freq, rel_freq = get_freqs(expression=exemplar, cname=corpus, username=USERNAME, api_key=API_KEY)
            model_freqs.loc[len(model_freqs)] = [exemplar, abs_freq, rel_freq]
            api_hits += 1
            model_freqs.to_csv(os.path.join(out_f, "freqs.csv"), index=False)
            all_freqs = pd.concat([all_freqs, model_freqs]).drop_duplicates(subset=["exemplar"])
            all_freqs.to_csv(out_all_f, index=False)
        if api_hits >= API_DAY_LIMIT:
            print(f"+++ Reached API DAY LIMIT ({API_DAY_LIMIT}) with user: {USERNAME}")
            time.sleep(86400)   # way 24h and reset hits
            api_hits = 0

    all_freqs = pd.concat([all_freqs, model_freqs]).drop_duplicates(subset=["exemplar"])
    
    print(f"saving {args.model} results in {out_f}/freqs.csv")
    print(f"saving aggregated results in {out_all_f}")
    model_freqs.to_csv(os.path.join(out_f, "freqs.csv"), index=False)
    all_freqs.to_csv(out_all_f, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["idefics2", "llama3.1", "llama3.1-70b", "nemo", "llava", "llama3.2", "mistral", "mixtral"])
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--modality", type=str, default="textual")
    parser.add_argument("--user", type=str, default="andrea")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
