import csv
import pandas as pd
import requests
import time

from tqdm import tqdm


USERNAME = 'andrea.pedrotti'
API_KEY = '10946b2a5d6e1307320873b0dacdd973'

# USERNAME = 'giulia.rambelli4'
# API_KEY = 'a07d33b5d17643f736bff204acbf5750'

# USERNAME = 'caterina.villani6'
# API_KEY = 'f60a89d26e4982674bf65948c5d1f442'

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


def query_sketchengine(data, corpus, username, api_key, sleeptime, outfile):
    API_LIMIT = 500
    api_calls = 0

    try:
        existing_data = pd.read_csv(outfile)
        already_queried = set(existing_data['exemplar'])
    except FileNotFoundError:
        already_queried = set()

    # Open the output file in append mode
    with open(outfile, 'a', newline='', buffering=1) as csvfile:
        fieldnames = ['exemplar', 'abs_freq', 'rel_freq']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header if the file was empty or new
        if csvfile.tell() == 0:
            writer.writeheader()

        for i, exemplar in enumerate(tqdm(data)):
            if exemplar not in already_queried:
                if pd.isna(exemplar) or exemplar == "":
                    abs_freq = 0
                    rel_freq = 0.0
                else:
                    abs_freq, rel_freq = get_freqs(expression=exemplar, cname=corpus, username=username, api_key=api_key)
                    api_calls += 1
                
                # Append (exemplar, abs_freq, rel_freq) to the output file
                # print(f"writing: {exemplar}")
                writer.writerow({'exemplar': exemplar, 'abs_freq': abs_freq, 'rel_freq': rel_freq})
                csvfile.flush()  # Ensure the data is written to the file
                already_queried.add(exemplar)

                if i % 100 == 0 and i != 0:
                    time.sleep(sleeptime)

    return


def fill_df(df):
    new_freqs = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(row.abs_freq):
            if pd.isna(row.exemplar):
                new_freqs.append(0.0)
            else:
                time.sleep(1)
                abs_freq, _ = get_freqs(expression=row.exemplar, cname="preloaded/ittenten20_fl1", username=USERNAME, api_key=API_KEY)
                new_freqs.append(abs_freq)
        else:
            new_freqs.append(row.abs_freq)
    
    df["abs_freq"] = new_freqs

    return df


if __name__ == "__main__":
    FILEPATH = "data/llava/temperature00/visual-alliters.csv"
    OUTFILE =  "data/freqs-sketchengine.csv" 

    print(f"{FILEPATH=} - {OUTFILE=}")

    df = pd.read_csv(FILEPATH)
    
    missing_freqs = df.abs_freq.isna().sum()
    print(f"{FILEPATH} exemplars with nan freqs: {missing_freqs}")

    query_sketchengine(df.exemplar, corpus, USERNAME, API_KEY, sleeptime=60, outfile=OUTFILE)