import requests
import os
import json
import time

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
# https://www.ncbi.nlm.nih.gov/research/pubtator3-api
# Alzheimer's related entities
entities = [
    "APP", "PSEN1", "PSEN2", "APOE", "MAPT", "BACE1", "TREM2", "CLU",
    "ABCA7", "SORL1", "BIN1", "PICALM", "8-OHdG", "Acetylcholine", "Acetylcholinesterase inhibitor", "Amyloid-beta",
    "Aβ38", "p-tau181", "p-tau217", "p-tau205", "Neurofilament light",
    "IL-6", "IL-7", "IL-12P70", "BDNF", "HSV1", "Histone", "DNA methylation",
    "TREM2", "Neurodegeneration marker", "Triggering receptor expressed on myeloid cells 2",
    "Phosphorylated tau", "Phosphorylated tau/total tau ratio",
    "Neurofilament", "Apparent fibre density"
]

output_dir = "/home/tadesa1/research/ADBMO-UNLV/data/pubtator_articles"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'Content-Type': 'application/json'
}

def get_entity_ids(entity):
    response = requests.get(f"{BASE_URL}/entity/autocomplete?query={entity}")
    if response.status_code == 200:
        return [item['_id'] for item in response.json()]
    return []

def get_pmids(entity_id):
    response = requests.get(f"{BASE_URL}/search?text={entity_id}&size=100")
    if response.status_code == 200:

        res = [item['pmid'] for item in response.json()['results']]
    return res
def get_articles(pmids):
    response = requests.post(
        f"{BASE_URL}/publications/export/biocjson?pmids={pmids[0]}")
    if response.status_code == 200:
        return response.json()
    return {}

for entity in entities:
    print(f"Processing Entity: {entity}")
    entity_ids = get_entity_ids(entity)

    for entity_id in entity_ids:
        print(f" -> Entity ID: {entity_id}")
        pmids = get_pmids(entity_id)

        print(f"   -> Retrieved {len(pmids)} PMIDs")

        # Batch the pmids if needed (optional step depending on limits)
        for pmid in pmids:
            print(f"     -> Downloading PMID: {pmid}")
            article = get_articles([pmid])

            with open(os.path.join(output_dir, f"{pmid}.json"), "w") as f:
                json.dump(article, f, indent=2)

            # time.sleep(1)  # Be nice to the API


print("Done!")
