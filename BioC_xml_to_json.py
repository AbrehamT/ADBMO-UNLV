import xml.etree.ElementTree as ET
from collections import defaultdict
import os

def clean_text(text):    
    if text is None:
        return None
    try:
        # Decode once
        fixed = text.encode('latin1').decode('utf-8')
        # If it's still broken, do it again (optional)
        if 'â' in fixed:
            fixed = fixed.encode('latin1').decode('utf-8')
        return fixed
    except:
        return text

def parse_xml_to_json(file_path):

    if os.stat(file_path).st_size == 0:
        print(f"[WARNING] Skipping empty file: {file_path}")
        return None
    result_list = []
    article_as_json = defaultdict(dict)

    # Safe utf-8 read
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ET.parse(f)

    root = tree.getroot()

    # Grab PMID
    pmid = root.findtext(".//infon[@key='article-id_pmid']") or "123456"

    article_as_json[pmid] = {}

    # Iterate through passages
    for passage in root.findall(".//passage"):
        passage_type = passage.findtext(".//infon[@key='section_type']")

        if passage_type:
            if passage_type not in article_as_json[pmid]:
                article_as_json[pmid][passage_type] = {
                    'text': [],
                    'annotation': []
                }

            text = passage.findtext('.//text')
            if text:
                article_as_json[pmid][passage_type]['text'].append(clean_text(text))

            # Handle annotations
            annotations = passage.findall('.//annotation')
            ann_dict = {}

            for ann in annotations:
                ann_id = ann.get('id')
                if not ann_id:
                    continue

                ann_dict[ann_id] = {
                    'identifier': clean_text(ann.findtext(".//infon[@key='identifier']")),
                    'type': clean_text(ann.findtext(".//infon[@key='type']")),
                    'offset': ann.find('.//location').get('offset') if ann.find('.//location') is not None else None,
                    'length': ann.find('.//location').get('length') if ann.find('.//location') is not None else None,
                    'text': clean_text(ann.findtext('.//text'))
                }

            if ann_dict:
                article_as_json[pmid][passage_type]['annotation'].append(ann_dict)

        else:
            article_as_json[pmid]['NULL-ART'] = 'NULL_TEMP'

    result_list.append(article_as_json)

    return result_list


import os
import json
from collections import defaultdict

Gnorm_tagged_path = "/home/tadesa1/research/GNormPlusJava/output"
tagger_one_tagged_path = "/home/tadesa1/research/TaggerOne-0.2.1/TaggerOne-0.2.1/tagged"


for root, dirs, files in os.walk(Gnorm_tagged_path):
    for file in files:
        res = parse_xml_to_json(f"{Gnorm_tagged_path}/{file}")
        res = dict(res[0])
        print(file)
        with open("gene_species_tagged_articles.json", 'a') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

for root, dirs, files in os.walk(tagger_one_tagged_path):
    for file in files:
        res = parse_xml_to_json(f"{tagger_one_tagged_path}/{file}")
        if res is None: continue
        res = dict(res[0])
        print(file)
        with open("disease_tagged_articles.json", 'a') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)