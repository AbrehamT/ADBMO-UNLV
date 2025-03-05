import requests
import json
import pandas



def fetch_full_text(article_id):
    
    from collections import defaultdict
    summary = defaultdict(str)

    # summary = {
    #     'Title': "",
    #     'Abstract': [],
    #     'Method': []
    # }
    
    response = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{article_id}/unicode")

    if response.text == f'No record can be found for the input: pmc{id}':
        print(f"No record found for article {id}.")

    else:
        try:
            data = json.loads(response.text[1:-1])
            summary['Title'] = data['documents'][0]['passages'][0]['text']
            for text in data['documents'][0]['passages'][:]:
                if text['infons']['section_type'] == 'INTRO':
                    summary['Intro']=(text['text'])                    
                if text['infons']['section_type'] == 'ABSTRACT':
                    summary['Abstract']=(text['text'])
                if text['infons']['section_type'] == 'METHODS':
                    summary['Method']=(text['text'])
                if text['infons']['section_type'] == 'RESULTS':
                    summary['Results']=(text['text'])
                if text['infons']['section_type'] == 'CONCL':
                    summary['Conclusion']=(text['text'])
                if text['infons']['section_type'] == 'REF':
                    summary['References']=(text['text'])
        except json.JSONDecodeError as e:
            print("Json error for string: ", response.text)
            summary['Title'] = "ARTICLE NOT FOUND"
    
    return summary

from collections import defaultdict
import time


def main():
    print("In main")
    uid_list = set()
    with open('uid_list2.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 8 and line[0] == 1:
                uid_list.add(line[:8])
                uid_list.add(line[8:])
            uid_list.add(line)

    full_text = {}
    for i, uid in enumerate(uid_list):
        full_text[uid] = fetch_full_text(uid)
        time.sleep(1)
        print(i)

    with open('full_text.json', 'a') as f:
        json.dump(full_text, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()