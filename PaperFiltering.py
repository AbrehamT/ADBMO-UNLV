import requests
import json
import time
from collections import defaultdict


def fetch_full_text(article_id):
    response = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{article_id}/unicode")

    summary = {
        'Abstract': [],
        'Method': []
    }

    if response.text == f'No record can be found for the input: pmc{article_id[3:]}':
        print(f"No record found for article {article_id}.")

    else:
        try:
            # print(len(response.text[1:-1]))
            data = json.loads(response.text[1:-1])
            for text in data['documents'][0]['passages'][:]:
                if text['infons']['section_type'] == 'ABSTRACT':
                    summary['Abstract'].append(text['text'])
                if text['infons']['section_type'] == 'METHODS':
                    summary['Method'].append(text['text'])
        except json.JSONDecodeError as e:
            data = json.loads(response.text[e.colno+2:-1])
            for text in data['documents'][0]['passages'][:]:
                if text['infons']['section_type'] == 'ABSTRACT':
                    summary['Abstract'].append(text['text'])
                if text['infons']['section_type'] == 'METHODS':
                    summary['Method'].append(text['text'])

    summary['Abstract'] = '\n'.join(summary['Abstract'])
    summary['Method'] = '\n'.join(summary['Method'])
    return summary

article_info = defaultdict(str)

with open("valid.txt", 'r') as f1:
    with open('valid_articles.txt', 'r') as f2:
        for id, line in zip(f1, f2):
            article_info[id.strip()] = {'class': 1, "title": line.strip()}
        f2.close()
    f1.close()
with open("invalid.txt", 'r') as f1:
    with open('invalid_articles.txt', 'r') as f2:
        for id, line in zip(f1, f2):
            article_info[id.strip()] = {'class': 0, "title": line.strip()}
        f2.close()
    f1.close()

valid_ids = []
invalid_ids = []

with open('valid.txt', 'r') as f:
    valid_ids = [line.strip() for line in f]

with open('invalid.txt', 'r') as f:
    invalid_ids = [line.strip() for line in f]

for id in valid_ids:
    try:
        article_info[id]['summary'] = fetch_full_text(id)
    except TypeError:
        print(article_info[id])
    
for id in invalid_ids:
    if id == '---' :
        continue
    try:
        article_info[id]['summary'] = fetch_full_text(id)
    except TypeError:
        print(article_info[id])


custom_template = """
You are going to be performing classifications on research articles regarding Alzheimer’s disease. Below are the rules on how 
to judge an article as being relevant or not. Please return a single Yes or No in your response.

## 1. Papers Must Be Original Research Articles
    * Metadata Filtering: Use metadata to identify papers labeled "original research" and exclude reviews, perspectives, posters, or preprints.
    * Keyword Identification: Scan sections for phrases like "data were collected" or "we conducted" to confirm original research.
## 2. Papers must have an AD focus, including a population of AD patients (at risk/MCI/AD) and/or looking at Alzheimer’s disease specific biomarkers (amyloid) (Many papers investigating neurodegenerative diseases will mention Alzheimer’s, even if it is not the focus)  
    * Criteria for Selection:
        * AD-Focused: Include papers explicitly studying Alzheimer’s disease (AD) topics like diagnosis, treatment, biomarkers (e.g., amyloid), or pathology.
        * AD Patients: Papers involving AD populations (at risk, MCI, or diagnosed) are relevant, even if AD is not the central focus.
        * Subset Context: Unless biomarkers or pathology are specific to AD, exclude papers focusing broadly on neurodegeneration with AD patients.
        * Biomarker Specificity: Include studies addressing AD biomarkers (e.g., amyloid, tau). Exclude them if they contain general neurodegeneration markers.
## 3.  Human Sample Size Must Be Over 50
    * Criteria for Inclusion:
        * Stated Sample Size: Include papers explicitly reporting a sample size of 50+ for AD patients (at risk, MCI, or diagnosed).
        * Missing Information: Exclude papers without specific sample size details unless other critical criteria (e.g., strong AD focus or biomarker analysis) are met.
## 4.  Must be looking at a protein (no genes, transcripts, or fragments)
    * Keyword Filtering: Use terms like "protein," "amyloid," "tau," or specific AD-related proteins (e.g., "beta-amyloid") to identify relevant studies. Exclude papers mentioning "gene," "RNA," "transcription," or "fragment" as indicators of a non-protein focus.
## 5. Include Fluids from Non-Clinical Models (Exclude Tissue Samples)
    * Fluid Criteria: Focus on animal studies using fluids like cerebrospinal fluid (CSF), blood, serum, or plasma. These fluids often contain biomarkers relevant to AD research.
    * Exclusion of Tissue Samples: Exclude studies involving tissue samples (e.g., brain slices, biopsy samples) using keywords like "tissue," "histology," or "brain slice."
## 6. Exclude “Blood Pressure” When Analyzing “Blood”
    * Keyword Exclusion: Identify "blood" as a relevant biomarker but exclude papers mentioning "blood pressure" (e.g., "blood pressure measurement" or "high blood pressure").
    * Contextual Filtering: Differentiate between "blood" used in biomarker sampling (e.g., "serum analysis") and circulatory assessments like "blood pressure."
    * Pattern Recognition: Exclude studies such as "hypertension study" or "vascular health,".
"""


def filter_request(sys_prompt, user_prompt, model_num = 4):
    generate_path = "http://oceanus.cs.unlv.edu:11434/api/generate"
    models = ["custom-llama3.2:latest","llama3.2:1b", "llama3.2:3b","phi3.5:3.8b", "llama3.1:70b", "medllama2:7b", "mistral:7b"]

    params = {
        "model": models[model_num],
        # "system": custom_template,
        "prompt": user_prompt + sys_prompt ,
        "stream": False
    }

    generate_response = requests.post(generate_path, json=params)
    generate_response = generate_response.json()
    return (generate_response, models[model_num])

def openai_req(sys_prompt, user_prompt):
    from openai import OpenAI 
    import os

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            { "role": 'system', "content": f'{sys_prompt}'},
            { 
                "role": 'user',
                "content": f"{user_prompt}"
            }
        ] 
    )

    return response.choices[0]

# def create_confusion_matrix(articles):
#     from sklearn import metrics
#     import matplotlib.pyplot as plt




def main():
    
    for id in article_info:
        if id == '---':
            continue
        user_question = f"""
            Here's a summary of an article that I want you to classify. Please respond with a yes or a no.
    
            -------------------------------------------------------------------
            Title
            {article_info[id]['title']}
            -------------------------------------------------------------------
    
            Abstract
            {article_info[id]['summary']['Abstract']}
            -------------------------------------------------------------------
    
            Method
            {article_info[id]['summary']['Method']}
    
        """
    
        res = filter_request(sys_prompt=custom_template, user_prompt=user_question, model_num=3)
        openai_res = openai_req(sys_prompt=custom_template, user_prompt=user_question)
        article_info[id]["ollama_model_response"] = res[0]['response']
        article_info[id]["openai_response"] = openai_res.message.content

    with open(f'./classification_results/classification_results_{res[1]}.json', 'w') as f:
        json.dump(article_info, f)


if __name__ == "__main__":
    main()