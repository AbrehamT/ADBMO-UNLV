import requests
import json
import time
from collections import defaultdict
import re

def fetch_full_text(article_id, title):
    response = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{article_id}/unicode")

    summary = {
        'Abstract': [],
        'Method': []
    }

    if response.text == f'No record can be found for the input: pmc{article_id[3:]}':
        print(f"No record found for article {article_id} titled {title}.")

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
        
        summary['Abstract'] = "\n".join(summary['Abstract'])
        summary['Method'] = "\n".join(summary['Method'])
    return summary

def populate_dict():
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
            article_info[id]['summary'] = fetch_full_text(id, article_info[id]['title'])
        except TypeError:
            print(article_info[id])

    for id in invalid_ids:
        if id == '---' :
            continue
        try:
            article_info[id]['summary'] = fetch_full_text(id, article_info[id]['title'])
        except TypeError:
            print(article_info[id])
    
    return article_info

def filter_request(user_prompt, model_num = 1, few_shot_examples = cot_few_shot_examples):
    generate_path = "http://oceanus.cs.unlv.edu:11434/api/chat"
    # generate_path = "http://127.0.0.1:11434/api/chat"
    models = ["LLAMA3.1:70b-ADBMO-filterer2-generate","LLAMA3.1:70B-ADBMO-filterer-chat", "LLAMA3.2:3b-ADBMO-filterer-chat", "LLAMA3.1:8b-ADBMO-filterer-chat", "ADBMO-filterer-meditron:70b"]

    # params = {
    #     "model": models[model_num],
    #     "prompt": user_prompt,
    #     "stream": False
    # }
    
    params = {
        "model": models[model_num],
        "messages": [
            {
                "role": "user",
                "content": few_shot_examples["user_one"]
            },
            {
                "role": "assistant",
                "content": few_shot_examples["assistant_one"]
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False            
    }

    generate_response = requests.post(generate_path, json=params)
    generate_response = generate_response.json()
    return (generate_response, models[model_num])

sys_prompt = """
You are going to be performing BINARY classifications on research articles regarding Alzheimer’s disease. The positive class for classification is a "relevant paper", while the negative class is an "irrelevant paper". 

There are six factors that you are to use as guidelines for classification. Assign the classification by sequentially evaluating the paper against the following factors. Return the results as a JSON object in the following format:

{
  "criteria": {
    "criterion_1": {"satisfied": true/false, "reason": "Brief explanation of reasoning"},
    "criterion_2": {"satisfied": true/false, "reason": "Brief explanation of reasoning"},
    ...
    "criterion_6": {"satisfied": true/false, "reason": "Brief explanation of reasoning"}
  },
  "final_classification": "relevant/irrelevant",
  "justification": "Overall reasoning for the final classification based on the criteria."
}

STRICTLY RETURN A SINGLE JSON OBJECT EVERY SINGLE TIME WITH NO EXTRA TEXT. 
#1 Is the paper an original research article?
    * Occurrence of terms like "study", "review", "perspectives", "posters" or "preprints" in an article are common indicators that the paper considered is not an original research and therefore must have the negative class of "irrelevant paper".
    * Original research articles often contain some kind of data collection followed by statistical analysis to support or refute a hypothesis. Often these kinds of information are located in the methods section of an article, so be aware of that. Obviously, the presence of these contexts should indicate that the paper is an original research and should influence you to classify it as the positive "relevant paper" class.

#2 Does the paper have Alzheimer's Disease (AD) as its main focus?
    * Papers explicitly studying AD topics like diagnosis, treatment, biomarkers, or pathology should be assigned the positive class of "relevant paper".
    * Papers involving AD patients (i.e., at risk, Mild Cognitive Impairment, AD) should be assigned the positive class of "relevant paper".
    * Biomarker specificity: Include studies addressing AD biomarkers such as Amyloid beta and tau proteins and assign the studies the positive class of "relevant paper". HOWEVER, if the biomarkers contain general neurodegeneration markers assign them the negative class of "irrelevant paper."
    
#3 If the paper mentions some kind of statistical analysis does the sample size exceed 50 (i.e., n >= 50)? 
    * Papers that are research conducted on a human sample of size greater than 50 should be assigned the positive class of "relevant paper".
    * However, this criterion can be relaxed if the paper strongly satisfies other criteria but has a sample size of less than 50.
    
#4 Does the paper look at proteins as biomarkers (not genes, nor transcripts nor fragments)?
    * Keyword Filtering: Use terms like "protein," "amyloid," "tau," or specific AD-related proteins (e.g., "beta-amyloid") to identify relevant studies. Papers mentioning these terms several times and base most of the context on these proteins should be assigned the positive class of "relevant paper."
    * Exclude papers mentioning "gene," "RNA," "transcription," or "fragment" several times and have them as the central context as indicators of a non-protein focus.
    
#5 Does the paper include Fluids from Non-Clinical Models to perform its study?
    * Fluid Criteria: Focus on animal studies using fluids like cerebrospinal fluid (CSF), blood, serum, or plasma. These fluids often contain biomarkers relevant to AD research, and papers satisfying these criteria should be assigned the positive class of "relevant paper."
    * Exclusion of Tissue Samples: Exclude studies involving tissue samples (e.g., brain slices, biopsy samples) using keywords like "tissue," "histology," or "brain slice." Studies involving these samples should be assigned the negative class of "irrelevant paper."
    
#6. If the term Blood occurs in the paper does it use Blood as an AD biomarker?
    * Keyword Exclusion: Identify "blood" as a relevant biomarker but exclude papers mentioning "blood pressure" (e.g., "blood pressure measurement" or "high blood pressure").
    * Contextual Filtering: Differentiate between "blood" used in biomarker sampling (e.g., "serum analysis") and circulatory assessments like "blood pressure."
    * Pattern Recognition: Classify studies containing terms such as "hypertension study" or "vascular health," with the negative class of "irrelevant paper."

Reasonably step through the mentioned criteria generating a response as to whether that criterion is satisfied as you move forward and return the final results in JSON format. 

Again, don't forget to STRICTLY RETURN A SINGLE JSON OBJECT EVERY SINGLE TIME WITH NO EXTRA TEXT. 
"""
def openai_req(user_prompt, sys_prompt=sys_prompt, cot_few_shot_examples=cot_few_shot_examples):
    from openai import OpenAI 
    import os

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            { "role": 'system', "content": f'{sys_prompt}'},
            { 
                "role": 'user',
                "content": f"{cot_few_shot_examples['user_one']}"
            },
            { 
                "role": 'assistant',
                "content": f"{cot_few_shot_examples['assistant_one']}"
            },
            { 
                "role": 'user',
                "content": f"{cot_few_shot_examples['user_two']}"
            },
            { 
                "role": 'assistant',
                "content": f"{cot_few_shot_examples['assistant_two']}"
            },
            { 
                "role": 'user',
                "content": user_prompt
            }
        ] 
    )

    return response.choices[0]