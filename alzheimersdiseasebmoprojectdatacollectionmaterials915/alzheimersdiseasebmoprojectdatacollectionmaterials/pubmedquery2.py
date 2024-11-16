import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os

def search_article_ids(query, api_key, max_articles_per_query):
    """
    Search articles matching the query, return their IDs with pagination.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    all_article_ids = []
    for start in range(0, max_articles_per_query, 100):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "api_key": api_key,
            "retstart": start,
            "retmax": 100
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            article_ids = data["esearchresult"]["idlist"]
            if not article_ids:
                break  # No more articles found
            all_article_ids.extend(article_ids)
            time.sleep(0.33)  # To respect PubMed's rate limit
        else:
            print(f"Failed to search article IDs: {response.status_code}")
            break
    return all_article_ids

def fetch_batch_details(article_ids, api_key):
    """
    Fetch details for a batch of article IDs.
    """
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    articles_info = []
    # Process in smaller batches if needed
    batch_size = 100
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(batch_ids),
            "api_key": api_key
        }
        response = requests.get(fetch_url, params=params)
        if response.status_code == 200:
            articles_info.extend(parse_article_details(response.text))
            time.sleep(0.33)  # To respect PubMed's rate limit
        else:
            print(f"Failed to fetch article details: {response.status_code}")
    return articles_info

def parse_article_details(xml_data):
    """
    Parse and collect article details including title, abstract, publication year,
    keywords, MeSH terms, authors, and PMC ID for full-text access where available.
    """
    root = ET.fromstring(xml_data)
    articles_info = []

    for article in root.findall('.//PubmedArticle'):
        article_info = {
            'title': article.find('.//ArticleTitle').text,
            'abstract': article.find('.//Abstract/AbstractText').text if article.find('.//Abstract/AbstractText') is not None else "No abstract",
            # Initialize other fields here...
        }

        # Publication Year
        pub_date = article.find('.//PubDate')
        article_date = article.find('.//ArticleDate')
        year = pub_date.find('Year').text if pub_date is not None and pub_date.find('Year') is not None else (article_date.find('Year').text if article_date is not None else "Unknown")
        article_info['publication_year'] = year

        # Keywords
        keywords = [kw.text for kw in article.findall('.//KeywordList/Keyword')]
        article_info['keywords'] = "; ".join(keywords)

        # MeSH Terms
        mesh_terms = [mh.find('DescriptorName').text for mh in article.findall('.//MeshHeadingList/MeshHeading')]
        article_info['mesh_terms'] = "; ".join(mesh_terms)

        # Authors
        authors = article.findall('.//Author')
        author_list = []
        for author in authors:
            last_name = author.find('LastName')
            initials = author.find('Initials')
            if last_name is not None and initials is not None:
                author_list.append(f"{last_name.text}, {initials.text}")
        article_info['authors'] = "; ".join(author_list)

        # PMC ID
        pmc_id = article.find('.//ArticleIdList/ArticleId[@IdType="pmc"]')
        pmc_id_text = pmc_id.text if pmc_id is not None else "Not Available"
        article_info['pmc_id'] = pmc_id_text

        # PMC URL
        base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"
        #base_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/" #Is this better?
        if pmc_id and pmc_id_text and pmc_id_text != "Not Available":
            # Construct and save the full URL
            full_url = f"{base_url}{pmc_id}/"
            article_info['pmc_url'] = full_url
        else:
            article_info['pmc_url'] = 'N/A'

        articles_info.append(article_info)

    return articles_info

def main(api_key, query, max_articles=1000):
    """
    Main function to orchestrate the search and data extraction process.
    """
    article_ids = search_article_ids(query, api_key, max_articles)
    if article_ids:
        articles_info = fetch_batch_details(article_ids, api_key)
        df = pd.DataFrame(articles_info)
        # Save to CSV
        df.to_csv(query+".csv", index=False)
        print(f"Found and saved {len(df)} articles.")
    else:
        print("No articles found.")

# Example usage
api_key = "496547d98a26cfb847f258ef044604727e08"
#query =  "Alzheimer's disease AND (fluid biomarker[Title/Abstract] OR cerebrospinal fluid[Title/Abstract] OR blood biomarker[Title/Abstract])"
#main(api_key, query, max_articles=10000)

with open('qlist.txt', 'r') as file:
    lines = file.readlines() # Read each line into a list
# Strip any leading/trailing whitespace (including newlines) from each line
qlist = [line.strip() for line in lines]

#Append to query date:
extratoken = ' AND (2023[pdat] OR 2024[pdat])'
for i in range(len(qlist)):
    if qlist[i] == '': continue  #skip blank lines
    qlist[i] = qlist[i] + extratoken
#print(qlist)

for query in qlist:
    if query == '': continue  #skip blank lines
    main(api_key, query, max_articles=10000)

#Done
