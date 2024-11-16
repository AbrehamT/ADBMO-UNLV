import requests


with open('alzheimersdiseasebmoprojectdatacollectionmaterials915/alzheimersdiseasebmoprojectdatacollectionmaterials/qlist.txt', 'r') as file:
    lines = file.readlines() # Read each line into a list
# Strip any leading/trailing whitespace (including newlines) from each line
qlist = [line.strip() for line in lines]

#Append to query date:
#extratoken = ''
#extratoken = ' AND 2023[pdat]'
#extratoken = ' AND 2024[pdat]'
extratoken = ' AND (2023[pdat] OR 2024[pdat])'
for i in range(len(qlist)):
    if qlist[i] == '': continue  #skip blank lines
    qlist[i] = qlist[i] + extratoken
#print(qlist)

results = []
for query in qlist:
    if query == '': continue  #skip blank lines
    # Query parameters
    #query = "Alzheimer + AND + Cerebrospinal fluid"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}einfo.fcgi"
    api_key = "14360a5489f9c6515569fae147b5295a3608"  # Replace with your actual API key
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "api_key": api_key,
    }

    # Make the request to PubMed API
    response = requests.get(search_url, params=params)

    if response.status_code == 200:
        data = response.json()
        total_count = data["esearchresult"]["count"]
        #print(f"Total articles found for query '{query}': {total_count}")
        results.append(f"Total articles found for query '{query}': {total_count}")
        print(".", sep='', end='')
    else:
        print("Failed to fetch article count:", response.status_code)

print('Done.\n',sep='', end='')

f = open("results.txt", "w")
for i in results:
    f.write(i+'\n')
f.close()