# IMPORTING LIBRARIES
import os
import openai
import pinecone
import datetime
import glob
import html

from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from time import sleep
from tqdm.auto import tqdm
from dotenv import load_dotenv

# INITIALIZING KEYS AND ENVIRONMENT
print('Initializing keys and Environment')
load_dotenv()

storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
storage_key = os.getenv('AZURE_STORAGE_KEY')
azure_container = os.getenv('AZURE_STORAGE_CONTAINER')
search_key = os.getenv('AZURE_SEARCH_KEY')
search_service = os.getenv('AZURE_SEARCH_SERVICE')
my_index_name = os.getenv('AZURE_SEARCH_INDEX')
form_service = os.getenv('AZURE_FORM_RECOGNIZER')
form_service_key = os.getenv('AZURE_FORM_RECOGNIZER_KEY')
tenant_id = os.getenv('TENANTID')
AZURE_OPENAI_GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv('PINECONE_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
embed_model = os.getenv('EMBEDDING_MODEL')

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100
# filenames = glob.glob('C:/Users/Urukpauu/Desktop/UzFolder/Projects/2023/inscale/new_data/*')
filenames = glob.glob('data/*')

# initialize connection to openai
openai.api_key = OPENAI_API_KEY

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(api_key=pinecone_key, environment=pinecone_env)

        
azd_credential = AzureDeveloperCliCredential() \
                if tenant_id == None else AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
default_creds = azd_credential if search_key == None or storage_key == None else None
search_creds = AzureKeyCredential(search_key)
storage_creds = default_creds if storage_key == None else storage_key
formrecognizer_creds = default_creds \
                        if form_service_key == None else AzureKeyCredential(form_service_key)

def get_embedding(text, model=embed_model):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

# GET DATA
def extract_pdf(filenames):
    offset = 0
    page_map = []
    form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{form_service}.cognitiveservices.azure.com/", credential=formrecognizer_creds, headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
    for filename in filenames:
        with open(filename, "rb") as f:
            poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
            form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text, filename))
            offset += len(page_text)
    return page_map

def get_search_query(text):
    query_template = """ Using the text enclosed within the triple backticks generate a semantic search query. Text: ```{}```""".format(text)

    completion = openai.Completion.create(
        engine=AZURE_OPENAI_GPT_DEPLOYMENT,
        prompt=query_template,
        temperature=0,
        max_tokens=100,
        )
    try:
        search_query = completion.choices[0].text.split(':')[1]
    except:
        search_query = completion.choices[0].text
    return search_query

def data_prep_pinecone(page_map):
    new_page_map = []
    for i, pm in tqdm(enumerate(page_map)):
        new_page_map.append(
            {'id':i,
            'text':list(pm[2:])[0],
            'title':list(pm[3:])[0].split('\\')[1],
            'document':'{}_{}'.format(list(pm[3:])[0].split('\\')[1], pm[0]),
            'search query': get_search_query(list(pm[2:])[0])
            }
            )
    return new_page_map

def two_page_maker(pages):
    new_page = []
    for num, page in enumerate(pages):
        if num < len(pages) - 1:
            two_page = page[2] + '\n'+ pages[num+1][2]
            two_page = two_page[:3701]
            new_page.append((page[0], page[1], two_page, page[3]))
        else:
            new_page.append(page)
    return new_page


print('Extracting Data......')
page_map = extract_pdf(filenames)
# print('Making it two page')
# page_map1 = two_page_maker(page_map)
print('Data Preparation for Pinecone.....')
new_data = data_prep_pinecone(page_map)
print('Data Sample')
print(new_data[0])      
# check if index already exists (it shouldn't if this is first time)
if my_index_name in pinecone.list_indexes():
    # print('{my_index_name} Already Exists.' )
    pinecone.delete_index(my_index_name)
    # if does not exist, create index
print('Creating Pinecone Index...')
pinecone.create_index(
    my_index_name,
    dimension= 1536,
    metric='cosine',
    metadata_config={'indexed': ['title', 'document']}
    )
# connect to index
index = pinecone.Index(my_index_name)
# view index stats
# index.describe_index_stats()

def upsert_pinecone(namespace, col):
    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [str(x['id']) for x in meta_batch]
        # get texts to encode
        texts = [x[col] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'title': x['title'],
            'text': x['text'],
            'document': x['document'],
            'search_query': x['search query']
            } for x in meta_batch]
        print('Mini Batch Sample')
        print(meta_batch[0])
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        print('to_upsert sample')
        print(to_upsert[0])
        index.upsert(vectors=to_upsert, namespace=namespace)

upsert_pinecone('Text', 'text')
upsert_pinecone('SearchQuery', 'search query')

def retrieve(query):
    limit = 3750
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt
