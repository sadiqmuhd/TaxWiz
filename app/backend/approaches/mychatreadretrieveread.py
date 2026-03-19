import os
import openai
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
import pinecone

# Replace these with your own values, either in environment variables or directly here
load_dotenv()
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_TENANT_ID = os.getenv('TENANTID')
# AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
embed_model = os.getenv('EMBEDDING_MODEL')


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv('PINECONE_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

KB_FIELDS_CONTENT = os.getenv("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.getenv("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.getenv("KB_FIELDS_SOURCEPAGE") or "sourcepage"

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service

# azure_credential = DefaultAzureCredential()
azure_credential = AzureDeveloperCliCredential() if AZURE_TENANT_ID == None else AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
search_creds = AzureKeyCredential(AZURE_SEARCH_KEY)

# Used by the OpenAI SDK
# openai.api_type = "azure"
# openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
# openai.api_version = "2022-12-01"

# Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable instead
# openai.api_type = "azure_ad"
# openai.api_key = azure_credential.get_token("https://cognitiveservices.azure.com/.default").token

# Set up clients for Cognitive Search and Storage
search_client = SearchClient(
    endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    index_name=AZURE_SEARCH_INDEX,
    credential=search_creds)
pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)

index = pinecone.Index(AZURE_SEARCH_INDEX)

def retrieve(query, namespace):
    limit = 3750
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=2, include_metadata=True, namespace=namespace)
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

# ChatGPT uses a particular set of tokens to indicate turns in conversations
# prompt_prefix = """<|im_start|>system
# Bankbot. Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
# Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

# Sources:
# {sources}

# <|im_end|>"""

# turn_prefix = """
# <|im_start|>user
# """

# turn_suffix = """
# <|im_end|>
# <|im_start|>assistant
# """

# history = []
# summary_prompt_type = """Let's think step by step. Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Based on the conversation so far answer True if the new question is related to the conversation so far in the summary or answer False if the new question is not related to the conversion in the summary so far.
# Summary:
# {summary}

# Question:
# {question}

# Answer:
# """

# summary_prompt_template = """Let's think step by step. Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a semantic search query for Azure Search Service based on the conversation and the new question.

# Summary:
# {summary}

# Question:
# {question}

# Search query:
# """

# Critically examine Source1 and Source2 and determine which of these sources can accurately and efficiently answer user question, then respond to user question using the best source starting with a brief definition of user question, if fact is available in the source then proceed to full response.

# If the source1 and source2 can be used to accurately answer user question then give a brief summary of source1 and source2 in not more than 5 lines, then seek for further clarification from user.

prompt_prefix = """
Assistant helps the company employees to answer questions about the organizaion's process, policy questions and employee handbook questions.####

#### Follow this steps to answer question.

Step 1: #### Examine Source1 and Source2 and determine which of these sources can sufficiently answer user enquiry, then give a full response to user enquiry using the best source, do not combine sources in your response treat each source separately. 

Step 2: #### If Source1 and Source2 can sufficietly answer the question give a brief summary of source1 and source2 then ask for further action from the staff.

Step 3: #### If source contains HTML table tags then response is most likely to be in the HTML table properly formatted for web browser.

Step 4: #### Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, try to attempt the question do not generate fictional answers. If asking a clarifying question to the user would help, ask the question.

Step 5: #### For tabular information return it as an html table. Do not return markdown format.

Source1:
{source1}

Source2:
{source2}

user question:
{user_question}

"""

turn_prefix = """
<|im_start|>user
"""

turn_suffix = """
<|im_end|>
<|im_start|>assistant
"""

history = []

summary_prompt_template =  """Below is the conversation history so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation history and the new question. Source names are not good search terms to include in the search query.


history:
{summary}

Question:
{question}

Search query:
"""
history_searching = []
history_main = []

# ADD DUMMY DATA
history_main.append({'here'})
history_searching.append({'here'})
# history_main.append({'role': 'system', 'content': prompt_prefix})

def answer_query_with_context(user_input, namespace, meta):
    # Execute this cell multiple times updating user_input to accumulate chat history

    # Exclude category, to simulate scenarios where there's a set of docs you can't see
    prompt_history = turn_prefix

    # if len(history) > 0:
    #     completion = openai.Completion.create(
    #         engine=AZURE_OPENAI_GPT_DEPLOYMENT,
    #         prompt=summary_prompt_type.format(summary="\n".join(history), question=user_input),
    #         temperature=0.3,
    #         max_tokens=500,
    #         stop=["\n"])
    #     search_type = completion.choices[0].text.title()
    #     print('........................................')
    #     print('........................................')
    #     print(f'Is this a follow-up question? {search_type}.........')
    #     if search_type == 'True':
    #         completion = openai.Completion.create(
    #             engine=AZURE_OPENAI_GPT_DEPLOYMENT,
    #             prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
    #             temperature=0.3,
    #             max_tokens=500,
    #             stop=["\n"])
    #         search = completion.choices[0].text
    #         print('........................................')
    #         print('........................................')
    #         print(f'Old Search Query: {search}.........')
    #         print(summary_prompt_template.format(summary="\n".join(history)))
    #     else:
    #         search = user_input
    #         print('........................................')
    #         print('........................................')
    #         print(f'New Search Query: {search}.........')
    # else:
    #     search = user_input

    if len(history) > 0:
        history_searching[0] = {'role': 'system', 'content': summary_prompt_template.format(summary="\n".join(history_searching), question=user_input)}
        history_searching.append({'role': 'user', 'content': user_input})
        completion = openai.ChatCompletion.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=history_searching,
            temperature=0.2,
            max_tokens=500,
)
        # search = completion.choices[0].text
        search = completion.choices[0]['message']['content']
        # history_searching.append({'role': 'user', 'content':user_input})
        # history_searching.append({'role': 'assistant', 'content':search})

    else:
        history_searching.append({'role': 'system', 'content': summary_prompt_template.format(summary="\n".join(''), question=user_input)})
        search = user_input
    
    # update history
    history_searching.append({'role': 'assistant', 'content':search})

    # Search on both the text and search query.
    print("Searching:", search) 
    print("-------------------")

    results = retrieval(search, namespace, meta)
    # content = "\n".join(results)
    # print(content)
    # print("-------------------")

    prompt = prompt_prefix.format(source1=results[0], source2=results[1], user_question=search) + prompt_history + user_input + turn_suffix
    print('Prompting Printing.......................')
    print(prompt)
    print('Prompt Printing Ended....................')
    history_main[0] = {'role': 'system', 'content':prompt_prefix.format(source1=results[0], source2=results[1], user_question=search)}
    history_main.append({'role': 'user', 'content': user_input})
    completion = openai.ChatCompletion.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=history_main,
        temperature=0.2, 
        max_tokens=1500,
        stop=["<|im_end|>", "<|im_start|>"])

    # reply = completion.choices[0].text
    reply = completion.choices[0]['message']['content']
    history_main.append({'role': 'assistant', 'content':reply})
    # prompt_history += user_input + turn_suffix + reply + "\n<|im_end|>" + turn_prefix
    # history.append("user: " + user_input)
    # history.append("assistant: " + reply)

    completion = openai.ChatCompletion.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT, 
        messages=[{'role': 'system', 'content': f'Using the text provided in the triple backticks, pretty format the text using HTML tags as it will be rendered using web browser. Text: {reply}'}], 
        temperature=0, 
        max_tokens=2024
        )
    
    reply1 = completion.choices[0]['message']['content']
    
    return reply1

'''You are an expert in answering questions. Please answer<question> Using source documents <doc>'''
def retrieval(query:str, namespace:str, meta:str):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    vec = res['data'][0]['embedding']
    
    res = index.query(vec, top_k=2, include_metadata=True, namespace=namespace)
    
    contexts = [
        # (x['metadata'][meta], x['metadata']['title']) for x in res['matches']
        # Try to retrieve the page title
        x['metadata'][meta]  for x in res['matches']
    ]
    
    print('Context 1.......................................')
    print(contexts[0])

    print('Context 2.......................................')
    print(contexts[1])
    print('THE END OF CONTEXT..............................')
    return contexts

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