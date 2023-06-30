import os
import pickle
import numpy as np
import pandas as pd
import urllib
from datetime import datetime, timedelta
import logging
import copy
import uuid
import json
import openpyxl
import time
import tiktoken

from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks.base import CallbackManager

from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.env_vars import *

## Use with Python 3.9+ ONLY
# """
# from utils import km_agents
# from utils import openai_helpers
# from utils import fr_helpers
# from utils import summarization
# folder = './docs_to_summarize'
# ref_summ_df = summarization.summarize_folder(folder, mode='refine', verbose=False)
# mp_summ_df  = summarization.summarize_folder(folder, mode='map_reduce', verbose=False)
# """



# mapreduce_prompt_template = """The maximum output is about 500 to 750 tokens, so make sure to take advantage of this to the maximum.\n
# Write an elaborate summary of 3 paragraphs of the following:
mapreduce_prompt_template = """The maximum output is about 800 to 1000 tokens, so make sure to take advantage of this to the maximum.\n
Do not include explanation about text, document or legal factuals.\n
Summary should include name of card and detailed infos like incurring annual fees, cancellation fees, and interest charges as name and value pairs.\n
Summarize the text below as a bullet point list of the most important points.\n
Summary should be translated into Korean.\n

----------
{text}
----------

Given the new context, refine the original summary.\b
If the context isn't useful, return the original summary.\b
Translate texts in English into Korean.

SUMMARY:"""


refine_prompt_template = """Write an elaborate summary of 3 paragraphs of the following:

{text}

"""

"""
#    "Your job is to produce a final summary of 3 to 6 paragraphs that is elaborate and rich in details.\n" 
Extract the important entities mentioned in the text below. First extract all company names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes

Desired format:
Company names: <comma_separated_list_of_company_names>
People names: -||-
Specific topics: -||-
General themes: -||-

Text: {text}
"""

refine_template_orig = (
    "Your job is to produce a final summary that is elaborate and rich in details.\n" 
    "The maximum output is about 800 to 1000 tokens, so make sure to take advantage of this to the maximum.\n"
    "Summary should include detailed numbers like incurring annual fees, cancellation fees, and interest charges as name and value pairs.\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary."
    "Summarize the text below as a bullet point list of the most important points."
    "Summary should be translated into Korean.If it's already in Korean, just let it there.\n"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary."
    "If the context isn't useful, return the original summary."
    "Translate texts in English into Korean.\n"
)

# "First extract name of card, then extract all information in numbers, then extract specific topics which fit the content and finally extract general overarching themes"
refine_template = (
    "Your job is to produce a final summary that is elaborate and rich in details.\n" 
    "The maximum output is about 800 to 1000 tokens, so make sure to take advantage of this to the maximum.\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "Extract the important entities mentioned in the text below. "
    "First extract name of card, then extract all information in numbers, then extract specific topics which fit the content.\n"
    "Desired format:\n\n"
    "\t상품명: \n"
    "\t국내전용 연회비: \n"
    "\t국내외 겸용 연회비: \n"
    "\t신용카드 이용한도: \n"
    "\t단기 카드대출(현금서비스) 이용 한도: \n"
    "\t주요 혜택 및 부가 서비스: \n"
    "\t기타 서비스: \n"
    "\t이용대금 상환 방법, 금리(수수료) 및 변동 여부: \n"
    "\t수수료율: \n"
    "\t리볼빙 예시: \n\n"
    "Summary should be translated into Korean.If it's already in Korean, just let it there.\n"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary."
    "If the context isn't useful, return the original summary."
    "Translate texts in English into Korean.\n"
)


def get_model_max_tokens(model):
    if model == "text-search-davinci-doc-001":
        return DAVINCI_003_EMB_MAX_TOKENS
    elif model == "text-search-davinci-query-001":
        return DAVINCI_003_EMB_MAX_TOKENS
    elif model == "text-davinci-003":
        return DAVINCI_003_MODEL_MAX_TOKENS
    elif model == "text-embedding-ada-002":
        return ADA_002_MODEL_MAX_TOKENS
    elif model == "chat":
        return GPT35_TURBO_COMPLETIONS_MAX_TOKENS
    elif model == "gpt-35-turbo":
        return GPT35_TURBO_COMPLETIONS_MAX_TOKENS
    elif model == "gpt-4-32k":
        return GPT4_32K_COMPLETIONS_MODEL_MAX_TOKENS
    elif model == "gpt-4":
        return GPT4_COMPLETIONS_MODEL_MAX_TOKENS
    else:
        return ADA_002_MODEL_MAX_TOKENS

def get_encoding_name(model):
    if model == "text-search-davinci-doc-001":
        return "p50k_base"
    elif model == "text-embedding-ada-002":
        return "cl100k_base"
    elif model == "chat":
        return "cl100k_base"
    elif model == "gpt-35-turbo":
        return "cl100k_base"
    elif model == "gpt-4-32k":
        return "cl100k_base"
    elif model == "gpt-4":
        return "cl100k_base"
    elif model == "text-davinci-003":
        return "p50k_base"
    else:
        return "gpt2"

def get_encoder(model):
    if model == "text-search-davinci-doc-001":
        return tiktoken.get_encoding("p50k_base")
    elif model == "text-embedding-ada-002":
        return tiktoken.get_encoding("cl100k_base")
    elif model == "chat":
        return tiktoken.get_encoding("cl100k_base")
    elif model == "gpt-35-turbo":
        return tiktoken.get_encoding("cl100k_base")
    elif model == "gpt-4-32k":
        return tiktoken.get_encoding("cl100k_base")
    elif model == "gpt-4":
        return tiktoken.get_encoding("cl100k_base")
    elif model == "text-davinci-003":
        return tiktoken.get_encoding("p50k_base")
    else:
        return tiktoken.get_encoding("gpt2")


def get_model_dims(embedding_model):
    if embedding_model == "text-search-davinci-doc-001":
        return DAVINCI_003_EMBED_NUM_DIMS
    elif embedding_model == "text-embedding-ada-002":
        return ADA_002_EMBED_NUM_DIMS
    else:
        return ADA_002_EMBED_NUM_DIMS

def get_generation(model):
    if model == "text-davinci-003":
        return 3
    elif model == "gpt-35-turbo":
        return 3.5
    elif model == "chat":
        return 3.5
    elif model == "gpt-4-32k":
        return 4
    elif model == "gpt-4":
        return 4
    else:
        assert False, f"Generation unknown for model {model}"

def get_llm(model = CHOSEN_COMP_MODEL, temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS, stream=False, callbacks=[]):
    gen = get_generation(model)

    if (gen == 3) :
        llm = AzureOpenAI(deployment_name=model, model_name=model, temperature=temperature,
                        openai_api_key=openai.api_key, max_retries=30,
                        request_timeout=120, streaming=stream,
                        callback_manager=CallbackManager(callbacks),
                        max_tokens=max_output_tokens, verbose = True)

    elif (gen == 4) or (gen == 3.5):
        llm = ChatOpenAI(model_name=model, model=model, engine=model,
                            temperature=0, openai_api_key=openai.api_key, max_retries=30, streaming=stream,
                            callback_manager=CallbackManager(callbacks),
                            request_timeout=120, max_tokens=max_output_tokens, verbose = True)
    else:
        assert False, f"Generation unknown for model {model}"

    return llm


def chunk_doc(all_text, mode='refine', model=CHOSEN_COMP_MODEL, max_output_tokens=MAX_OUTPUT_TOKENS, chunk_overlap=500):

    # enc_name = openai_helpers.get_encoding_name(model)
    # enc = openai_helpers.get_encoder(model)
    enc_name = get_encoding_name(model)
    enc = get_encoder(model)

    max_tokens = get_model_max_tokens(model)

    if mode == 'refine':
        max_tokens = max_tokens - len(enc.encode(refine_prompt_template)) - len(enc.encode(refine_template)) - 2*MAX_OUTPUT_TOKENS - chunk_overlap
    elif mode == 'map_reduce':
        max_tokens = max_tokens - len(enc.encode(mapreduce_prompt_template)) - MAX_OUTPUT_TOKENS - chunk_overlap
    else:
        raise Exception('Invalid mode')

    text_splitter = TokenTextSplitter(encoding_name=enc_name, chunk_size = max_tokens, chunk_overlap=chunk_overlap)
    
    texts = text_splitter.split_text(all_text)
    docs = [Document(page_content=t) for t in texts]

    enc = get_encoder(CHOSEN_COMP_MODEL)

    l_arr = []
    for d in texts:
        l_arr.append(str(len(enc.encode(d))))

    print("Chunks Generated", len(docs), ' | max_tokens', max_tokens, " | Chunk Lengths:", ', '.join(l_arr))

    return docs


def clean_up_text(text):
    text = text.replace('....', '')
    return text



def get_refined_summarization(docs, form_recognizer_client, model=CHOSEN_COMP_MODEL, max_output_tokens=MAX_OUTPUT_TOKENS, stream=False, callbacks=[]):

    PROMPT = PromptTemplate(template=refine_prompt_template, input_variables=["text"])
    refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"],template=refine_template)

    llm = get_llm(model, temperature=0, max_output_tokens=max_output_tokens, stream=stream, callbacks=callbacks)

    chain = load_summarize_chain(llm, chain_type="refine",  question_prompt=PROMPT, refine_prompt=refine_prompt, return_intermediate_steps=True)
    summ = chain({"input_documents": docs}, return_only_outputs=True)
    
    return summ


def get_mapreduced_summarization(docs, form_recognizer_client, model=CHOSEN_COMP_MODEL, max_output_tokens=MAX_OUTPUT_TOKENS, stream=False, callbacks=[]):

    PROMPT = PromptTemplate(template=mapreduce_prompt_template, input_variables=["text"])

    llm = get_llm(model, temperature=0, max_output_tokens=max_output_tokens, stream=stream, callbacks=callbacks)

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT, return_intermediate_steps=True)
    summ = chain({"input_documents": docs}, return_only_outputs=True)
    
    return summ



@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(10))
def fr_analyze_local_doc_with_dfs(path, form_recognizer_client, verbose = True):

    with open(path, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-document", document=f)

    result = poller.result()
    
    contents = ''
    kv_contents = ''
    t_contents = ''

    for kv_pair in result.key_value_pairs:
        key = kv_pair.key.content if kv_pair.key else ''
        value = kv_pair.value.content if kv_pair.value else ''
        kv_pairs_str = f"{key} : {value}"
        kv_contents += kv_pairs_str + '\n'

    for paragraph in result.paragraphs:
        contents += paragraph.content + '\n'


    for table_idx, table in enumerate(result.tables):
        row = 0
        row_str = ''
        row_str_arr = []

        for cell in table.cells:
            if cell.row_index == row:
                row_str += ' \t ' + str(cell.content)
            else:
                row_str_arr.append(row_str )
                row_str = ''
                row = cell.row_index
                row_str += ' \t ' + str(cell.content)

        row_str_arr.append(row_str )
        t_contents += '\n'.join(row_str_arr) +'\n\n'  
            
    dfs = []

    # for idx, table in enumerate(result.tables):
        
    #     field_list = [c['content'] for c in table.to_dict()['cells'] if c['kind'] == 'columnHeader'] 
    #     print('\n', field_list)
        
    #     table_dict = table.to_dict()
    #     row_count = table_dict['row_count']
    #     col_count = table_dict['column_count']

    #     cells = [c for c in table_dict['cells'] if c['kind'] == 'content']
    #     rows = []
    #     max_cols = 0

    #     for i in range(row_count - 1):
    #         row = [c['content'] for c in cells if c['row_index'] == i + 1]
    #         # print(row, i)
    #         if len(row) > 0: rows.append(row)
    #         if len(row) > max_cols: max_cols = len(row)

    #     if len(field_list) < max_cols: field_list += [''] * (max_cols - len(field_list))
    #     df = pd.DataFrame(rows, columns=field_list)
    #     if verbose: display(df)
    #     dfs.append(df)

    return contents, kv_contents, dfs, t_contents

def read_document(path, form_recognizer_client, verbose = False):
    if verbose: print(f"Reading {path}")
    
    all_text = ''
    ext = os.path.splitext(path)[1]

    if ext == '.xlsx':
        dataframe = openpyxl.load_workbook(path, data_only=True)
        sheets = [s for s in dataframe.sheetnames if 'HiddenCache' not in s]
        for sheet in sheets:
            print('sheet', sheet)
            all_text += pd.read_excel(path, sheet_name=sheets[0]).to_string(na_rep='') + '\n\n\n\n'
    elif ext == '.csv':
        return None
    elif ext == '.pdf':
        contents, kv_contents, dfs, t_contents = fr_analyze_local_doc_with_dfs(path, form_recognizer_client, verbose = verbose)
        all_text = ' '.join([kv_contents , contents ,  t_contents])
    else:
        return None
    
    all_text = clean_up_text(all_text)

    return all_text


def summarize_document(path, form_recognizer_client, mode='refine', verbose = False):

    print(f"##########################\nStarting Processing {path} ...")
    start = time.time()
    text = read_document(path, form_recognizer_client, verbose=verbose)
    if text is None: return None

    summ = summarize_text(text, form_recognizer_client, mode=mode, verbose=verbose)
    end = time.time()

    summary = {
        'file': os.path.basename(path),
        'intermediate_steps': summ['intermediate_steps'],
        'summary': summ['output_text'],
        'proc_time': end-start
    }

    print(f"Done Processing {path} in {end-start} seconds\n##########################\n")
    return summary 


def summarize_text(text, form_recognizer_client, mode='refine', verbose = False):    
    docs = chunk_doc(text, mode=mode)

    if mode == 'refine':
        summ = get_refined_summarization(docs, form_recognizer_client)
    elif mode == 'map_reduce':
        summ = get_mapreduced_summarization(docs, form_recognizer_client)
    else:
        raise Exception("Invalid mode")

    print(f"summarize_text: {summ['output_text']}")

    return summ



def summarize_folder(folder, form_recognizer_client, mode='refine', save_to_csv=True, save_to_pkl=True, verbose = False):
    files = os.listdir(folder)
    print(f"Files in folder {len(files)}")
    pkl_file = os.path.join(folder, f'summaries_{mode}.pkl')
    csv_file = os.path.join(folder, f'summaries_{mode}.csv')

    if os.path.exists(csv_file):
        summ_df = pd.read_csv(csv_file)
    else:
        summ_df = pd.DataFrame(columns=['file', 'intermediate_steps', 'summary', 'proc_time'])

    processed_files = list(summ_df['file'])
    print(f"List of already processed files {processed_files}")
     
    for f in files:        
        path = os.path.join(folder, f)
        if f in processed_files: continue
        
        summary = summarize_document(path, form_recognizer_client, mode=mode, verbose=verbose)
        if summary is None: continue
        summ_df = pd.concat([summ_df, pd.DataFrame([summary])], ignore_index=True)

        if save_to_csv: summ_df.to_csv(csv_file)
        if save_to_pkl: summ_df.to_pickle(pkl_file)

    return summ_df

