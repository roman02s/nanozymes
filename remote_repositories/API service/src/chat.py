import os
from time import sleep

import openai

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader

import pandas as pd
from IPython.display import display, Markdown
print("SUCCESS IN CHAT")

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = 'gpt-3.5-turbo'
chat = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=1024)
embeddings = OpenAIEmbeddings()



