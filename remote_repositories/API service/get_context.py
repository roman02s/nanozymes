import gc

import numpy as np
import pandas as pd

import faiss

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]

def build_index(file_paths, chunk_size, chunk_overlap):
    documents = [load_single_document(path) for path in file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    fixed_documents = []
    for doc in documents:
        doc.page_content = process_text(doc.page_content)
        if not doc.page_content:
            continue
        fixed_documents.append(doc)

    print(f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы.")
    return fixed_documents

fixed_documents = build_index(file_paths=["data/C4RA15675G.pdf"], chunk_size=200, chunk_overlap=10)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

print("device: ", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("device: ", device)
def e5_get_embeddings(knowledge_texts): # каждый элемент обязательно начинаетмся с "passage:"
    passage_batch_dict = tokenizer(knowledge_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    passage_outputs = model(**passage_batch_dict.to(device))
    passage_embeddings = average_pool(passage_outputs.last_hidden_state, passage_batch_dict['attention_mask'])
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
    return passage_embeddings
all_texts = [item.page_content for item in fixed_documents]
texts = all_texts[:20].copy()

path_data = "data"
embeddings_data = path_data + "/" + "embeddings"

embds = []
for i, parag in tqdm(enumerate(texts), total=len(texts)):

    torch.cuda.empty_cache()
    gc.collect()

    embd = e5_get_embeddings(f"passage: {parag}")
    embds.append(embd.to('cpu'))
    del embd

    if len(embds) == 100:

        torch.save(embds, f"embds_{i+1 - 100}_{i+1}.pt")
        embds = []

    if i + 1 == len(texts):
        torch.save(embds, embeddings_data + "/" + f"embds_last_part.pt")

all_embds = torch.load(embeddings_data + "/" + "embds_last_part.pt")
combined_tensor = torch.cat(all_embds, dim=0)

dim = combined_tensor.shape[1] #передаем размерность пр-ва
size = combined_tensor.shape[0] #размер индекса

index = faiss.IndexFlatL2(dim)
print(index.ntotal)  # пока индекс пустой
index.add(combined_tensor.cpu().detach().numpy())
print(index.ntotal)  # теперь в нем sentence_embeddings.shape[0] векторов


def get_query_embedding(query_text): # query_text = "query: Какие основания для получения 33 услуги?"
    query_batch_dict = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
    query_outputs = model(**query_batch_dict.to(device))
    query_embedding = average_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().detach().numpy()

# Создадим поиск с помощью FAISS
def sem_search_faiss(query_text, index, top_k=5):
    query = get_query_embedding(query_text)
    D, I = index.search(query, top_k)
    # resp = np.array(values)[I]
    return D, I
query_text = "query:  Fe3O4 NPs"

result_sem = sem_search_faiss(
           query_text=query_text,
           index=index,
          #  values = np.array(parags),
           top_k=10
           )


print(np.array(texts)[result_sem[1]])
print("END")
