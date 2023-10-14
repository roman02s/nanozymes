import faiss
import torch
import torch.nn.functional as F
import numpy as np

from .pdf2text import PDF2text
from src.embedder_e5 import (
    e5_get_embeddings,
    average_pool,
    model,
    tokenizer,
    device
)
print("SUCCESS IN GET CONTEXT")
# Константы
path_data = "data"
embeddings_data = path_data + "/" + "embeddings"


def get_context(document: str, query_text: str):
    """
    # Входной параметр
    пример:
    document = "C4RA15675G.pdf"
    query_text = "query:  Fe3O4 NPs"
    # Выходной параметр
    контекст
    пример: 

    [['than HRP (0.062 mM) and Fe3O4 NPs (0.010 mM). After'
  '320 (cid:1)C to 330 (cid:1)C. These CoFe2O4 NPs exhibit size and shape-dependent peroxidase-like activity'
  'polyhedron; this order was closely related to their particle size and crystal morphology. CoFe2O4NPs\nexhibited high stability in HAc–NaAc buﬀer (pH ¼ 4.0) and high activity over a broad pH (2.5–6.0).'
  'A simple route to CoFe2O4 nanoparticles with\nshape and size control and their tunable\nperoxidase-like activity†\nKe Zhang, Wei Zuo, Zhiyi Wang, Jian Liu, Tianrong Li, Baodui Wang*\nand Zhengyin Yang*'
  'Furthermore, the Michaelis constants Km value for the CoFe2O4 NPs (0.006 mM) with TMB as the\nfurther surface\nsubstrate was lower'
  'functionalization with folic acid (FA), the folate-conjugated CoFe2O4 nanoparticles allow discrimination'
  'tronics, and many other areas of nanotechnology.1 CoFe2O4\nferrite (CF), which is a well-known inverse spinel with Co2+ ions\non B sites and Fe3+ ions distributed equally among A and B'
  'depend on local chemical composition, size, and shape. Here, we report a new precursor-mediated\ngrowth of monodisperse magnetic cobalt ferrite (CoFe2O4) NPs with controlled size and shape. CoFe2O4'
  'of HeLa cells (folate receptor overexpression) from NIH-3T3 cells (without folate receptor expression).'
  'NPs with near corner-grown cubic, near cubic and polyhedron shape can be successfully prepared by']]

    """
    pdf2text = PDF2text([path_data + "/" + document])
    fixed_documents = pdf2text.build_index()
    all_texts = [item.page_content for item in fixed_documents]
    texts = all_texts[:20].copy()
    print(len(texts), len(all_texts), "all texts and 20 first")

    from tqdm import tqdm
    import gc

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
            torch.save(embds, embeddings_data + "/" + f"embds_last_part_{document}.pt")


    all_embds = torch.load(embeddings_data + "/" + f"embds_last_part_{document}.pt")
    combined_tensor = torch.cat(all_embds, dim=0)

    dim = combined_tensor.shape[1] #передаем размерность пр-ва
    size = combined_tensor.shape[0] #размер индекса

    index = faiss.IndexFlatL2(dim)
    # print(index.ntotal)  # пока индекс пустой
    index.add(combined_tensor.cpu().detach().numpy())
    # print(index.ntotal)  # теперь в нем sentence_embeddings.shape[0] векторов


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
    # тут используется query_text
    result_sem = sem_search_faiss(
        query_text=query_text,
        index=index,
        #  values = np.array(parags),
        top_k=10
    )
    return np.array(texts)[result_sem[1]]


print("~SUCCESS IN GET CONTEXT")
