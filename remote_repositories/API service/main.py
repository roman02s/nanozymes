from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn

from src.find_simulary import find_simulary
from src.chat import ChatGPT
from src.find_params import SubstanceSizeExtractor
from src.pdf2text import PDF2text


app = FastAPI()

class NanozymesBotRequest(BaseModel):
    article: dict
    query_text: str
    context: str

class NanozymesBotResponse(BaseModel):
    answer: str
    context: str

class FindParametersRequest(BaseModel):
    article: dict

class FindParametersResponse(BaseModel):
    params: dict


@app.post("/nanozymes_bot", response_model=NanozymesBotResponse)
async def handler_nanozymes_bot(request: NanozymesBotRequest):
    try:
        link = request.article.get("link", None)
        if link is None:
            return {"answer": "No link", "context": request.context}
        document = link.split("/")[-1]
        # document = "C4RA15675G.pdf"
        print("My document: ", document)
        # query_text = "query:  Fe3O4 NPs"
        get_context_for_query = find_simulary(document, request.query_text)
        llm = ChatGPT()
        llm_response = llm(
            query=request.query_text,
            previous_questions=request.context,
            context=get_context_for_query[0])
        new_context = request.context + "\n\n" + request.query_text + llm_response + "\n\n"
        return {"answer": llm_response, "context": new_context}
    except ValueError as e:
        return {"answer": "Error,"+ str(document) + str(e), "context": request.context}

@app.post("/find_parameters", response_model=FindParametersResponse)
async def handler_find_simulary(request: FindParametersRequest):
    link = request.article.get("v_max")
    if link is None:
        return {"params": {}}
    document = link.split("/")[-1]
    
    extractor = SubstanceSizeExtractor()
    text_document = PDF2text(["data" + "/" + document]).build_index()
    print(text_document)
    for example in text_document:
        sizes = extractor.extract_sizes(example)
        if sizes:
            print(f"Substance sizes in '{example}': {', '.join(sizes)}")

    # articles = {
    #     # Пример заполнения
    #     "article_1": {
    #         "doi_url": "...",
    #         "text_with_parameters": {"K_m": 123,}
    #     }
    # }
    
    return {"params": {}}


if __name__ == "__main__":
    print("RUN SERVER")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
