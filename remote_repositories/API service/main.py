from typing import List, Dict
from src.logger import Logger

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import json

from src.find_similary import find_similary
from src.chat import ChatGPT
from src.find_params import SubstanceSizeExtractor
from src.pdf2text import PDF2text
from src.get_parameters import get_parameters


app = FastAPI()


class NanozymesBotRequest(BaseModel):
    article: dict
    query_text: str
    context: str

class NanozymesBotResponse(BaseModel):
    answer: str
    context: str

class FindParametersRequest(BaseModel):
    k_m: str
    v_max: str

class FindParametersResponse(BaseModel):
    articles: dict


@app.post("/nanozymes_bot", response_model=NanozymesBotResponse)
async def handler_nanozymes_bot(request: NanozymesBotRequest):
    Logger.info(f"/nanozymes_bot::request : {request}")
    # try:
    link = request.article.get("link", None)
    if link is None:
        return {"answer": "No link", "context": request.context}
    document = link.split("/")[-1]
    # document = "C4RA15675G.pdf"
    # Logger.info(f"My document: {document}")
    # query_text = "query:  Fe3O4 NPs"
    get_context_for_query = find_similary(document, request.query_text)
    llm = ChatGPT()
    llm_response = llm(
        query=request.query_text,
        previous_questions=request.context,
        context=get_context_for_query[0])
    new_context = request.context + "\n\n" + request.query_text + "\n\nresponse: " + llm_response + "\n\n"
    return {"answer": llm_response, "context": new_context}
    # except BaseException as e:
    #     error_message = f"Error: {document} - {e}"
    #     # Logger.error(error_message)
    #     return {"answer": "Error, "+ str(document) + " " + str(e), "context": request.context}

@app.post("/find_parameters", response_model=FindParametersResponse)
async def handler_find_parameters(request: FindParametersRequest):
    Logger.info(f"/find_parameters::request : {request}")
    k_m = request.k_m
    v_max = request.v_max
    if k_m is None and v_max is None:
        return {"articles": {}}
    

    articles: List[Dict[str, str]] = get_parameters(k_m, v_max)

    result = {}
    for id, article in enumerate(articles, start=1):
        result[f"article_{id}"] = {}
        for key, value in article.items():
            result[f"article_{id}"][key] = str(value)

    Logger.info(f"/find_parameters::result : {result}")
    
    return {"articles": result}


if __name__ == "__main__":
    Logger.info("RUN SERVER")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
