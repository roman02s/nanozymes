from typing import List, Dict
import logging

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn


from src.find_simulary import find_simulary
from src.chat import ChatGPT
from src.find_params import SubstanceSizeExtractor
from src.pdf2text import PDF2text
from src.get_parameters import get_parameters

# # logger = logging.getLogger('nanozymes_bot')
# # logger.setLevel(logging.INFO)

# # Создаем обработчик для записи логов в файл
# file_handler = logging.FileHandler('logs/nanozymes_bot.log')
# file_handler.setLevel(logging.INFO)

# # Создаем форматтер для записи логов в удобочитаемом формате
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # Добавляем обработчик к логгеру
# # logger.addHandler(file_handler)

# Опционально, можно добавить обработчик для вывода логов в консоль
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)
# # logger.addHandler(console_handler)

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
    # logger.info(f"/nanozymes_bot::request : {request}")
    # try:
    link = request.article.get("link", None)
    if link is None:
        return {"answer": "No link", "context": request.context}
    document = link.split("/")[-1]
    # document = "C4RA15675G.pdf"
    # logger.info(f"My document: {document}")
    # query_text = "query:  Fe3O4 NPs"
    get_context_for_query = find_simulary(document, request.query_text)
    llm = ChatGPT()
    llm_response = llm(
        query=request.query_text,
        previous_questions=request.context,
        context=get_context_for_query[0])
    new_context = request.context + "\n\n" + request.query_text + "\n\nresponse: " + llm_response + "\n\n"
    return {"answer": llm_response, "context": new_context}
    # except BaseException as e:
    #     error_message = f"Error: {document} - {e}"
    #     # logger.error(error_message)
    #     return {"answer": "Error, "+ str(document) + " " + str(e), "context": request.context}

@app.post("/find_parameters", response_model=FindParametersResponse)
async def handler_find_simulary(request: FindParametersRequest):
    # logger.info(f"/find_parameters::request : {request}")
    k_m = request.k_m
    v_max = request.v_max
    if k_m is None and v_max is None:
        return {"articles": {}}
    

    articles: List[Dict[str, str]] = get_parameters(k_m, v_max)

    result = {}
    for id, article in enumerate(articles, start=1):
        result[f"article_{id}"] = {
            "distance": article["distance"],
            "text_with_parameters": str(article.items()),
        }
    
    return {"articles": result}


if __name__ == "__main__":
    # logger.info("RUN SERVER")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
