from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn

from src.find_simulary import find_simulary


app = FastAPI()

class NanozymesBotRequest(BaseModel):
    article: dict
    query_text: str
    context: str

class NanozymesBotResponse(BaseModel):
    answer: str
    context: str

class FindSimilaryRequest(BaseModel):
    params: dict

class FindSimilaryResponse(BaseModel):
    articles: dict


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
        result = find_simulary(document, request.query_text)
        print("type(result): ", type(result[0][0]))
        print(str(result[0]))
        data_to_send = {
            "query_text": request.query_text,
            "context": request.context
        }
        new_context = request.context + "\n\n" + request.query_text
        return {"answer": str(result[0]), "context": new_context}
    except BaseException as e:
        return {"answer": "Error,"+ str(document) + str(e), "context": request.context}

@app.post("/find_simulary", response_model=FindSimilaryResponse)
async def handler_find_simulary(request: FindSimilaryRequest):
    v_max = request.params.get("v_max")
    K_m = request.params.get("K_m")
    
    # Здесь код для поиска статей с заданными параметрами
    # ...
    
    articles = {
        # Пример заполнения
        "article_1": {
            "doi_url": "...",
            "text_with_parameters": {"K_m": 123,}
        }
    }
    
    return {"articles": articles}


if __name__ == "__main__":
    print("RUN SERVER")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
