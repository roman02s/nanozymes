from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn

from src.find_simulary import find_simulary

document = "C4RA15675G.pdf"
query_text = "query:  Fe3O4 NPs"
result = find_simulary(document, query_text)
print("type(result): ", type(result))


# app = FastAPI()

# class NanozymesBotRequest(BaseModel):
#     article: dict
#     query_text: str
#     context: str

# class NanozymesBotResponse(BaseModel):
#     answer: str
#     context: str

# class FindSimilaryRequest(BaseModel):
#     params: dict

# class FindSimilaryResponse(BaseModel):
#     articles: dict

# @app.post("/nanozymes_bot", response_model=NanozymesBotResponse)
# async def nanozymes_bot(request: NanozymesBotRequest):
#     # chatgpt_url = "https://api.openai.com/v1/chatgpt/..."
#     # headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN_HERE"}
    
#     # data_to_send = {
#     #     "query_text": request.query_text,
#     #     "context": request.context
#     # }
    
#     # async with httpx.AsyncClient() as client:
#     #     resp = await client.post(chatgpt_url, headers=headers, json=data_to_send)
    
#     # chatgpt_response = resp.json()
    
#     # answer = chatgpt_response.get("answer", "")
#     # new_context = request.context + "\n" + request.query_text
#     answer = ""
#     new_context = ""
#     return {"answer": answer, "context": new_context}

# @app.post("/find_similary", response_model=FindSimilaryResponse)
# async def find_similary(request: FindSimilaryRequest):
#     v_max = request.params.get("v_max")
#     K_m = request.params.get("K_m")
    
#     # Здесь код для поиска статей с заданными параметрами
#     # ...
    
#     articles = {
#         # Пример заполнения
#         "article_1": {
#             "doi_url": "...",
#             "text_with_parameters": {"K_m": 123,}
#         }
#     }
    
#     return {"articles": articles}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="localhost", port=8000, reload=True)
