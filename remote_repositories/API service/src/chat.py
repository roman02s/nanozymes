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



print("SUCCESS IN CHAT")


class BaseLLM:
    def __call__(self, query, context, previous_questions):
        raise NotImplementedError
    

class ChatGPT(BaseLLM):
    os.environ['OPENAI_API_KEY'] = "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"
    openai.api_key = os.environ['OPENAI_API_KEY']
    llm_model = 'gpt-3.5-turbo'


    # chat = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=1024)
    embeddings = OpenAIEmbeddings()

    def __call__(self, query, context, previous_questions):
        # query =  "How is synthesis carried out in this article?"
        # \n\nSynthesis искать это
        # какие реагенты и оборудование. попробовать агентов. если ты не нашёл слово синтез то попробуй поискать словосочетания с ключевыми словами ...
        # query = f'What is needed for synthesis of {size} nm or other size {formula} NPs? NPs means nanoparticles. Please indicate in the response all the parameters of the experiment specified in the article, including equipment and reagents and mmols. If the article does not say anything about synthesis, then answer it. Answer as fully as possible, try to take the maximum of the original text from the article. Your answer should consist of several paragraphs and be quite voluminous, while preserving the source text as much as possible'
        TEMPLATE = f"""
        Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know".
        \n\nContext: {str(context)}
        \n\nPrevious questions: {str(previous_questions)}
        \n\nQuestion: {str(query)}
        """
        # Создаем экземпляр чата
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Замените на вашу модель
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": TEMPLATE}
            ],
            max_tokens=1024,
            temperature=0.0
        )

        # Получаем ответ
        response = chat.choices[0].message['content']
        print(response)

        # Создаем экземпляр векторных представлений (embeddings)
        # embeddings = openai.Embedding.create(
        #     inputs=[
        #         "This is a sample sentence.",
        #         "Another sample sentence."
        #     ]
        # )

        # Получаем векторные представления
        # vector_embeddings = embeddings['data']

        # print(vector_embeddings)
        return response

