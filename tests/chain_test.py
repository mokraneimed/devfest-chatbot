from pathlib import Path
import sys

current_directory = Path(__file__).resolve().parent
parent_directory = current_directory.parent
sys.path.append(str(parent_directory))

from creds import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
from config import embedding_model

import pinecone
from database import VectorDB
from langchain.embeddings.openai import OpenAIEmbeddings

from chain import create_conversational_chain

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embedding_model)
vectorstore = VectorDB.from_existing_index(PINECONE_INDEX_NAME, embeddings)

chat_chain = create_conversational_chain(vectorstore)

while True:
    q = input("Q: ")
    a = chat_chain(q)

    print("A: ", a['answer'])