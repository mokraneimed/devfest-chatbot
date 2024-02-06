from pathlib import Path
import sys
import pandas as pd
import json
import time

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



with open('data/devfest_qa.json', 'r') as json_file:
    data = json.load(json_file)  
questions = []
for item in data:
    questions.extend(item["questions"])
df = pd.DataFrame({"question": questions})

def answer(query):
  chat_chain = create_conversational_chain(vectorstore)
  result = chat_chain({"question": query})
  print(f"[A] {result['answer']}")
  time.sleep(10)
  return result["answer"]


df["answer"] = df["question"].apply(answer)
df.to_json('tests/test_results.json', orient='records')



