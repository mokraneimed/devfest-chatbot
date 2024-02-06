from creds import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, OPENAI_API_KEY
from config import embedding_model

import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

with open("data/devfest_chunks.json", "r") as f:
	texts = json.load(f)

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment=PINECONE_ENV     
)
print("[VDB] Vector DB initilized!")

index = pinecone.Index(PINECONE_INDEX_NAME)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embedding_model)

print("[VDB] Indexing...")
docsearch = Pinecone.from_texts(texts, embedding, index_name=PINECONE_INDEX_NAME)
print("[VDB] Indexing complete!")