from pathlib import Path
import sys

current_directory = Path(__file__).resolve().parent
parent_directory = current_directory.parent
sys.path.append(str(parent_directory))

from creds import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME

import pinecone
import index

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment=PINECONE_ENV      
)      

Index = pinecone.Index(PINECONE_INDEX_NAME)

prompt = input("Prompt: ")
results = index.docsearch.similarity_search(prompt, 5)
for i, result in enumerate(results):
    print(f"CHUNK: {i}")
    print(result, '\n')