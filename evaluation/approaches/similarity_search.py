import json
import ai21
import pinecone
import pandas as pd
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
<<<<<<< HEAD:evaluation/approaches/similarity_search_based_refrasing.py
import json
import requests
import ai21
from dotenv import load_dotenv
=======

import os
from dotenv import load_dotenv
load_dotenv()
ai21.api_key = os.getenv("AI21_API_KEY")
>>>>>>> 9e01866e1228f2e40a6f1608da51d7d4b23506ca:evaluation/approaches/similarity_search.py

pinecone.init(      
	api_key='0f974a83-d911-40bc-916d-8c3356131b3a',      
	environment='gcp-starter'      
)      
index = pinecone.Index('chatbot')
<<<<<<< HEAD:evaluation/approaches/similarity_search_based_refrasing.py
questions_df = pd.read_json("../../data/devfest_qa.json")
questions = questions_df["questions"].explode()
embeddings = HuggingFaceEmbeddings(model_name="bert-large-uncased")
docsearch = Pinecone.from_existing_index("chatbot", embeddings)
load_dotenv()
ai21.api_key = os.getenv("AI21_API_KEY")

qa_json = []
=======

questions_df = pd.read_json("../../data/devfest_qa.json")
questions = questions_df["questions"].explode()

embeddings = HuggingFaceEmbeddings(model_name="bert-large-uncased")

docsearch = Pinecone.from_existing_index("chatbot", embeddings)


qa_json = {}

>>>>>>> 9e01866e1228f2e40a6f1608da51d7d4b23506ca:evaluation/approaches/similarity_search.py
for question in questions:
        docs = docsearch.similarity_search(question,10)
        qa_obj = {}
        answers = []
        for doc in docs:
            answers.append(str(doc.page_content))
        context = " ".join(answers)
        answer = ai21.Answer.execute(context=context,question=question)['answer']
        qa_obj["question"] = question
        if(answer == None):
            qa_obj["answer"] = "Sorry, the chatbot cannot answer this question"
        else:
<<<<<<< HEAD:evaluation/approaches/similarity_search_based_refrasing.py
            qa_obj["answer"] = answer
        qa_json.append(qa_obj)
            

with open('similarity_search_based_refrasing.json', 'w', encoding='utf-8') as f:
    json.dump(qa_json, f, ensure_ascii=False, indent=4)




=======
            qa_json[question] = answer

with open('similarity_search_based_refrasing_1.json', 'w', encoding='utf-8') as f:
    json.dump(qa_json, f, ensure_ascii=False, indent=4)
>>>>>>> 9e01866e1228f2e40a6f1608da51d7d4b23506ca:evaluation/approaches/similarity_search.py
