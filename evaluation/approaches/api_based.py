import json
import time
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.document import Document

from keys import OPENAI_API_KEY

#load chunks
with open('data/devfest_chunks.json', 'r') as json_file:
    data = json.load(json_file)

#create documents for langchain model
documents = []

for _data in data:
  document = Document(page_content=_data,metadata ={'source': 'devfest_chunks.json'})
  documents.append(document)

#create vectorestore
embeddings = OpenAIEmbeddings()  
vectorstore = Chroma.from_documents(documents, embeddings)

#qa model
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
At the beginning of standalone question add this 'You're a chatbot designed to answer questions about Devfest hackathon.'.
At the end of standalone question add this 'Answer the question with short, energetic, and happy-sounding responses and add emojis.
If the question is unrelated to the context reply with 'I am sorry, I am not CHAT GPT !',
and if you can't find the information, reply with 'I'm sorry, there is no information about it right now'.'
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

#function to answer questions 
def answer(query):
  qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), vectorstore.as_retriever(), memory=memory, condense_question_prompt=CONDENSE_QUESTION_PROMPT)
  result = qa({"question": query})
  print(f"[A] {result['answer']}")
  time.sleep(10)
  return result["answer"]

#load questions
with open('data/devfest_qa.json', 'r') as json_file:
    data = json.load(json_file)  
questions = []
for item in data:
    questions.extend(item["questions"])
df = pd.DataFrame({"question": questions})

#answers in json file
df["answer"] = df["question"].apply(answer)
df.to_json('api_based_qa.json', orient='records')