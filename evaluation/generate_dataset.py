import json
import time
import pandas as pd

from keys import OPENAI_API_KEY

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY)
template = """You are a helpful assistant that reformulates an answer from a question and useful information
You make your output strictly as a valid JSON as follows:
{{
    "answer": "..."
}}
Given the following question and the information to answer from, reformulate an answer:
Question
{question}
Information
{information}
Answer:
"""
prompt = PromptTemplate.from_template(template=template)
chain = LLMChain(prompt=prompt, llm=llm)

with open('evaluation/devfest_qa.json') as f:
    qa = json.load(f)
    
questions = []
answers = []

for i in range(len(qa)):
    for j in range(len(qa[i]['questions'])):
        questions.append(qa[i]['questions'][j])
        answers.append(eval(chain.run(
        question=qa[i]['questions'][j],
        information=qa[i]['chunk'],
            ))['answer'].replace("\"", ''))
        print(f"[INFO] Answer generated for {i+1}th chunk and {j+1}th question! Sleeping...")
        time.sleep(5)

df = pd.DataFrame({
    'Prompt': questions,
    'Response': answers
})

df.to_csv('data/devfest_qa_dataset.csv', index=False)