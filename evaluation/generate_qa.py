import json
import time
from keys import OPENAI_API_KEY

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

n_questions = 3
llm = OpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY)
template = """You are a helpful assistant that generates questions about given information
You make your output strictly as a valid JSON as follows, Respect the exact following format which will be evaluated by eval() in python:
{{
    "questions": [q1, q2, q3]
}}
Given the following information, generate {n_questions} questions that can be asked about it
Information:
{information}
Questions:
"""
prompt = PromptTemplate.from_template(template=template)
chain = LLMChain(prompt=prompt, llm=llm)

qa = []
with open('data/devfest_chunks.json') as f:
    chunks = json.load(f)
print("[INFO] Chunks loaded!")
    
for i, chunk in enumerate(chunks, 1):
    output = chain.run(
        information=chunk,
        n_questions=n_questions
    )
    try:
        eval(output)
    except:
        print("[ERROR] Invalid output!")
    qa.append({
        'chunk': chunk,
        'question': eval(output)['questions']
    })
    print(f"[INFO] Generated Questions for {i}th chunk! Sleeping...")
        
    time.sleep(5)
    
with open('data/devfest_qa.json', 'w') as f:
    json.dump(qa, f, indent=4)
print("[INFO] Saved QA!")