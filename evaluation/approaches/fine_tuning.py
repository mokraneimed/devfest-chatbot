import json
import time
import openai

def generate(prompt):
    try:
        systemRole = "You are a kind helpful assitant in GDG Algiers event: DevFest"
        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::85iF2qDK",  # to be changed when needed
            temperature=0,
            messages=[
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
            ]
        )
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        return str(e)
    
with open("data/devfest_qa.json") as f:
    qa = json.load(f)
    
finetuning_qa = []
for q in qa:
    for question in q["questions"]:
        answer = generate(question)
        finetuning_qa.append({"question": question, "answer": answer})
        print(f"[Q] {question}")
        print(f"[A] {answer}")
        print("\n")
        time.sleep(5)
        

with open("data/finetuning_qa.json", "w") as f:
    json.dump(finetuning_qa, f, indent=4)