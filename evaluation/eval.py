import json
import pandas as pd

questions = []
q_generated = False
approaches = {
    "finetuning": [],
    "api_based": [],
    "similarity_search": []
}

for approach in approaches:
    with open(f"data/approaches/{approach}_qa.json") as f:
        qas = json.load(f)
        for qa in qas:
            approaches[approach].append(qa["answer"])
            if not q_generated:
                questions.append(qa["question"])
    q_generated = True

df = pd.DataFrame({
    "Question": questions,
    "Fine-tuning": approaches["finetuning"],
    "API-based": approaches["api_based"],
    "Similarity Search": approaches["similarity_search"]
})

df.to_csv("data/analysis/qa_analysis.csv", index=False)