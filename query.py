import os
import joblib
from dotenv import load_dotenv
import openai

import index_embedding
from index_embedding import *
import pandas as pd

# Ensure embed_text_list is imported or defined
from index_embedding import embed_text_list

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
data = pd.read_csv("data/rawdata.csv")
index_path = os.getenv("DOCTOR_INDEX_PATH")

def get_answer(data,query,index_path = None):

    query_embedding = embed_text_list([query])
    if index_path is None:
        raise ValueError("index_path must be provided and cannot be None")
    index = faiss.read_index(index_path)
    D, I = index.search(query_embedding, k=1)
    answer = data.iloc[I[0][0]]['Doctor']

    prompt = f"""You are a helpful medical assistant. The patient asked: "{query}". 
    Based on the following doctor's response, search the internet, 
    use chain of thoughts & reply professionally:\n\n{answer}"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


get_answer(data,
"I have a headache and fever, what should I do?",
index_path = index_path)