import json
import openai
import numpy as np
from llm_tools import *
from tqdm import tqdm

def llm_predict(text, stop=[]):
    return chatgpt(text, stop)

def read_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_history_datasets() -> list:
    text = read_jsonl('event_pool.jsonl.jsonl')
    embeddings = read_jsonl('similarity_embeddings.jsonl')
    history_event = [i | j for i in text for j in embeddings if i['url'] == j['url']]
    return history_event

history_event = get_history_datasets()

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def get_similar_events(event: dict) -> list:
    embeddings = get_embedding(event['event_intro'])
    similar_events = sorted([
        (e, vector_similarity(e['embeddings'], embeddings)) for e in history_event if e['history_event_text'] != event['event_name']
    ], reverse=True, key=lambda x: x[1])
    return [e[0] for e in similar_events]

if __name__ == "__main__":
    testset = read_jsonl('testset.jsonl')
    for event in tqdm(testset[:]): 
        with open('output_file.jsonl', 'a+', encoding='utf-8') as f:      
            candidate = get_similar_events(event)
            result = candidate[0]['history_event_text']
            event['analogy_event'] = result
            event['candidate'] = []
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
