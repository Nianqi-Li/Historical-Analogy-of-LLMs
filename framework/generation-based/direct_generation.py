import ast
import wikipedia
import json
from tqdm import tqdm
from llm_tools import *

def llm_predict(text, stop=[]):
    return chatgpt(text, stop)

def get_analogy(event:dict) -> str:
    template = '''You are a historical analogy bot. For input events, your goal is to find the event that best fits the analogy. Here is a case:
    
    ==== case 
    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    Historical Analogies Events:
    Spanish flu
    
    ==== Answer the following questions using the format given above
    Input Event:
    {event}
    Historical Analogies Events:
    '''
    analogy_event = llm_predict(template.format(event=f"{event['event_name']}\n{event['event_intro']}"),stop=['\n'])
    return analogy_event

def read_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    testset = read_jsonl('testset.jsonl')
    for event in tqdm(testset[:]): 
        with open('output.jsonl', 'a+', encoding='utf-8') as f:      
            ans = get_analogy(event)
            event['analogy_event'] = ans
            event['candidate'] = []
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
