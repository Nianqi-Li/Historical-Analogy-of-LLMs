import os
import wikipedia
from openai import OpenAI
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
import random
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--testset", required=True, type=str)
args = parser.parse_args()

os.environ.update({"OPENAI_API_KEY": ""})
gpt4 = ChatOpenAI(model_name="gpt-4",temperature=0.0000000001)

def patched_beautifulsoup(html):
    return BeautifulSoup(html, 'html.parser')

wikipedia.wikipedia.BeautifulSoup = patched_beautifulsoup

def wiki(entity):
    try:
        # intro = wikipedia.summary(entity,auto_suggest=False)
        intro = wikipedia.summary(entity)
    except wikipedia.exceptions.PageError as e:
        entity_candidates = wikipedia.search(entity)
        intro = wikipedia.summary(entity_candidates[0],auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        entity = random.choice(e.options)
        intro = wikipedia.summary(entity,auto_suggest=False)
    except:
        intro = wikipedia.summary(entity)
    if len(intro) >= 4096:
        intro = intro[:4096]
    return intro

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# pass_1
def pass_1(dataset):
    pass_1 = 0
    for data in tqdm(dataset):
        target_events = wikipedia.search(data['target_event'])
        analogy_events = wikipedia.search(data['analogy_event'])
        for e in target_events:
            if e in analogy_events:
                pass_1 += 1
                break
    return pass_1/len(dataset)
    
# abstract similarity
def extract_features(event:dict, input_example:dict=None) -> dict:
    if input_example is None:        
        template = '''
        You are an event summary robot. For the long event description input, please combine your knowledge and summarize it into four parts: summary, background, process and result. The summary should be concise, with each parts consisting of only one sentence and no more than 100 words.
        The following is an example:
        
        Input Event: 
        September 11 attacks: The September 11 attacks, commonly known as 9/11,[f] were four coordinated Islamist suicide terrorist attacks carried out by al-Qaeda against the United States in 2001...
        Output:
        1. Summary: The September 11 attacks, orchestrated by al-Qaeda, involved four coordinated terrorist hijackings, resulting in the deadliest terrorist attack in history with 2,977 fatalities.
        2. Background: Al-Qaeda, led by Osama bin Laden, targeted the U.S. due to its support of Israel, military presence in Saudi Arabia, and sanctions against Iraq.
        3. Process: On September 11, 2001, 19 terrorists hijacked four planes, crashing two into the World Trade Center in New York, one into the Pentagon, and the fourth in Pennsylvania after passengers revolted.
        4. Result: The attacks led to the U.S. launching the War on Terror, including invasions of Afghanistan and Iraq, substantial global anti-terrorism legislation, and long-term impacts on global security and economy.
        
        Input Event: {event}
        Output:
        '''
        event_output = gpt4.invoke(input=template.format(event=f"{event['event_name']}: {event['event_intro']}"))
    else:
        template = '''
        You are an event summary robot. For the long event description input, please combine your knowledge and summarize it into four parts: summary, background, process and result. The summary should be concise, with each parts consisting of only one sentence and no more than 100 words.
        The following is an example:
        
        Input Event: 
        {event_name}: {event_intro}
        Output:
        1. Summary: {event_summary}
        2. Background: {event_background}
        3. Process: {event_process}
        4. Result: {event_result}
        
        Input Event: {event}
        Output:
        '''
        event_output = gpt4.invoke(input=template.format(event_name=input_example['event_name'],
                                                         event_intro=input_example['event_intro'],
                                                         event_summary=input_example['topic'],
                                                         event_background=input_example['background'],
                                                         event_process=input_example['process'],
                                                         event_result=input_example['result'],
                                                         event=f"{event['event_name']}: {event['event_intro']}"))
    event["topic"] = event_output.content[event_output.content.find('1. Summary: ')+len('1. Summary: '):event_output.content.find('2. Background: ')]
    event["background"] = event_output.content[event_output.content.find('2. Background: ')+len('2. Background: '):event_output.content.find('3. Process: ')]
    event["process"] = event_output.content[event_output.content.find('3. Process: ')+len('3. Process: '):event_output.content.find('4. Result: ')]
    event["result"] = event_output.content[event_output.content.find('4. Result: ')+len('4. Result: '):]
    return event

def abstract_similarity(text1, text2):
    template = ''' You are a sentence-level analogy scoring robot. For the two input texts, please judge the quality of the analogy and give it a score (1-4). It should be noted that the quality of an analogy only focuses on the abstract-level similarity of descriptions, not the surface similarity of descriptions. For example, in a good analogy, two descriptions may belong to the same topic and express similar general situations, but they may not necessarily be the same specific process or description.
    
    ## Grading
    1 point: The description belongs to a completely different topic or field, has no connection, and cannot be compared.
    2 points: The descriptions belong to the same general theme, but the general situation or aspect expressed is significantly different, and the quality of the analogy is low.
    3 points: The descriptions belong to the same topic and express similar general situations, but are somewhat different in details or focus. This is an acceptable analogy.
    4 points: The descriptions belong to exactly the same topic, the general situation expressed is highly similar, the concepts and key points are highly overlapping, and it is a good analogy.
    In addition, there are several points to note:
    1. [Self-analogy is bad!!!]. Similarly, if one description overwrites another description, it is also a bad analogy.
    2. The quality of an analogy is only affected by abstract-level similarity and the similarity or identity of entities does not affect the quality of the analogy. For example, "The United States attacked Japan" and "The United States helped Japan" are completely incomparable; while "The United States attacked Japan" and "Germany invaded France" are good analogies.
    
    ## The following is two case:
    Case Description 1: On September 11, 2001, 19 terrorists hijacked four planes, crashing them into the World Trade Center, the Pentagon, and a field in Pennsylvania After a passenger revolt.
    Case Description 2: On December 7, 1941, 353 Japanese aircraft attacked Pearl Harbor, damaging or sinking eight battleships and destroying over 180 U.S. aircraft.
    Score: 3
    
    Case Description 1: The spillover of the Syrian Civil War had significant impacts in the Arab world and beyond, leading to a wider regional conflict and the rise of the Islamic State of Iraq and the Levant.\n\n
    Case Description 2: The Revolutions of 1989 were a series of political changes that led to the end of communist rule in Central and Eastern Europe, marking the end of the Cold War.\n\n
    Score: 2
    
    ## Question
    Description 1: {text1}
    Description 2: {text2}
    Score: 
    '''
    result = gpt4.invoke(input=template.format(text1=text1,text2=text2)).content
    try:
        score = int(result.strip())
    except:
        score = int(re.search(r'\d+', result).group())
        if score > 4: print('error score')
    return score

# literal similarity
def jacc(text1, text2):
    stop_words = set(stopwords.words('english'))
    text1 = word_tokenize(text1)
    set1 = set([word.lower() for word in text1 if word.lower() not in stop_words])
    text2 = word_tokenize(text2)
    set2 = set([word.lower() for word in text2 if word.lower() not in stop_words])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# multi-dimensional similarity score
def multi_dimensional_similarity(testset):
    score = []
    dimensions = ["topic","background","process","result"]      
    for data in tqdm(testset):
        input_event = extract_features(data)  
        analog_event = {"event_name": data['analogy_event'],
                        "event_intro": wiki(data['analogy_event'])} 
        analog_event = extract_features(analog_event, input_example=input_event)   
        data['score'] = {"topic":{"high_level":{},"low_level":{}},
                         "background":{"high_level":{},"low_level":{}},
                         "process":{"high_level":{},"low_level":{}},
                         "result":{"high_level":{},"low_level":{}}}
        for d in dimensions:
            data['score'][d]['abstract_level'] = abstract_similarity(input_event[d],analog_event[d])
            data['score'][d]['literal_level'] = jacc(input_event[d],analog_event[d])
        score.append(data)

    abstract_score = {"topic":0,"background":0,"process":0,"result":0} 
    literal_score = {"topic":0,"background":0,"process":0,"result":0} 
    overall_score = {"topic":0,"background":0,"process":0,"result":0,"all":0} 
    for data in score:
        overall_temp = {"topic":0,"background":0,"process":0,"result":0}
        for d in dimensions:
            abstract_level = data['score'][d]['abstract_level']
            literal_level = data['score'][d]['literal_level']
            abstract_score[d] += abstract_level/len(score)
            literal_score[d] += literal_level/len(score)
            if literal_level >= 0.35:
                literal_level = 0
            else:
                literal_level = 0.35-literal_level
            overall = abstract_level * literal_level
            overall_score[d] += overall/len(score)
            overall_temp[d] = overall
        overall_score['all'] += (overall_temp['topic']*0.5+overall_temp['background']*1+overall_temp['process']*2+overall_temp['result']*2)/len(score)
    return abstract_score, literal_score, overall_score

if __name__ == "__main__":
    testset = read_jsonl(args.testset)
    abstract_score, literal_score, overall_score = multi_dimensional_similarity(testset)
    print(f"abstract score: {abstract_score}")
    print(f"literal score: {literal_score}")
    print(f"overall multi-dimensional similarity: {overall_score}")
        
