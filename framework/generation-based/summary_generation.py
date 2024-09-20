import ast
import wikipedia
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=['chatgpt', 'gpt4', 'gemini'], type=str)
parser.add_argument("--testset", required=True, type=str)
args = parser.parse_args()

def llm_predict(text, stop=[]):
    if args.model == 'chatgpt':
        return chatgpt(text, stop)
    elif args.model == 'gpt4':
        return gpt4(text, stop)
    elif args.model == 'gemini':
        return gemini(text, stop)

def event_analysis(event:dict) -> dict:
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
    event_output = llm_predict(template.format(event=f"{event['event_name']}: {event['event_intro']}"))
    event["summary"] = event_output[event_output.find('1. Summary: ')+len('1. Summary: '):event_output.find('2. Background: ')]
    event["background"] = event_output[event_output.find('2. Background: ')+len('2. Background: '):event_output.find('3. Process: ')]
    event["process"] = event_output[event_output.find('3. Process: ')+len('3. Process: '):event_output.find('4. Result: ')]
    event["result"] = event_output[event_output.find('4. Result: ')+len('4. Result: '):]
    
    return event

def get_candidate(event:dict) -> list:
    template = '''
    You are a historical analogy robot. 
    For input events, please consider the summary, background, process and results, output 10 historical events that are similar in many aspects above, and return them in list format.
    The following is an example:
    
    Input Event: 
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
    Output: ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa","SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague pandemics"]
    
    Input Event: 
    {event}
    Output: '''
    
    candidate_text = llm_predict(template.format(event=f"{event['event_name']}: {event['summary']} {event['background']} {event['process']} {event['result']}"))
    candidate = ast.literal_eval(candidate_text)
    return candidate

def get_candidate_details(candidate:list) -> list:
    candidate_details = []
    for c in candidate:
        try:
            c_event = {'event_name':c,'event_intro':wikipedia.summary(c)}
            c_event = event_analysis(c_event)
            candidate_details.append(c_event)
        except:
            continue
    return candidate_details

def llm_choice(event:dict, candidate:list) -> str:
    template = '''You are an analogy robot. For the input event and the historical event used for selection, your goal is to find the best event that can be used for analogies. Here is a case:
    
    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
    Optional Historical Events:
    2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
    Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...
    Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
    Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945...
    Historical Analogies Events:
    Spanish flu
    
    Input Event:
    {input_event}
    Optional Historical Events:
    {candidate_events}
    Historical Analogies Events:
    '''
    ans = llm_predict(template.format(input_event=f"{event['event_name']}: {event['summary']} {event['background']} {event['process']} {event['result']}",
                                      candidate_events="\n".join([f"{e['event_name']}: {e['summary']} {e['background']} {e['process']} {e['result']}" for e in candidate])),['\n'])
    
    return ans

def read_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    testset = read_jsonl(args.testset)
    for event in tqdm(testset[:]): 
        with open('output.jsonl', 'a+', encoding='utf-8') as f:      
            event = event_analysis(event)
            candidate = get_candidate(event)
            candidate_withdetails = get_candidate_details(candidate)
            ans = llm_choice(event, candidate_withdetails)
            event['analogy_event'] = ans
            event['candidate'] = [candidate]
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
