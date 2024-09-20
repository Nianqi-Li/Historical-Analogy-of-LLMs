import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import ast
import json
from tqdm import tqdm
import wikipedia
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=['chatgpt', 'gpt4'], type=str)
parser.add_argument("--testset", required=True, type=str)
args = parser.parse_args()
        
os.environ.update({"OPENAI_API_KEY": ""})

# get candidate conversation-chain
if args.model == 'chatgpt':
    llm_getcandidate = ChatOpenAI(model_name="gpt-3.5-turbo")
elif args.model == 'gpt4':
    llm_getcandidate = ChatOpenAI(model_name="gpt-4")
memory_getcandidate = ConversationBufferMemory()
template_getcandidate = """ You're a robot for getting historical analogies events. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
For input events, please consider the summary, background, process and results, and output 5 historical events that are similar in many aspects above, and return them in list format.
If there is any reflection, please modify the recommended events based on the reflection.
The following is an example:

Input Event: 
2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
Output: ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa"]

{chat_history}

{input_type}:
{input}
Output: 
"""

class InputPromptTemplate(PromptTemplate):
    template: str
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)
    
prompt_getcandidate = InputPromptTemplate(
    template=template_getcandidate,
    input_variables=["chat_history","input_type","input"]
)

memory_getcandidate = ConversationBufferMemory(memory_key="chat_history",input_key="input",ai_prefix="Output",human_prefix="Input")
getcandidateChain = LLMChain(llm=llm_getcandidate, prompt=prompt_getcandidate, verbose=False, memory=memory_getcandidate)

# get analogy choose model
if args.model == 'chatgpt':
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
elif args.model == 'gpt4':
    llm = ChatOpenAI(model_name="gpt-4")

def llm_choice(event:dict, candidate:list, warm_up:bool=False ,thought:str=''):
    if warm_up:
        template = '''You are a historical analogy reflection robot. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
        For the input event and the candidate event set, please make a comparison, reflect on the shortcomings of the candidate set, and make suggestions for obtaining a better analogous candidate set. Suggestions should be succinct and concise, with a single sentence indicating the direction of change for the candidate set.
        Here is a example:
        
        == example
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
        
        Optional Historical Events:
        2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
        Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...
        
        Thought:
        The 2019–20 coronavirus pandemic is a global epidemic, so the themes of 2022 South Asian floods are completely different. The Croydon typhoid outbreak of 1937 was smaller in scope, while the 2019–20 coronavirus pandemic were global influenza pandemics, so there is no suitable analogy here and I need to reflect.
        
        Reflection:
        Candidate events need to focus on the epidemic and its impact on a global scale.
        
        ==== question
        Input Event:
        {input_event}
        
        Optional Historical Events:
        {candidate_events}
        
        Thought:
        {thought}'''
        
    else:
        template = '''You are a historical analogy robot. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
        For the input event and the candidate event set used for selection, your goal is to find a most suitable event that can be used for historical analogies, which means the two events are similar in causes, processes, results, etc. If the events in the candidate set are not appropriate or better analogies may exist, you should reflect on the shortcomings of these events in the analogies, pointing out the desired focus of the analogies to help find a new candidate set of events.
        Here are two case:
        
        ==== case 1
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
        
        Optional Historical Events:
        2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
        Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...
        
        Thought:
        The 2019–20 coronavirus pandemic is a global epidemic, so the themes of 2022 South Asian floods are completely different. The Croydon typhoid outbreak of 1937 was smaller in scope, while the 2019–20 coronavirus pandemic were global influenza pandemics, so there is no suitable analogy here and I need to reflect.
        
        Reflection:
        Candidate events need to focus on the epidemic and its impact on a global scale.

        ==== case 2
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
        
        Optional Historical Event:
        Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
        Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945...

        Thought:
        The Cold War has nothing to do with the epidemic. The Spanish flu is also an epidemic and has had a great impact in Europe, so it is a qualified analogy for the 2019–20 coronavirus pandemic.
        
        Final Answer:
        Spanish flu
        
        ==== question
        Input Event:
        {input_event}
        
        Optional Historical Events:
        {candidate_events}
        
        Thought:
        {thought}'''
    
    ans = llm.invoke(input=template.format(input_event=f"{event['event_name']}: {event['summary']} {event['background']} {event['process']} {event['result']}",
                                           candidate_events="\n".join([f"{e['event_name']}: {e['summary']} {e['background']} {e['process']} {e['result']}" for e in candidate]),
                                           thought=thought),
                     stop=['Input Event:'])
    return ans.content

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
    event_output = llm.invoke(input=template.format(event=f"{event['event_name']}: {event['event_intro']}"))
    event["summary"] = event_output.content[event_output.content.find('1. Summary: ')+len('1. Summary: '):event_output.content.find('2. Background: ')]
    event["background"] = event_output.content[event_output.content.find('2. Background: ')+len('2. Background: '):event_output.content.find('3. Process: ')]
    event["process"] = event_output.content[event_output.content.find('3. Process: ')+len('3. Process: '):event_output.content.find('4. Result: ')]
    event["result"] = event_output.content[event_output.content.find('4. Result: ')+len('4. Result: '):]
    
    return event

# get detail of candidate from wikipedia
def get_candidate_details(candidate:list) -> list:
    candidate_details = []
    for c in candidate:
        try:
            c_event = {'event_name':c,'event_intro':wikipedia.summary(c)}
            candidate_details.append(event_analysis(c_event))
        except:
            continue
    return candidate_details

def historical_analogy(event_dict:dict) -> str:
    warm_up_rounds = 0
        
    event = event_analysis(event_dict)
    candidate_set = getcandidateChain.predict(input_type="Input Event",input=f"{event['event_name']}: {event['summary']} {event['background']} {event['process']} {event['result']}")
    candidate_set = ast.literal_eval(candidate_set)
    event_dict['candidate'] = [candidate_set]
    # print(candidate_set)
    candidate = get_candidate_details(candidate_set)
    if warm_up_rounds > 0:
        choice = llm_choice(event, candidate, warm_up=True)
        warm_up_rounds -= 1
    else:
        choice = llm_choice(event, candidate)
    # print(choice)
    while 'Reflection' in choice:
        candidate_set = getcandidateChain.predict(input_type="Reflection",input=choice[choice.find('Reflection:') + len('Reflection:'):])
        candidate_set = ast.literal_eval(candidate_set)
        event_dict['candidate'].append(candidate_set)
        # print(candidate_set)
        candidate = get_candidate_details(candidate_set)
        if warm_up_rounds > 0:
            choice = llm_choice(event, candidate, warm_up=True)
            warm_up_rounds -= 1
        else:
            choice = llm_choice(event, candidate)
        # print(choice)
    if 'Final Answer:' not in choice:
        choice += '\n\nFinal Answer:'
        choice += llm_choice(event, candidate, thought=choice)
        # print(choice)
    choice = choice[choice.find('Final Answer:') + len('Final Answer:'):]
    choice = choice.strip()
    return choice

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
            ans = historical_analogy(event)
            event['analogy_event'] = ans
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
