import ast
import wikipedia
import json
from tqdm import tqdm
from llm_tools import *

def llm_predict(text, stop=[]):
    return chatgpt(text, stop)

# get candidate from llm
def get_candidate(event:dict) -> list:
    template = '''
    You are a historical analogy robot. 
    For input events, please consider the summary, background, process and results, output 10 historical events that are similar in many aspects above, and return them in list format.
    
    ===== The following is an example:
    Input Event: 
    2019–20 coronavirus pandemic
    The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history.\nCOVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles. 
    The 10 historical events that are similar with input:  
    ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa","SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague pandemics"]
    
    ===== question
    Input Event: 
    {event}
    The 10 historical events that are similar with input: 
    '''
    
    candidate_text = llm_predict(template.format(event=f"{event['event_name']}\n{event['event_intro']}"))
    print(candidate_text)
    try:
        candidate = ast.literal_eval(candidate_text)
    except:
        meta_prompt = '''Output the given analog historical events in [Python list format], which means enclosed by "[]", each item is framed by double quotes (") and items are connected by commas(,). 
        The following is an output example:
        ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa",
            "SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague's pandemics"]
        Only analog historical events need to be output, and all other information is ignored.
        Use single quotes(') instead of double quotes(") in event names. And events need to be enclosed in double quotes: like "A"B" and "A"B is not allowed item

        Output the historical analogy events in the text according to the above format：
        {text}

        Python list analog historical events:
        '''
        candidate_text = chatgpt(meta_prompt.format(text=candidate_text))
        print(candidate_text)
        candidate = ast.literal_eval(candidate_text)
    return candidate

# get detail of candidate from wikipedia
def get_candidate_details(candidate:list) -> list:
    candidate_details = []
    for c in candidate:
        try:
            c_event = {'event_name':c,'event_intro':wikipedia.summary(c)}
            candidate_details.append(c_event)
        except:
            continue
    return candidate_details

# llm choice analogy event
def llm_choice(event:dict, candidate:list) -> str:
    template = '''You are an analogy robot. For the input event and the historical event used for selection, your goal is to find the best event that can be used for analogies. Here is a case:
    
    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    Optional Historical Events:
    2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
    Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry.The source of the illness remained a mystery until the cases were mapped out using epidemiological method. The origin was found to be the polluted chalk water well at Addington, London, which supplied water to up to one-fifth of the area that is now the London Borough of Croydon. Coupled with issues around the co-operation between the medical officers and the administrators of the Borough, three coincidental events were blamed; changes to the well structure by repair work, the employment of a new workman who was an unwitting carrier of typhoid, and failure to chlorinate the water.
    Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
    Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. Historians do not fully agree on its starting and ending points, but the period is generally considered to span the 1947 Truman Doctrine to the 1991 Dissolution of the Soviet Union. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945. Aside from the nuclear arsenal development and conventional military deployment, the struggle for dominance was expressed via indirect means such as psychological warfare, propaganda campaigns, espionage, far-reaching embargoes, rivalry at sports events and technological competitions such as the Space Race.
    
    Among the options, the most appropriate one to use as an analogy for 2019–20 coronavirus pandemic is Spanish flu
    
    Input Event:
    {input_event}
    Optional Historical Events:
    {candidate_events}
    
    Among the options, the most appropriate one to use as an analogy for {input_name} is '''
    ans = llm_predict(template.format(input_event=f"{event['event_name']}: {event['event_intro']}",
                                      candidate_events="\n".join([f"{e['event_name']}: {e['event_intro']}" for e in candidate]),
                                      input_name=event['event_name']),['\n'])
    
    print(ans)
    return ans

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
            candidate = get_candidate(event)
            candidate_withdetails = get_candidate_details(candidate)
            ans = llm_choice(event, candidate_withdetails)
            event['analogy_event'] = ans
            event['candidate'] = [candidate]
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
