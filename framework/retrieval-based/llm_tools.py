import os
from langchain_openai import ChatOpenAI
import google.generativeai as genai

# GPT4
os.environ.update({"OPENAI_API_KEY": ""})
llm1 = ChatOpenAI(model_name="gpt-4")

def gpt4(text, stop=[]):
    return llm1.invoke(input=text,stop=stop).content

# ChatGPT
llm2 = ChatOpenAI(model_name="gpt-3.5-turbo")

def chatgpt(text, stop=[]):
    return llm2.invoke(input=text,stop=stop).content

# Gemini
genai.configure(api_key='', transport='rest')
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def gemini(text, stop=[]):
    model = genai.GenerativeModel('gemini-pro')
    generation_config=genai.GenerationConfig(stop_sequences = stop,
                                             max_output_tokens = 256,
                                             temperature=0,
                                             top_p=1)
    response = model.generate_content(contents=text,
                                      safety_settings=safety_settings,
                                      generation_config=generation_config)
    ans = response.text
    return ans
