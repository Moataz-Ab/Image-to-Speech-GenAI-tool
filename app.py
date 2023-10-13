import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain import OpenAI, PromptTemplate, LLMChain
import requests
import streamlit as st

load_dotenv()
HUGGINGFACE_KEY = os.getenv('HUGGINGFACE_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_TOKEN')

#img to text

def image_to_text(image_url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = pipe(image_url)[0]['generated_text']
    print(text)
    return text


#llm
def text_to_story(text, model='gpt-3.5-turbo-instruct-0914', temperature=0.9):
    llm = OpenAI(openai_api_key=OPENAI_KEY, model=model, temperature=temperature)
    prompt = PromptTemplate(
        input_variables = ['text'],
        template = '''
        You are a talented story teller who can create a story from a simple narrative./
        Create a story using the following scenario; the story should have be maximum 20 words long.
        context = {text}
        '''
        )
    chain = LLMChain(llm=llm, prompt=prompt)
    story = chain.predict(text=text)

    print(story)
    return story



# text to speech
def story_to_speech(story):
    API_URL = 'https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits'
    headers = {"Authorization": f'Bearer {HUGGINGFACE_KEY}'}
    payload = {"inputs": story}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('story_speech.flac', 'wb') as file:
        file.write(response.content)


text = image_to_text("img/test_image1.jpg")
story = text_to_story(text)
speech = story_to_speech(story)
