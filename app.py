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
        Create a story using the following scenario; the story should have be maximum 30 words long.
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
    with open('audio-img/story_speech.flac', 'wb') as file:
        file.write(response.content)


# user interface

def main():

    st.set_page_config(page_title= "IMAGE TO STORY CONVERTER", page_icon= "🖼️") #title
    st.header("Image-to-Story converter")
    #file uploader
    file_upload = st.file_uploader("please upload a jpg image here", type="jpg")
    #save file
    if file_upload is not None:
        try:
            image_bytes = file_upload.getvalue()
            with open(f'audio-img/{file_upload.name}', "wb") as file:
                file.write(image_bytes)
            #display image
            st.image(file_upload, caption = "Uploaded image")
            #run functions
            file_name = file_upload.name
            text = image_to_text(f'audio-img/{file_name}')
            if text:
                story = text_to_story(text)
                with st.expander('Generated image scenario'):
                        st.write(text)
                if story:
                    story_to_speech(story)
                    with st.expander('Generated short story'):
                        st.write(story)
                    st.audio('audio-img/story_speech.flac')
                else:
                    st.error("Failed to generate a story from the text.")
            else:
                st.error("Failed to generate a text from the image.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image to generate a story.")


if __name__ == '__main__':
    main()
