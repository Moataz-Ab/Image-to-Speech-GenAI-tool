# Image-to-Speech-GenAI-tool

An AI tool that receives an image and uses generative AI to create a narrated short story in an audio file about the context of the image.

The tool utilizes Hugging Face Transformers open-source framework for deep learning in combination with OpenAI prompting via Langchain framework.

![](audio-img/app-snapshot.jpg)

# Approach

Execution is devided into 3 parts:
- **Image to text:**
  a transformers ([Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base))model is used to generate a scenario based on the on the AI understanding of the image context
- **Text to story:**
  OpenAI LLM model is prompted to create a short story based on the generated scenario
- **Story to speech:**
  a transformers ([espnet/kan-bayashi_ljspeech_vits](https://huggingface.co/espnet/kan-bayashi_ljspeech_vits))model is used to convert the generated short story into an narrated audio file

# requirements

- os
- dotenv
- transformers
- langchain
- requests
- streamlit

# Using the app

- Before using the app the user shoudl have personal toeks for Hugging Face and Open AI
- The user should save the personal tokens in an ".env" file within the package under object names HUGGINGFACE_TOKEN and OPENAI_TOKEN
- The user can then run the app using the command: streamlin run app.py
- Once the app is running on streamlit, the user can upload the target image
- The processing will start automatically and it may take a few minutes to complete
- Once execution is complete, the app will display:
  - The scenario text generated by the image-to-text transformer model
  - The short story generated by prompting the OpenAI LLM
  - The autio file narrating the short story generated by the text-to-speech transformer model
- The audio file along with the image will be saved in "audio-img" folder inside the package

