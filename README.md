# Image-to-Speech-GenAI-tool

An AI tool that receives an image and uses generative AI to create a narrated short story in an audio file about the context of the image.

The tool utilizes Hugging Face Transformers open-source framework for deep learning in combination with OpenAI prompting via Langchain framework.

![](audio-img/app-snapshot.jpg)

# Approach

Execution is devided into 3 parts:
- Image to text: a transformers model is used to generate a scenario based on the on the AI understanding of the image context
- Text to story: OpenAI LLM model is prompted to create a short story based on the generated scenario
- Story to speech: a transformers model is used to convert the generated short story into an narrated audio file

# requirements

- os
- dotenv
- transformers
- langchain
- requests
- streamlit

