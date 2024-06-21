import os
import json

import google.generativeai as genai

# Getting the current working directory(to avoid hard coding the locations)
working_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_dir}/config.json"
config_data = json.load(open(config_file_path))

# Fetching the API key from the config file
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configuring the model(gemini) with our API key
genai.configure(api_key=GOOGLE_API_KEY)



# Loading the gemini_model

#Function to load Gemini-model for chatbot
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model


#Function to load Gemini-pro-vision-model(for image Captioning)
def load_gemini_pro_vision_model(prompt,image):
    gemini_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_vision_model.generate_content([prompt,image])
    result = response.text
    return result



# Function to get embedding for text
def embedding_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model = embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")

    embedding_list = embedding["embedding"]

    return embedding_list


# Function to get a response from gemini-pro LLM
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result  = response.text
    return result

