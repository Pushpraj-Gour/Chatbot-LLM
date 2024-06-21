import os
from PIL import Image


import streamlit as st
from streamlit_option_menu import option_menu

from gemini_utility import load_gemini_pro_model,load_gemini_pro_vision_model,embedding_model_response,gemini_pro_response


# Getting the current working directory(to avoid hard coding the locations)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Setting up the page configuration

st.set_page_config(
    page_title="Catlin AI",
    page_icon="🤖",
    layout = "centered"
)

with st.sidebar:
    selected = option_menu("Cat",
                            ["ChatBot",
                             "Image Captioning",
                             "Embed text",
                             "Ask me anything"],
                            menu_icon ="robot",
                           icons =["chat-dots","image-alt","cursor-text","person-raised-hand"],
                            default_index =0)


# function to translate role between gemini-pro and streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role=="model":
        return "assistant"
    else:
        return user_role



# Setting ChatBot navigation page setting
if selected=="ChatBot":
    # ChatBot page title
    st.title("🤖 ChatBot")
    
    model = load_gemini_pro_model()

    # Initialize chat session in streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])



    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for users message
    user_prompt = st.chat_input("Ask Cat")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        #displaying Gemini-response
        with st.chat_message("assitant"):
            st.markdown(gemini_response.text)


# Setting Image Captioning Page
if selected=="Image Captioning":
    st.title("📷 Image Caption Generator")
    uploaded_image = st.file_uploader("Upload Image whose caption you want to know.....",type=["jpg","jpeg","png"])


    if uploaded_image:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_image)
            resized_img = img.resize((800,500))
            st.image(resized_img)

        with col2:
            if st.button("Generate Caption"):
                prompt = "Generate caption for the image"
                response = load_gemini_pro_vision_model(prompt,img)
                st.info(response)


# Setting the Embed Text page
if selected=="Embed text":

    st.title("🖹 Embedding for text")

    # Creating the text box
    input_text = st.text_area(label="",placeholder="Input the text to get the embedding for it.")

    if st.button("Get Embeddings"):
        response = embedding_model_response(input_text)
        st.markdown(response)


# Setting the Ask me any question page
if selected == "Ask me anything":
    st.title("🙋🏻 Feel free to ask.... ")

    # text box to enter prompt
    user_prompt = st.text_area(label="",placeholder="Ask Cat...")
    if st.button("Get an answer"):
        response = gemini_pro_response((user_prompt))
        st.markdown(response)