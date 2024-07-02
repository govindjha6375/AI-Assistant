import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from PIL import Image
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    
def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

def get_response(input, image):
    model = genai.GenerativeModel("gemini-pro-vision")
    if input!="":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

def get_session_state():
    return st.session_state

def get_response_image(image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([image[0], prompt])
    return response.text
#Prep Image Data
def prep_image(uploaded_file):
    #Check if there is any data
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        #Get the image part information
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File is uploaded!")

def get_response_planner(prompt, input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, input])
    return response.text

def get_response_diet(prompt, input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, input])
    return response.text

def get_response_nutrition(image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([image[0], prompt])
    return response.text

def read_pdf(pdf):
    text = ""
    for file in pdf:
        pdf_read = PdfReader(file)
        for page in pdf_read.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_Store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_Store.save_local("faiss_index")

def get_conversation_chain_pdf():
    prompt_template = """
    Your role is to be meticulous researcher. Answer the question using only the information found within the context.
    Be detailed, but avoid unnecessary rambling.
    If you cannot find the answer, simply state 'answer is not available in the context'

    Context: \n{context}?\n
    Question: \n{question}?\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt=prompt)
    return chain

def user_input(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    load_vector_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = load_vector_db.similarity_search(user_query)
    chain = get_conversation_chain_pdf()

    response = chain(
        {"input_documents":docs, "question": user_query},
        return_only_outputs = True
    )
    print(response)
    st.write("AI Response", response["output_text"])
