import os

from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from utils import (load_gemini_pro_model,extract_transcript_details,generate_gemini_content,
                   gemini_pro_response,get_response,get_session_state,get_response_image,prep_image,get_response_planner,
                   get_response_nutrition,get_response_diet,get_response_travel)

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                           ['ChatBot',
                            'Personal Nutritionist',
                            'TravelBot',
                            'YouTube Summarizer',
                            'Ask me anything',
                            'Planner',
                            'Image ChatBot'],
                           menu_icon='robot', icons=['chat-dots-fill', 'clipboard-pulse','airplane', 'youtube', 'patch-question-fill','list-task','card-image'],
                           default_index=0
                           )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

#chatbot  
if selected == 'ChatBot':
    model = load_gemini_pro_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)  # Renamed for clarity

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)


if selected == "Image ChatBot":
    st.title("📷 Snap Narrate")
    st.image("qna.png", width=100)
    session_state = get_session_state()
    if "history" not in session_state:
        session_state.history = []
    input_query = st.text_input("Input: ", key="input")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    image_data = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
    submit = st.button("Ask question!")
    if submit:
        response = get_response(input_query, image)
    #Append the history
        session_state.history.append({
            "query":"Image",
            "image":image,
            "response": response
        })
        st.subheader("Vision AI: ")
        st.write(response)
    #Display history
    st.subheader("History")
    for entry in session_state.history:
        query = entry["query"]
        image = entry.get("image")
        response = entry["response"]
    #Use the st.expander
        with st.expander(f"Query: {query}"):
            if image is not None:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Response :", response)

if selected == "YouTube Summarizer":
    prompt="""You are Yotube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points
    within 250 words. Please provide the summary of the text given here:  """
    st.image("youtube.jpg", width=120)
    st.title("▶️ YouTube Summarizer")
    
    youtube_link = st.text_input("Enter YouTube Video Link:")
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    if st.button("Get Detailed Notes"):
        transcript_text=extract_transcript_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)

if selected == "Ask me anything":
    st.image("download.jpeg", width=120)
    st.title("❓ Ask me a question")

    # text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)

if selected == "Planner":
    st.image("planner.jpg", width=120)
    st.title("🎒🏕️📸 Planner: Discover and Plan your Culinary Adventures!")
    section_choice = st.radio("Choose Section:", ("Location Finder", "Trip Planner", "Weather Forecasting", "Restaurant & Hotel Planner"))
    if section_choice == "Location Finder":
        upload_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])
        image = ""
        if upload_file is not None:
            image = Image.open(upload_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        #Prompt Template
        input_prompt_loc = """
        You are an expert Tourist Guide. As a expert your job is to provide summary about the place and,
        - Location of the place,
        - State & Capital
        - Cordinates of the place
        - Some popular places nearby
        Retun the response using markdown.
        """
        submit = st.button("Get Location!")
        if submit:
            image_data = prep_image(upload_file)
            response = get_response_image(image_data, input_prompt_loc)
            st.subheader("Tour Bot: ")
            st.write(response)
    
    if section_choice == "Trip Planner":
        #Prompt Template
        input_prompt_planner = """
        You are an expert Tour Planner. Your job is to provide recommendations and plan for given location for giner number of days,
        even if number of days is not provided.
        Also, suggest hidden secrets, hotels, and beautiful places we shouldn't forget to visit
        Also tell best month to visit given place.
        Retun the response using markdown.
        """
        input_plan = st.text_area("Provide location and number of days to obtain itinerary plan!")
        #Button
        submit1 = st.button("Plan my Trip!")
        if submit1:
            response = get_response_planner(input_prompt_planner, input_plan)
            st.subheader("Planner Bot: ")
            st.write(response)

    if section_choice == "Weather Forecasting":
        #Prompt Template
        input_prompt_planner = """
        You are an expert weather forecaster. Your job is to provide forecast for given place and you have to provide for next 7 days,
        forecast also, from the current date.
        - Provide Precipitation
        - Provide Humidity
        - Provide Wind
        - Provide Air Quality
        - Provide Cloud Cover
        Retun the response using markdown.
        """
        input_plan = st.text_area("Provide location to forecast weather!")
        #Button
        submit1 = st.button("Forecast Weather!")
        if submit1:
            response = get_response_planner(input_prompt_planner, input_plan)
            st.subheader("Weather Bot: ")
            st.write(response)

    if section_choice == "Restaurant & Hotel Planner":
        #Prompt Template
        input_prompt_planner = """
        You are an expert Restaurant & Hotel Planner. 
        Your job is to provide Restaurant & Hotel for given place and you have to provide not very expensive and not very cheap,
        - Provide rating of the restaurant/hotel
        - Top 5 restaurants with address and average cost per cuisine
        - Top 5 hotels with address and average cost per night
        Retun the response using markdown.
        """
        input_plan = st.text_area("Provide location to find Hotel & Restaurants!")
        #Button
        submit1 = st.button("Find Restaurant & Hotel!")
        if submit1:
            response = get_response_planner(input_prompt_planner, input_plan)
            st.subheader("Acomodation Bot: ")
            st.write(response)

if selected == "Personal Nutritionist":
    st.image("nutrition.jpg", width=120)
    st.title("👩🏻‍⚕️ 🍽 Health: Nutrition Calculator & Diet Planner")
    section_choice1 = st.radio("**Choose Section:**", ("Nutrition Calculator","Diet Planner"))
    if section_choice1 == "Nutrition Calculator":
        upload_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
        image = ""
        if upload_file is not None:
        #Show the image
            image = Image.open(upload_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        input_prompt_nutrition = """
        You are an expert Nutritionist. As a skilled nutritionist, you're required to analyze the food iems
        in the image and determine the total nutrition value. 
        Additionally, you need to furnish a breakdown of each food item along with its respective content.

        Food item, Serving size, Tatal Cal., Protien (g), Fat,
        Carb (g), Fiber (g), Vit B-12, Vit B-6,
        Iron, Zinc, Mang.

        Use a table to show above informaion.
        """
        submit = st.button("Calculate Nutrition value!")
        if submit:
            image_data = prep_image(upload_file)
            response = get_response_nutrition(image_data, input_prompt_nutrition)
            st.subheader("Nutrition AI: ")
            st.write(response)
    if section_choice1 == "Diet Planner":
        input_prompt_diet = """
        You are an expert Nutritionist. 
        If the input contains list of items like fruits or vegetables, you have to give diet plan and suggest
        breakfast, lunch, dinner wrt given item.
        If the input contains numbers, you have to suggest diet plan for breakfast, luncg=h, dinner within
        given number of calorie for the whole day.

        Return the response using markdown.

        """
        input_diet = st.text_area(" Input the list of items that you have at home and get diet plan! OR \
                              Input how much calorie you want to intake perday?:")
        submit1 = st.button("Plan my Diet!")
        if submit1:
            response = get_response_diet(input_prompt_diet, input_diet)
            st.subheader("Diet AI: ")
            st.write(response)

if selected == "TravelBot":
    st.title("TravelMate.AI ✈️")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am Yatra Sevak.AI How can I help you?"),
        ]
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
    response = get_response_travel(user_query, st.session_state.chat_history)
    response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()
    with st.chat_message("AI"):
        st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))
