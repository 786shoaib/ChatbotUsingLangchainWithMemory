import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Initialize model and parser
model = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True)
parser = StrOutputParser()

# Store for session histories in Streamlit session state
if 'store' not in st.session_state:
    st.session_state['store'] = {}

# Function to get session history from Streamlit's session state
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:
        st.session_state['store'][session_id] = InMemoryChatMessageHistory()
    return st.session_state['store'][session_id]

# Initialize Streamlit app
st.title("Gemini Chatbot with Memory")

# Define a session ID input and message input
session_id = st.sidebar.text_input("Enter Session ID:", "firstchat")

# Initialize user message in session state if not already initialized
if 'user_message' not in st.session_state:
    st.session_state['user_message'] = ''

# User input field (controlled via session state)
user_message = st.text_area("Your Message:", value=st.session_state['user_message'], key="user_input")

# Model and memory management setup
model_with_memory = RunnableWithMessageHistory(model, get_session_history)

# Initialize the download string for storing conversation
download_str = []

# Process user input and display model's response
if st.button('Send'):
    if user_message:
        # Send user message to the model and fetch the response
        config = {"configurable": {"session_id": session_id}}
        response = model_with_memory.invoke([HumanMessage(content=user_message)], config=config).content
        
        # Add to session history only once (Prevent duplicate)
        chat_history = get_session_history(session_id)
        
        # Only add to history if the message doesn't already exist
        if not any(msg.content == user_message for msg in chat_history.messages):
            chat_history.add_message(HumanMessage(content=user_message))
        if not any(msg.content == response for msg in chat_history.messages):
            chat_history.add_message(AIMessage(content=response))
        
        # Add the messages to the download string for downloading later
        download_str.append(f"You: {user_message}")
        download_str.append(f"AI: {response}")
        
        # Display the response once
        st.write(f"Model Response: {response}")
        
        # Clear user input field by resetting session state value (but not affecting history)
        st.session_state['user_message'] = ''  # Clear the message input field in the UI
        st.rerun()  # Rerun the app to refresh the input field

    else:
        st.write("Please enter a message to send.")


# Display session history for current session in an expander
with st.expander("Conversation History", expanded=True):
    chat_history = get_session_history(session_id)
    if chat_history.messages:
        for message in chat_history.messages:
            # Use columns to display user and AI messages differently
            if isinstance(message, HumanMessage):
                # Display user message on the left
                col1, col2 = st.columns([1, 6])  # User column and AI column
                with col1:
                    st.markdown(f"**You:**")
                with col2:
                    st.info(f"{message.content}", icon="🧐")  # User message
            elif isinstance(message, AIMessage):
                # Display AI message on the right
                col1, col2 = st.columns([6, 1])  # User column and AI column
                with col1:
                    st.success(f"{message.content}", icon="🤖")  # AI response
                with col2:
                    st.markdown(f"**AI:**")
    else:
        st.write("No messages yet. Start the conversation!")

# Allow the user to download the conversation history as a text file
if download_str:
    download_str = '\n'.join(download_str)
    st.download_button('Download Conversation', download_str, file_name="conversation.txt", mime="text/plain")
