import streamlit as st
from groq import Groq
import os

# ==========================================
# 1. SECURE API KEY CONFIGURATION
# ==========================================
# This looks for the hidden key stored safely in your hosting server's vault
if "GROQ_API_KEY" in os.environ:
    api_key = os.environ["GROQ_API_KEY"]
else:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

# Fallback error check if the key is missing entirely
if not api_key:
    st.error("🔑 Groq API Key missing! Please add it to your Streamlit Advanced Settings / Secrets.")
    st.stop()

# Initialize the Groq client connection
client = Groq(api_key=api_key)

# ==========================================
# 2. WEB PAGE INTERFACE DESIGN
# ==========================================
st.set_page_config(page_title="Gapusy AI", page_icon="🤖", layout="centered")
st.title("🤖 Gapusy AI")
st.caption("Made by Atlasthu AKA Dhruv Mishra")

# ==========================================
# 3. MULTI-USER MEMORY (Session State)
# ==========================================
# Keeps chat history isolated per user browser tab so histories don't mix
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw previous chat messages on page refresh
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==========================================
# 4. CHAT LOGIC AND RESPONSE STREAMING
# ==========================================
# Look for user input from the chat bar
if user_query := st.chat_input("Ask your AI anything..."):
    
    # Render user input to the screen and log to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate assistant streaming text block
    with st.chat_message("assistant"):
        try:
            # Query the fast Llama-3.3 architecture hosted on Groq
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,  # Tells the brain to stream text back token by token
            )
            
            # Helper function to unpack data chunks from the network stream
            def generate_tokens():
                for chunk in stream:
                    if chunk.choices.delta.content:
                        yield chunk.choices.delta.content

            # Streamlit writes the incoming tokens directly onto the UI in real time
            response_text = st.write_stream(generate_tokens())
            
            # Append the completed AI response back into historical context
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
