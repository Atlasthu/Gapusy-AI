import streamlit as st
from groq import Groq
import os

# ==========================================
# 1. SECURE API KEY CONFIGURATION
# ==========================================
if "GROQ_API_KEY" in os.environ:
    api_key = os.environ["GROQ_API_KEY"]
else:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("🔑 Groq API Key missing! Please add it to your Streamlit Advanced Settings / Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# ==========================================
# 2. WEB PAGE INTERFACE DESIGN
# ==========================================
st.set_page_config(page_title="My Custom Groq AI", page_icon="🤖", layout="centered")
st.title("🤖 My First Custom AI Chatbot")
st.caption("Built with Groq LPU Hardware, GitHub, and Streamlit")

# ==========================================
# 3. MULTI-USER MEMORY (Session State)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==========================================
# 4. CHAT LOGIC AND RESPONSE STREAMING
# ==========================================
if user_query := st.chat_input("Ask your AI anything..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # Bulletproof streaming function
            def generate_tokens():
                for chunk in stream:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                            if choice.delta.content:
                                yield choice.delta.content

            response_text = st.write_stream(generate_tokens())
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
