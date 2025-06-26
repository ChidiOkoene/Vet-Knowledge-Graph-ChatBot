# veterinary_chatbot_gui.py
import streamlit as st
from veterinary_chatbot import VeterinaryChatbot  # Import your chatbot class
import json
import time
import uuid
import os
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
from streamlit_chat import message

# Initialize the chatbot
def init_chatbot():
    return VeterinaryChatbot(
        neo4j_uri=st.secrets["NEO4J_URI"],
        neo4j_user=st.secrets["NEO4J_USER"],
        neo4j_password=st.secrets["NEO4J_PASSWORD"],
        neo4j_db=st.secrets["NEO4J_DB"],
        faiss_index_dir=st.secrets["FAISS_INDEX_DIR"],
        k=5,
        memory_window=10
    )

# Session state initialization
def init_session_state():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = init_chatbot()
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = st.session_state.chatbot.create_session()
        
    if "session_history" not in st.session_state:
        st.session_state.session_history = {}
        
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
        
    if "thinking" not in st.session_state:
        st.session_state.thinking = False

# Page configuration
st.set_page_config(
    page_title="VetAI Assistant",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styles */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Chat containers */
    .user-message {
        background-color: #e3f2fd !important;
        border-radius: 20px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
    }
    
    .bot-message {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 20px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Source citations */
    .source-badge {
        background-color: #e0f7fa;
        border-radius: 12px;
        padding: 4px 8px;
        margin: 4px;
        font-size: 0.8em;
        display: inline-block;
    }
    
    /* Thought process */
    .thought-process {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9em;
    }
    
    /* Sidebar */
    .sidebar .block-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Sidebar - Session Management
with st.sidebar:
    st.title("🐾 VetAI Assistant")
    st.subheader("Session Management")
    
    # Create new session
    if st.button("➕ New Chat Session"):
        new_session_id = st.session_state.chatbot.create_session()
        st.session_state.current_session = new_session_id
        st.experimental_rerun()
    
    # Session selector
    sessions = st.session_state.chatbot.list_sessions()
    selected_session = st.selectbox(
        "Active Sessions", 
        sessions, 
        index=sessions.index(st.session_state.current_session) if sessions else 0
    )
    
    if selected_session and selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        st.experimental_rerun()
    
    st.divider()
    
    # Knowledge sources info
    st.subheader("Knowledge Sources")
    st.markdown(f"**Neo4j Database:** `{st.secrets['NEO4J_DB']}`")
    st.markdown(f"**FAISS Indices:** `{st.secrets['FAISS_INDEX_DIR']}`")
    st.markdown(f"**LLM Models:** `{', '.join(['llama3:70b', 'mixtral:8x22b'])}`")
    
    st.divider()
    
    # Advanced options
    st.subheader("Advanced Options")
    st.session_state.temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7)
    st.session_state.show_thoughts = st.toggle("Show Thought Process", True)
    st.session_state.show_sources = st.toggle("Show Sources", True)

# Main chat interface
def main():
    colored_header(
        label="Veterinary AI Assistant",
        description="Ask about animal diseases, treatments, or clinical protocols",
        color_name="blue-70"
    )
    
    # Display chat history
    history = st.session_state.chatbot.get_session_history(st.session_state.current_session)
    chat_history = history["history"] if history else []
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for msg in chat_history:
            if msg.type == "human":
                message(msg.content, is_user=True, key=f"user_{msg.additional_kwargs['timestamp']}")
            elif msg.type == "ai":
                message(msg.content, key=f"ai_{msg.additional_kwargs['timestamp']}")
                
                # Display thought process if enabled
                if st.session_state.show_thoughts:
                    thought_chain = history.get("thought_chain", [])
                    with st.expander("🧠 Thought Process", expanded=False):
                        for thought in thought_chain:
                            if thought["phase"] == "retrieval":
                                st.markdown(f"**🔍 Retrieval Phase**")
                                if thought.get("kg_entities"):
                                    st.markdown(f"- KG Entities: {', '.join(thought['kg_entities'][:3])}{'...' if len(thought['kg_entities']) > 3 else ''}")
                                if thought.get("chunk_ids"):
                                    st.markdown(f"- Document Chunks: {len(thought['chunk_ids'])} found")
                                
                            elif thought["phase"] == "refinement":
                                st.markdown(f"**🔄 Refinement Phase**")
                                st.caption("Original response refined for accuracy")
    
    # Chat input
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area(
            "You:", 
            value=st.session_state.user_input,
            placeholder="Ask about animal diseases, treatments, or clinical protocols...",
            height=100,
            key="input"
        )
        
        submit_button = st.form_submit_button(label="Send")
        
    if submit_button and user_input.strip():
        st.session_state.thinking = True
        st.session_state.user_input = ""
        
        # Add user message to history
        if history:
            history["history"].append({
                "type": "human",
                "content": user_input,
                "additional_kwargs": {"timestamp": str(time.time())}
            })
        
        # Generate response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(user_input, st.session_state.current_session)
            
            # Add AI response to history
            if history:
                history["history"].append({
                    "type": "ai",
                    "content": response,
                    "additional_kwargs": {
                        "timestamp": str(time.time()),
                        "sources": history["thought_chain"][-1].get("chunk_ids", [])[:3]
                    }
                })
        
        st.session_state.thinking = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()