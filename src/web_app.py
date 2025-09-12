"""Streamlit web interface for the multimodel RAG chatbot."""

import os
import sys
from pathlib import Path
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from multimodel_rag_chatbot.core.chatbot import MultimodelRAGChatbot
from multimodel_rag_chatbot.core.config import settings

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multimodel RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = MultimodelRAGChatbot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False


def main():
    """Main web interface."""
    st.title("🤖 Multimodel RAG Chatbot")
    st.markdown("Chat with your documents using multiple AI models!")
    
    # Sidebar for configuration and controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = st.session_state.chatbot.get_available_models()
        
        if available_models:
            selected_model = st.selectbox(
                "Select AI Model",
                options=list(available_models.keys()),
                format_func=lambda x: f"{x} - {available_models[x]}",
                index=0
            )
        else:
            st.error("❌ No AI models available!")
            st.info("Please configure API keys in your .env file")
            selected_model = None
        
        # RAG settings
        st.header("📚 RAG Settings")
        use_rag = st.checkbox("Enable RAG (Retrieval)", value=True)
        num_docs = st.slider("Documents to retrieve", 1, 10, 4)
        
        # Vector store info
        vector_info = st.session_state.chatbot.get_vector_store_info()
        st.metric("Documents in Vector Store", vector_info['count'])
        
        # Document upload and management
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'pptx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, PPTX, TXT, MD"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing documents..."):
                    process_uploaded_files(uploaded_files)
        
        # Load from directory
        directory_path = st.text_input(
            "Or load from directory",
            placeholder="/path/to/documents"
        )
        
        if directory_path and st.button("Load from Directory"):
            if Path(directory_path).exists():
                with st.spinner("Loading documents..."):
                    count = st.session_state.chatbot.load_documents(directory_path)
                    if count > 0:
                        st.success(f"✅ Loaded {count} document chunks")
                        st.session_state.documents_loaded = True
                        st.rerun()
                    else:
                        st.warning("⚠️ No documents found")
            else:
                st.error("Directory does not exist")
        
        # Clear documents
        if st.button("🧹 Clear All Documents", type="secondary"):
            st.session_state.chatbot.clear_documents()
            st.success("Documents cleared")
            st.rerun()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.chatbot.clear_chat_history()
            st.success("Chat history cleared")
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat display
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(message["query"])
                
                # Assistant message
                with st.chat_message("assistant"):
                    st.write(message["response"])
                    
                    # Show model and context info
                    if "model_used" in message:
                        st.caption(f"Model: {message['model_used']} | "
                                 f"Context docs: {message.get('context_count', 0)}")
                    
                    # Show sources if available
                    if message.get("sources"):
                        with st.expander("📚 Sources"):
                            for j, source in enumerate(message["sources"][:3], 1):
                                source_file = Path(source['source']).name
                                score = source.get('similarity_score', 0)
                                st.write(f"{j}. **{source_file}** (similarity: {score:.3f})")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not available_models:
                st.error("No AI models available. Please configure API keys.")
                return
            
            # Add user message to chat
            st.session_state.chat_history.append({"query": prompt, "role": "user"})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.chatbot.chat(
                        query=prompt,
                        model_id=selected_model,
                        use_rag=use_rag,
                        k=num_docs
                    )
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.write(result['response'])
                    
                    # Show model and context info
                    st.caption(f"Model: {result['model_used']} | "
                             f"Context docs: {len(result.get('context_used', []))}")
                    
                    # Show sources if available
                    if result.get('sources'):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result['sources'][:3], 1):
                                source_file = Path(source['source']).name
                                score = source.get('similarity_score', 0)
                                st.write(f"{i}. **{source_file}** (similarity: {score:.3f})")
                    
                    # Add to chat history
                    chat_entry = {
                        "query": prompt,
                        "response": result['response'],
                        "model_used": result['model_used'],
                        "context_count": len(result.get('context_used', [])),
                        "sources": result.get('sources', []),
                        "role": "assistant"
                    }
                    st.session_state.chat_history.append(chat_entry)
    
    with col2:
        # Information panel
        st.header("ℹ️ System Info")
        
        # Model status
        st.subheader("🧠 Available Models")
        if available_models:
            for model_id, description in available_models.items():
                status = "✅" if model_id == selected_model else "⚪"
                st.write(f"{status} **{model_id}**")
                st.caption(description)
        else:
            st.warning("No models configured")
        
        # Vector store status
        st.subheader("📊 Knowledge Base")
        st.metric("Documents", vector_info['count'])
        st.metric("Collection", vector_info['name'])
        
        # Configuration
        st.subheader("⚙️ Settings")
        st.write(f"**Chunk Size:** {settings.chunk_size}")
        st.write(f"**Chunk Overlap:** {settings.chunk_overlap}")
        st.write(f"**RAG Enabled:** {'✅' if use_rag else '❌'}")
        st.write(f"**Retrieval Count:** {num_docs}")


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the vector store."""
    processed_count = 0
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temp directory
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Process the directory
        count = st.session_state.chatbot.load_documents(temp_dir)
        processed_count += count
    
    if processed_count > 0:
        st.success(f"✅ Processed {processed_count} document chunks from {len(uploaded_files)} files")
        st.session_state.documents_loaded = True
    else:
        st.warning("⚠️ No content extracted from uploaded files")


if __name__ == "__main__":
    main()