"""
Main Streamlit Application for AI Document QA System
"""
import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

# Import utility modules - Using absolute imports
from utils.pdf_loader import PDFLoader
from utils.text_splitter import TextSplitter
from utils.embeddings import EmbeddingGenerator
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document QA System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B00;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #FF6B00;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #E55A00;
        color: white;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B00;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Initialize components with caching
@st.cache_resource
def init_pipeline():
    """Initialize RAG pipeline with all components"""
    try:
        pdf_loader = PDFLoader()
        text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(persist_directory="./chroma_db")
        pipeline = RAGPipeline(
            pdf_loader,
            text_splitter,
            embedding_generator,
            vector_store
        )
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        return None

# Initialize pipeline
if st.session_state.pipeline is None:
    with st.spinner("🔄 Initializing system..."):
        st.session_state.pipeline = init_pipeline()

# Main Header
st.markdown('<h1 class="main-header">📄 AI Document QA System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Groq LLM • Upload PDFs and Ask Questions</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pdf.png", width=80)
    st.markdown("---")
    
    st.markdown("### 📁 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents"
    )
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        
        if st.button("📤 Upload & Process", use_container_width=True):
            with st.spinner("⏳ Processing PDFs..."):
                try:
                    # Save uploaded files temporarily
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            pdf_paths.append(tmp_file.name)
                    
                    # Process PDFs
                    result = st.session_state.pipeline.process_pdfs(pdf_paths)
                    
                    if result['success']:
                        st.session_state.documents_processed = True
                        st.session_state.processing_complete = True
                        
                        # Store file names
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name not in st.session_state.processed_files:
                                st.session_state.processed_files.append(uploaded_file.name)
                        
                        st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs into {result['chunks']} chunks!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### 📋 Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"✅ {file}")
        
        # Get stats
        stats = st.session_state.pipeline.get_stats()
        if stats and 'vector_store' in stats:
            doc_count = stats['vector_store'].get('document_count', 0)
            st.metric("📊 Total Chunks", doc_count)
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset System", use_container_width=True):
        if st.session_state.pipeline:
            st.session_state.pipeline.reset()
        st.session_state.documents_processed = False
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.session_state.processing_complete = False
        st.success("✅ System reset successfully!")
        st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("### 📊 System Status")
    if st.session_state.documents_processed:
        st.success("✅ Documents loaded and indexed")
    else:
        st.info("⏳ No documents loaded yet")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Built with ❤️ using<br>
        Streamlit • LangChain • Groq
    </div>
    """, unsafe_allow_html=True)

# Main Chat Interface
st.markdown("### 💬 Ask Questions About Your Documents")

# Question input
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What are the main topics discussed in this document?",
    key="question_input",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_button = st.button("🔍 Ask", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear Chat", use_container_width=True)

# Clear chat history
if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# Process question
if ask_button and question:
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get answer
                result = st.session_state.pipeline.ask_question(question, n_results=3)
                
                if result.get('error', False):
                    st.error(result['answer'])
                else:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result['answer'],
                        "sources": result['sources']
                    })
                    
                    # Display answer
                    st.markdown("### 🤖 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if result['sources']:
                        st.markdown("### 📚 Source Content")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"📖 Source {i} (Relevance: {source['relevance_score']:.3f})"):
                                st.text_area(
                                    "Content",
                                    source['content'],
                                    height=150,
                                    key=f"source_{i}_{len(st.session_state.chat_history)}",
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                                if source.get('metadata'):
                                    st.caption(f"Metadata: {source['metadata']}")
                                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📝 Chat History")
    
    # Show latest 10 conversations
    history_to_show = st.session_state.chat_history[-10:]
    for i, chat in enumerate(reversed(history_to_show)):
        with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}"):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            if chat.get('sources'):
                st.markdown("**Sources:**")
                for j, source in enumerate(chat['sources'][:3], 1):
                    st.caption(f"Source {j}: {source['content'][:200]}...")