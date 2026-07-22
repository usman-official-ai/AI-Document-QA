# AI Document QA System

An intelligent document question-answering system that allows users to upload PDF documents and ask questions using Groq's LLM (Llama 3 70B). The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based solely on the uploaded documents.

## 🚀 Features

- 📄 **PDF Upload**: Upload one or multiple PDF documents
- 🔍 **Document Processing**: Extract and split text from PDFs
- 🧠 **Vector Search**: Store and search document embeddings using ChromaDB
- 🤖 **Intelligent Q&A**: Ask questions powered by Groq's Llama 3 70B model
- 📚 **Source Attribution**: View the source text used for each answer
- 📝 **Chat History**: Track your question-answer history
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3 70B)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Framework**: LangChain
- **PDF Processing**: PyPDF2, PyPDFLoader

## 📋 Prerequisites

- Python 3.8+
- Groq API Key
- pip package manager

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-Document-QA.git
cd AI-Document-QA  
  
2. Create virtual environmentbash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies

bash
pip install -r requirements.txt


4. Set up environment variables
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
5. Run the application  
  streamlit run app.py  
    
    📖 Usage Guide
1. Upload Documents
Click the file uploader in the sidebar

Select one or more PDF files

Click "Upload & Process"

Wait for processing to complete

2. Ask Questions
Type your question in the input field

Click "Ask" or press Enter

View the AI-generated answer

Expand the source content to see the context used

3. Manage Session
View processed files in the sidebar

Reset the system to clear all documents

Clear chat history as needed

🎯 Example Questions
"What are the main topics discussed in this document?"

"Summarize the key findings in section 3"

"What are the steps mentioned for [specific process]?"

"Compare the different approaches presented"

🏗️ Project Structure
text
AI-Document-QA/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables
├── README.md             # Project documentation
├── data/                  # Uploaded PDFs storage
├── chroma_db/            # Vector database storage
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── pdf_loader.py      # PDF loading utilities
│   ├── text_splitter.py   # Text chunking
│   ├── embeddings.py      # Embedding generation
│   ├── vector_store.py    # ChromaDB operations
│   └── rag_pipeline.py    # RAG pipeline with Groq
└── screenshots/           # Application screenshots
🔍 How It Works
1. Document Processing
PDFs are loaded using PyPDFLoader

Text is split into overlapping chunks

Each chunk is processed for embedding

2. Embedding Generation
Sentence Transformer creates embeddings

Each chunk is converted to 384-dimension vector

Embeddings are stored in ChromaDB

3. Question Answering
User question is embedded

Similar chunks retrieved via vector search

Groq's Llama 3 generates answer from context

💡 Advanced Features
Document Metadata: Track source and chunk IDs

Relevance Scoring: See how relevant each source is

Persistent Storage: Documents persist across sessions

Error Handling: Robust error handling and fallbacks

Batch Processing: Efficient document processing

📊 Performance Tips
Use smaller chunks (500-800 words) for better precision

Increase overlap (200-300) for better context

Limit n_results to 3-5 for faster responses

Use GPU if available for faster embeddings

🤝 Contributing
Fork the repository

Create feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add some AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is for educational purposes.

🙏 Acknowledgments
Groq for the powerful LLM API

Streamlit for the web interface

LangChain for RAG framework

ChromaDB for vector storage

📧 Support
For issues and questions, please open an issue on GitHub.
  
  