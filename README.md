# AI Document QA System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-1C3C3C.svg)](https://langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM-FF6B00.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.22-00A3E0.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B.svg)](https://ai-document-app-d6ulqvgkdzmaukepd6ueor.streamlit.app/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/usman-official-ai/AI-Document-QA.svg)](https://github.com/usman-official-ai/AI-Document-QA/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/usman-official-ai/AI-Document-QA.svg)](https://github.com/usman-official-ai/AI-Document-QA/network)
[![GitHub issues](https://img.shields.io/github/issues/usman-official-ai/AI-Document-QA.svg)](https://github.com/usman-official-ai/AI-Document-QA/issues)  

  <img width="1536" height="1024" alt="ChatGPT Image Jul 22, 2026, 03_24_33 PM" src="https://github.com/user-attachments/assets/4ae9292c-b12c-4eeb-8962-0484acd2ec82" />  

    


An intelligent document question-answering system that allows users to upload PDF documents and ask questions using Groq's LLM (Llama 3). The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based solely on the uploaded documents.

## 🌐 Live Demo

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-document-app-d6ulqvgkdzmaukepd6ueor.streamlit.app/)

**👉 [Try the Live App Here](https://ai-document-app-d6ulqvgkdzmaukepd6ueor.streamlit.app/)**

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Upload** | Upload one or multiple PDF documents |
| 🔍 **Document Processing** | Extract and split text from PDFs |
| 🧠 **Vector Search** | Store and search document embeddings using ChromaDB |
| 🤖 **Intelligent Q&A** | Ask questions powered by Groq's Llama 3 70B model |
| 📚 **Source Attribution** | View the source text used for each answer |
| 📝 **Chat History** | Track your question-answer history |
| 🎨 **Modern UI** | Clean and intuitive Streamlit interface |
| 🔒 **Data Privacy** | Your documents are not stored on servers |

## 🛠️ Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Frontend** | Streamlit | 1.28.0 |
| **LLM** | Groq (Llama 3 70B) | Latest |
| **Vector DB** | ChromaDB | 0.4.22 |
| **Embeddings** | Sentence Transformers | 2.2.2 |
| **Framework** | LangChain | 0.1.0 |
| **PDF Processing** | PyPDF2, pypdf | 3.0.1, 3.17.1 |

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key ([Get it here](https://console.groq.com/))
- pip package manager

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/usman-official-ai/AI-Document-QA.git
cd AI-Document-QA
```

### 2. Create virtual environment

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
# Linux/Mac
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Windows
echo GROQ_API_KEY=your_groq_api_key_here > .env
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. 📤 Upload Documents
- Click the file uploader in the sidebar
- Select one or more PDF files
- Click **"Upload & Process"**
- Wait for processing to complete

### 2. ❓ Ask Questions
- Type your question in the input field
- Click **"Ask"** or press Enter
- View the AI-generated answer
- Expand the source content to see the context used

### 3. ⚙️ Manage Session
- View processed files in the sidebar
- Reset the system to clear all documents
- Clear chat history as needed

## 🎯 Example Questions

| Type | Example Question |
|------|------------------|
| General | "What is this document about?" |
| Summary | "Summarize the main points" |
| Specific | "What are the steps mentioned in Section 3?" |
| Comparison | "Compare the different approaches presented" |
| Technical | "How does the system work?" |

## 🏗️ Project Structure

```
AI-Document-QA/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── pdf_loader.py          # PDF loading utilities
│   ├── text_splitter.py       # Text chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # ChromaDB operations
│   └── rag_pipeline.py        # RAG pipeline with Groq
│
├── data/                      # Uploaded PDFs storage
├── chroma_db/                 # Vector database storage
└── screenshots/               # Application screenshots
```

## 🔍 How It Works

### 1. Document Processing
- PDFs are loaded using PyPDF2
- Text is split into overlapping chunks
- Each chunk is processed for embedding

### 2. Embedding Generation
- Sentence Transformer creates embeddings
- Each chunk is converted to 384-dimension vector
- Embeddings are stored in ChromaDB

### 3. Question Answering
- User question is embedded
- Similar chunks retrieved via vector search
- Groq's Llama 3 generates answer from context

## 💡 Advanced Features

| Feature | Description |
|---------|-------------|
| **Document Metadata** | Track source and chunk IDs |
| **Relevance Scoring** | See how relevant each source is |
| **Persistent Storage** | Documents persist across sessions |
| **Error Handling** | Robust error handling and fallbacks |
| **Batch Processing** | Efficient document processing |

## 📊 Performance Tips

1. **Use smaller chunks** (500-800 words) for better precision
2. **Increase overlap** (200-300) for better context
3. **Limit n_results** to 3-5 for faster responses
4. **Use GPU** if available for faster embeddings

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository and branch
5. Set main file as `app.py`
6. Add your `GROQ_API_KEY` in Streamlit secrets
7. Click **"Deploy"**

### Environment Variables on Streamlit

Add this to your Streamlit secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### Deploy on Other Platforms

| Platform | Instructions |
|----------|--------------|
| **Heroku** | Add `Procfile` with `web: streamlit run app.py` |
| **AWS** | Use AWS Elastic Beanstalk with Python environment |
| **Google Cloud** | Deploy using Cloud Run |
| **Azure** | Deploy using Azure App Service |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

| Organization | Technology | Purpose |
|--------------|------------|---------|
| [Groq](https://groq.com/) | Llama 3 70B | LLM for answer generation |
| [Streamlit](https://streamlit.io/) | Framework | Web interface |
| [LangChain](https://langchain.com/) | Framework | RAG pipeline |
| [ChromaDB](https://www.trychroma.com/) | Vector DB | Vector storage |
| [Sentence Transformers](https://www.sbert.net/) | Embeddings | Text embeddings |

## 👤 Author

**Usman** ([@usman-official-ai](https://github.com/usman-official-ai))

[![GitHub](https://img.shields.io/badge/GitHub-usman--official--ai-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/usman-official-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/your-handle)

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/usman-official-ai/AI-Document-QA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/usman-official-ai/AI-Document-QA/discussions)
- **Email**: your-email@example.com

## 🌟 Show Your Support

If you found this project helpful, please give it a ⭐️ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/usman-official-ai/AI-Document-QA.svg?style=social)](https://github.com/usman-official-ai/AI-Document-QA/stargazers)

## 📊 Project Status

[![Active Development](https://img.shields.io/badge/Status-Active%20Development-success.svg)](https://github.com/usman-official-ai/AI-Document-QA)
[![Last Commit](https://img.shields.io/github/last-commit/usman-official-ai/AI-Document-QA.svg)](https://github.com/usman-official-ai/AI-Document-QA/commits/main)
[![Code Size](https://img.shields.io/github/languages/code-size/usman-official-ai/AI-Document-QA.svg)](https://github.com/usman-official-ai/AI-Document-QA)

---

**Made with ❤️ by Usman**
