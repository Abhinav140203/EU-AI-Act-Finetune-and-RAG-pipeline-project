# EU AI Act Q&A Assistant

A comprehensive question-answering system for the EU AI Act, powered by Retrieval-Augmented Generation (RAG) with support for both cloud-based (Groq LLaMA3) and local fine-tuned (TinyLLaMA) language models.

## 🎯 Project Overview

This application provides an interactive interface for users to ask questions about the EU AI Act and receive accurate, context-aware answers. The system uses:

- **RAG Pipeline**: Retrieves relevant context from the EU AI Act document
- **Dual Model Support**: 
  - Groq LLaMA3 (cloud-based, fast)
  - Fine-tuned TinyLLaMA (local, privacy-focused)
- **FAISS Vector Database**: For efficient similarity search
- **Streamlit UI**: User-friendly web interface
- **Docker Support**: Containerized deployment with health checks

## 🚀 Features

- **Interactive Q&A**: Ask questions about the EU AI Act in natural language
- **Source Attribution**: View the source chunks used to generate answers
- **Model Selection**: Choose between cloud and local models
- **Real-time Processing**: Get instant answers with context retrieval
- **Privacy Options**: Use local TinyLLaMA model for sensitive queries
- **Docker Deployment**: Easy containerized setup with volume persistence
- **Health Monitoring**: Built-in health checks for production deployment

## 📋 Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (optional, for faster local inference)
- Groq API key (for cloud model)
- Docker and Docker Compose (for containerized deployment)

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EU-AI-Act-Finetune-and-RAG-pipeline-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Download required models and data**
   - Ensure the `tinyllama-euai-finetuned/` directory contains the fine-tuned model
   - Verify the `embeddings/faiss_index/` directory contains the FAISS vector database
   - Check that `data/pdfs/` directory contains the EU AI Act PDF documents

### Option 2: Docker Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EU-AI-Act-Finetune-and-RAG-pipeline-project
   ```

2. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   Or run in detached mode:
   ```bash
   docker-compose up -d --build
   ```

4. **Access the application**
   Open your browser and go to `http://localhost:8501`

## 🎮 Usage

### Running the Application

#### Local Installation
```bash
streamlit run app.py
```

#### Docker Installation
```bash
# Start the application
docker-compose up

# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up --build
```

The application will open in your browser at `http://localhost:8501`.

### Using the Interface

1. **Select Model**: Choose between "Groq (LLaMA3)" or "Fine-Tuned TinyLLaMA"
2. **Enter Question**: Type your question about the EU AI Act
3. **Get Answer**: Click "Get Answer" to receive a response
4. **View Sources**: Toggle "Show source chunks" to see the retrieved context

### Model Comparison

| Feature | Groq (LLaMA3) | Fine-Tuned TinyLLaMA |
|---------|---------------|---------------------|
| Speed | Fast (1-3 seconds) | Slower (30-90 seconds) |
| Privacy | Cloud-based | Local processing |
| Accuracy | High | Domain-specific |
| Cost | API usage | Free (local) |

## 📁 Project Structure

```
EU-AI-Act-Finetune-and-RAG-pipeline-project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── .dockerignore                  # Docker ignore file
├── .gitignore                     # Git ignore patterns
├── .env                           # Environment variables (create this)
├── data/                          # Data directory
│   └── pdfs/                      # EU AI Act PDF documents
│       ├── EU AI act1.pdf         # Primary EU AI Act document
│       └── EU AI Act2.pdf         # Additional EU AI Act document
├── embeddings/                    # FAISS vector database (auto-generated)
├── tinyllama-euai-finetuned/      # Fine-tuned TinyLLaMA model
├── logs/                          # Application logs (auto-created)
└── scripts/                       # Data processing and training scripts
    ├── extract_data.py           # Extract text from PDF documents
    ├── chunk_text.py             # Split text into chunks
    ├── embed_and_store.py        # Create embeddings and FAISS index
    ├── generate_qa_dataset.py    # Generate Q&A training data
    ├── rag_pipeline_tinyllama.py # RAG pipeline for TinyLLaMA
    ├── run_rag_pipeline.py       # Run RAG pipeline
    └── TinyLLaMA_EUAI_FineTune (1).ipynb  # Fine-tuning notebook
```

## 🔧 Development

### Data Processing Pipeline

1. **Extract Data**: `python scripts/extract_data.py`
2. **Chunk Text**: `python scripts/chunk_text.py`
3. **Create Embeddings**: `python scripts/embed_and_store.py`
4. **Generate Q&A Dataset**: `python scripts/generate_qa_dataset.py`

### Model Training

To fine-tune the TinyLLaMA model, use the Jupyter notebook:
```bash
jupyter notebook scripts/TinyLLaMA_EUAI_FineTune\ \(1\).ipynb
```

### Testing RAG Pipeline

```bash
python scripts/run_rag_pipeline.py
```

## 🐳 Docker Commands

### Build the image
```bash
docker build -t eu-ai-act-qa .
```

### Run the container
```bash
docker run -p 8501:8501 --env-file .env eu-ai-act-qa
```

### Run with volumes for data persistence
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/tinyllama-euai-finetuned:/app/tinyllama-euai-finetuned \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  eu-ai-act-qa
```

### View container logs
```bash
docker-compose logs -f eu-ai-act-qa
```

### Stop and remove containers
```bash
docker-compose down
```

### Check container health
```bash
docker-compose ps
```

## 🔒 Privacy and Security

- **Local Processing**: The TinyLLaMA model runs entirely on your local machine
- **No Data Transmission**: Local model queries don't send data to external servers
- **Environment Variables**: API keys are stored securely in `.env` file
- **Container Isolation**: Docker provides additional security isolation
- **Volume Persistence**: Data and models persist across container restarts

## 📊 Logging and Monitoring

The application includes comprehensive logging:
- **Application Logs**: Stored in `logs/` directory
- **Health Checks**: Docker health monitoring for production deployment
- **Error Tracking**: Detailed error logging for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EU AI Act document for the source material
- Hugging Face for the transformer models and libraries
- Groq for the cloud inference API
- Streamlit for the web interface framework

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This application is for educational and informational purposes. Always refer to the official EU AI Act document for legal compliance and official interpretations. 