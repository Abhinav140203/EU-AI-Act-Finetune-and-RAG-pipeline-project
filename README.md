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

## 🚀 Features

- **Interactive Q&A**: Ask questions about the EU AI Act in natural language
- **Source Attribution**: View the source chunks used to generate answers
- **Model Selection**: Choose between cloud and local models
- **Real-time Processing**: Get instant answers with context retrieval
- **Privacy Options**: Use local TinyLLaMA model for sensitive queries

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster local inference)
- Groq API key (for cloud model)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EU_Project
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
   - Check that `data/` directory contains the processed documents

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
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
EU_Project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .env                           # Environment variables
├── data/                          # Processed documents and datasets
├── embeddings/                    # FAISS vector database
├── tinyllama-euai-finetuned/      # Fine-tuned TinyLLaMA model
└── scripts/                       # Data processing and training scripts
    ├── extract_data.py           # Extract text from documents
    ├── chunk_text.py             # Split text into chunks
    ├── embed_and_store.py        # Create embeddings and FAISS index
    ├── generate_qa_dataset.py    # Generate Q&A training data
    ├── fine_tune_local.py        # Fine-tune TinyLLaMA model
    ├── rag_pipeline_tinyllama.py # RAG pipeline for TinyLLaMA
    └── run_rag_pipeline.py       # Run RAG pipeline
```

## 🔧 Development

### Data Processing Pipeline

1. **Extract Data**: `python scripts/extract_data.py`
2. **Chunk Text**: `python scripts/chunk_text.py`
3. **Create Embeddings**: `python scripts/embed_and_store.py`
4. **Generate Q&A Dataset**: `python scripts/generate_qa_dataset.py`

### Model Training

To fine-tune the TinyLLaMA model:

```bash
python scripts/fine_tune_local.py
```

### Testing RAG Pipeline

```bash
python scripts/run_rag_pipeline.py
```

## 🔒 Privacy and Security

- **Local Processing**: The TinyLLaMA model runs entirely on your local machine
- **No Data Transmission**: Local model queries don't send data to external servers
- **Environment Variables**: API keys are stored securely in `.env` file

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