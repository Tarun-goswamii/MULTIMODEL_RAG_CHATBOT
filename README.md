# Multimodel RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that supports multiple AI models and can process various document types to provide contextually aware responses.

## Features

- 🤖 **Multiple AI Models**: Support for OpenAI GPT models and Anthropic Claude models
- 📚 **Document Processing**: Process PDF, DOCX, PPTX, TXT, and Markdown files
- 🔍 **Vector Search**: Intelligent document retrieval using ChromaDB and embeddings
- 💬 **Dual Interface**: Both command-line and web-based interfaces
- ⚙️ **Configurable**: Flexible configuration for different models and settings
- 📊 **Rich Responses**: Responses include source attribution and similarity scores

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Tarun-goswamii/MULTIMODEL_RAG_CHATBOT.git
cd MULTIMODEL_RAG_CHATBOT

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Usage

#### Web Interface (Recommended)

```bash
# Start the web interface
python src/cli.py web

# Or directly with streamlit
streamlit run src/web_app.py
```

Then open http://localhost:8501 in your browser.

#### Command Line Interface

```bash
# Interactive chat
python src/cli.py chat --documents ./data/documents

# Load documents first
python src/cli.py load ./data/documents

# Then start chat
python src/cli.py chat

# Get system info
python src/cli.py info
```

## Usage Examples

### Loading Documents

Place your documents in the `data/documents` directory or any directory of your choice:

```
data/
└── documents/
    ├── research_paper.pdf
    ├── company_handbook.docx
    ├── presentation.pptx
    └── notes.txt
```

### Web Interface

1. Upload documents using the file uploader in the sidebar
2. Select your preferred AI model
3. Enable/disable RAG as needed
4. Start chatting with your documents!

### CLI Interface

```bash
# Load documents and start interactive chat
python src/cli.py chat -d ./data/documents -m openai

# Chat without RAG (direct model interaction)
python src/cli.py chat --no-rag

# Load documents into vector store
python src/cli.py load ./path/to/documents

# Check system status
python src/cli.py info

# Clear all documents
python src/cli.py clear
```

## Supported File Types

- **PDF** (.pdf) - Extracts text from PDF documents
- **Word** (.docx) - Processes Word documents
- **PowerPoint** (.pptx) - Extracts text from presentations
- **Text** (.txt) - Plain text files
- **Markdown** (.md) - Markdown formatted files

## Available AI Models

### OpenAI Models
- `openai` - GPT-3.5 Turbo (default)
- `openai-gpt4` - GPT-4

### Anthropic Models
- `anthropic` - Claude 3 Haiku
- `anthropic-sonnet` - Claude 3 Sonnet

## Configuration Options

The system can be configured through environment variables or the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_MODEL` | Default model to use | `openai` |
| `DEFAULT_EMBEDDING_MODEL` | Embedding model for vector search | `sentence-transformers/all-MiniLM-L6-v2` |
| `VECTOR_DB_PATH` | Path to store vector database | `./data/vectordb` |
| `CHUNK_SIZE` | Size of text chunks for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between text chunks | `200` |
| `WEB_PORT` | Port for web interface | `8501` |

## Architecture

```
├── src/
│   ├── multimodel_rag_chatbot/
│   │   ├── core/
│   │   │   ├── chatbot.py          # Main chatbot implementation
│   │   │   ├── config.py           # Configuration management
│   │   │   ├── document_processor.py # Document processing
│   │   │   └── vector_store.py     # Vector database operations
│   │   ├── models/
│   │   │   └── model_manager.py    # AI model management
│   │   └── __init__.py
│   ├── cli.py                      # Command-line interface
│   └── web_app.py                  # Streamlit web interface
├── data/
│   ├── documents/                  # Place your documents here
│   └── vectordb/                   # Vector database storage
├── requirements.txt
├── .env.example
└── README.md
```

## How It Works

1. **Document Processing**: Documents are processed and split into chunks for better retrieval
2. **Vector Storage**: Text chunks are embedded and stored in ChromaDB for efficient similarity search
3. **Query Processing**: User queries are used to retrieve relevant document chunks
4. **Response Generation**: Retrieved context is combined with the query and sent to the selected AI model
5. **Response Enhancement**: The AI model generates a response based on the retrieved context

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **No models available**: Ensure you have configured at least one API key in your `.env` file
2. **Document processing errors**: Check that your documents are in supported formats and not corrupted
3. **Vector store issues**: Try clearing the vector store with `python src/cli.py clear`
4. **Permission errors**: Ensure the application has write permissions to the `data/` directory

### Getting Help

- Check the console output for detailed error messages
- Use `python src/cli.py info` to check system status
- Ensure all dependencies are installed correctly

## Roadmap

- [ ] Support for more document types (HTML, CSV, etc.)
- [ ] Integration with cloud storage providers
- [ ] Advanced search and filtering options
- [ ] Chat history persistence
- [ ] API endpoint for integration with other applications
- [ ] Support for local/offline models
- [ ] Advanced document preprocessing options