"""Command-line interface for the multimodel RAG chatbot."""

import os
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodel_rag_chatbot.core.chatbot import MultimodelRAGChatbot
from multimodel_rag_chatbot.core.config import settings

# Load environment variables
load_dotenv()

console = Console()


@click.group()
def cli():
    """Multimodel RAG Chatbot CLI"""
    pass


@cli.command()
@click.option('--documents', '-d', type=click.Path(exists=True), 
              help='Path to documents directory')
@click.option('--model', '-m', type=str, help='Model to use for responses')
@click.option('--no-rag', is_flag=True, help='Disable RAG (no context retrieval)')
def chat(documents, model, no_rag):
    """Start an interactive chat session."""
    console.print(Panel.fit("🤖 Multimodel RAG Chatbot", style="bold blue"))
    
    # Initialize chatbot
    chatbot = MultimodelRAGChatbot()
    
    # Load documents if provided
    if documents:
        console.print(f"\n📁 Loading documents from: {documents}")
        count = chatbot.load_documents(documents)
        if count > 0:
            console.print(f"✅ Loaded {count} document chunks", style="green")
        else:
            console.print("⚠️ No documents loaded", style="yellow")
    
    # Show available models
    available_models = chatbot.get_available_models()
    if available_models:
        console.print("\n🧠 Available Models:")
        model_table = Table(show_header=True, header_style="bold magenta")
        model_table.add_column("Model ID", style="cyan")
        model_table.add_column("Description", style="white")
        
        for model_id, description in available_models.items():
            model_table.add_row(model_id, description)
        
        console.print(model_table)
        
        if model and model not in available_models:
            console.print(f"⚠️ Model '{model}' not available. Using default.", style="yellow")
            model = None
    else:
        console.print("❌ No models available. Please configure API keys.", style="red")
        return
    
    # Show vector store info
    vector_info = chatbot.get_vector_store_info()
    console.print(f"\n📊 Vector Store: {vector_info['count']} documents indexed")
    
    # Interactive chat loop
    console.print("\n💬 Chat started! Type 'quit' or 'exit' to stop.")
    console.print("Commands: /help, /models, /info, /clear, /history")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("👋 Goodbye!", style="yellow")
                break
            
            # Handle commands
            if user_input.startswith('/'):
                handle_command(user_input, chatbot)
                continue
            
            # Generate response
            console.print("🤔 Thinking...", style="dim")
            
            result = chatbot.chat(
                query=user_input,
                model_id=model,
                use_rag=not no_rag
            )
            
            # Display response
            if 'error' in result:
                console.print(f"❌ Error: {result['error']}", style="red")
            else:
                # Show model used
                model_info = f"Model: {result['model_used']} ({result.get('model_description', 'Unknown')})"
                if result['use_rag'] and result['context_used']:
                    model_info += f" | Context: {len(result['context_used'])} docs"
                
                console.print(f"[dim]{model_info}[/dim]")
                
                # Show response
                response_panel = Panel(
                    Markdown(result['response']),
                    title="🤖 Assistant",
                    border_style="green"
                )
                console.print(response_panel)
                
                # Show sources if available
                if result.get('sources'):
                    console.print("\n📚 Sources:", style="dim")
                    for i, source in enumerate(result['sources'][:3], 1):
                        source_file = Path(source['source']).name
                        score = source.get('similarity_score', 0)
                        console.print(f"  {i}. {source_file} (score: {score:.3f})", style="dim")
        
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="yellow")
            break
        except Exception as e:
            console.print(f"❌ Error: {str(e)}", style="red")


def handle_command(command, chatbot):
    """Handle CLI commands."""
    cmd = command.lower().strip()
    
    if cmd == '/help':
        help_text = """
**Available Commands:**
- `/help` - Show this help message
- `/models` - List available models
- `/info` - Show vector store information
- `/clear` - Clear chat history
- `/history` - Show chat history
- `quit`, `exit`, `bye` - Exit the chat
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))
    
    elif cmd == '/models':
        models = chatbot.get_available_models()
        if models:
            console.print("\n🧠 Available Models:")
            for model_id, description in models.items():
                console.print(f"  • {model_id}: {description}")
        else:
            console.print("❌ No models available")
    
    elif cmd == '/info':
        info = chatbot.get_vector_store_info()
        console.print(f"\n📊 Vector Store Info:")
        console.print(f"  • Collection: {info['name']}")
        console.print(f"  • Documents: {info['count']}")
        console.print(f"  • Metadata: {info.get('metadata', {})}")
    
    elif cmd == '/clear':
        chatbot.clear_chat_history()
        console.print("🧹 Chat history cleared", style="green")
    
    elif cmd == '/history':
        history = chatbot.get_chat_history()
        if history:
            console.print(f"\n📜 Chat History ({len(history)} interactions):")
            for i, entry in enumerate(history[-5:], 1):  # Show last 5
                console.print(f"\n{i}. Q: {entry['query'][:50]}...")
                console.print(f"   A: {entry['response'][:100]}...")
                console.print(f"   Model: {entry['model_used']}")
        else:
            console.print("📜 No chat history")
    
    else:
        console.print(f"❓ Unknown command: {command}", style="yellow")


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--collection', '-c', default='multimodel_rag', 
              help='Collection name for vector store')
def load(directory, collection):
    """Load documents from a directory into the vector store."""
    console.print(f"📁 Loading documents from: {directory}")
    
    chatbot = MultimodelRAGChatbot(collection_name=collection)
    count = chatbot.load_documents(directory)
    
    if count > 0:
        console.print(f"✅ Successfully loaded {count} document chunks", style="green")
    else:
        console.print("⚠️ No documents found or loaded", style="yellow")


@cli.command()
@click.option('--collection', '-c', default='multimodel_rag',
              help='Collection name for vector store')
def info(collection):
    """Show information about the vector store."""
    chatbot = MultimodelRAGChatbot(collection_name=collection)
    
    # Vector store info
    vector_info = chatbot.get_vector_store_info()
    console.print(f"\n📊 Vector Store Info:")
    console.print(f"  • Collection: {vector_info['name']}")
    console.print(f"  • Documents: {vector_info['count']}")
    
    # Available models
    models = chatbot.get_available_models()
    console.print(f"\n🧠 Available Models ({len(models)}):")
    for model_id, description in models.items():
        console.print(f"  • {model_id}: {description}")
    
    # Configuration
    console.print(f"\n⚙️ Configuration:")
    console.print(f"  • Default Model: {settings.default_model}")
    console.print(f"  • Embedding Model: {settings.default_embedding_model}")
    console.print(f"  • Vector DB Path: {settings.vector_db_path}")
    console.print(f"  • Chunk Size: {settings.chunk_size}")


@cli.command()
@click.option('--collection', '-c', default='multimodel_rag',
              help='Collection name for vector store')
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
def clear(collection):
    """Clear all documents from the vector store."""
    chatbot = MultimodelRAGChatbot(collection_name=collection)
    chatbot.clear_documents()
    console.print("🧹 All documents cleared from vector store", style="green")


@cli.command()
def web():
    """Launch the web interface."""
    console.print("🌐 Starting web interface...")
    console.print(f"🔗 Will be available at: http://localhost:{settings.web_port}")
    
    # Import and run the web app
    try:
        import subprocess
        import sys
        
        # Get the path to the web app
        web_app_path = Path(__file__).parent / "web_app.py"
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_app_path), 
            "--server.port", str(settings.web_port),
            "--server.address", "localhost"
        ])
    except Exception as e:
        console.print(f"❌ Error starting web interface: {str(e)}", style="red")
        console.print("💡 Try running: streamlit run src/web_app.py", style="blue")


if __name__ == "__main__":
    cli()