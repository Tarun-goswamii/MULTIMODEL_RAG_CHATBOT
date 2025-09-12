#!/usr/bin/env python3
"""Simplified web interface demo without Streamlit dependency."""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from multimodel_rag_chatbot.core.chatbot import MultimodelRAGChatbot


def create_simple_html_demo():
    """Create a simple HTML demo page."""
    
    # Initialize chatbot
    chatbot = MultimodelRAGChatbot()
    
    # Load sample documents
    docs_path = Path(__file__).parent.parent / "data" / "documents"
    if docs_path.exists():
        count = chatbot.load_documents(str(docs_path))
    else:
        count = 0
    
    # Get system info
    vector_info = chatbot.get_vector_store_info()
    models = chatbot.get_available_models()
    
    # Test a sample query
    test_result = chatbot.chat("What is RAG?", use_rag=True)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodel RAG Chatbot</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status.success {{ background-color: #d4edda; color: #155724; }}
        .status.warning {{ background-color: #fff3cd; color: #856404; }}
        .status.info {{ background-color: #d1ecf1; color: #0c5460; }}
        .feature {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .feature-icon {{
            font-size: 1.5em;
            margin-right: 10px;
            width: 30px;
        }}
        .demo-response {{
            background-color: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Multimodel RAG Chatbot</h1>
        <p>A powerful Retrieval-Augmented Generation chatbot supporting multiple AI models</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="feature">
                <span class="feature-icon">📚</span>
                <div>
                    <strong>Vector Store:</strong> {vector_info['name']}<br>
                    <span class="status {'success' if vector_info['count'] > 0 else 'warning'}">
                        {vector_info['count']} documents loaded
                    </span>
                </div>
            </div>
            
            <div class="feature">
                <span class="feature-icon">🧠</span>
                <div>
                    <strong>AI Models:</strong><br>
                    <span class="status {'success' if models else 'warning'}">
                        {len(models)} models available
                    </span>
                    {('<br><small>Configure API keys to enable models</small>' if not models else '')}
                </div>
            </div>
            
            <div class="feature">
                <span class="feature-icon">⚙️</span>
                <div>
                    <strong>Storage Type:</strong><br>
                    <span class="status info">{vector_info.get('metadata', {}).get('type', 'unknown')}</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Features</h2>
            <div class="feature">
                <span class="feature-icon">📄</span>
                <strong>Document Processing</strong><br>
                <small>PDF, DOCX, PPTX, TXT, Markdown</small>
            </div>
            
            <div class="feature">
                <span class="feature-icon">🔍</span>
                <strong>Vector Search</strong><br>
                <small>Intelligent document retrieval</small>
            </div>
            
            <div class="feature">
                <span class="feature-icon">🤖</span>
                <strong>Multiple Models</strong><br>
                <small>OpenAI, Anthropic support</small>
            </div>
            
            <div class="feature">
                <span class="feature-icon">💬</span>
                <strong>Dual Interface</strong><br>
                <small>Web and CLI interfaces</small>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>💬 Demo Response</h2>
        <p><strong>Query:</strong> "What is RAG?"</p>
        <div class="demo-response">
            {test_result['response'][:500]}{'...' if len(test_result['response']) > 500 else ''}
        </div>
        {f'<p><strong>Context used:</strong> {len(test_result.get("context_used", []))} documents</p>' if test_result.get("context_used") else ''}
    </div>
    
    <div class="card">
        <h2>🛠️ Getting Started</h2>
        <ol>
            <li><strong>Install dependencies:</strong> <code>pip install -r requirements.txt</code></li>
            <li><strong>Configure API keys:</strong> Copy <code>.env.example</code> to <code>.env</code> and add your keys</li>
            <li><strong>Run web interface:</strong> <code>streamlit run src/web_app.py</code></li>
            <li><strong>Or use CLI:</strong> <code>python src/cli.py chat --documents ./data/documents</code></li>
        </ol>
    </div>
    
    <div class="card">
        <h2>📁 Project Structure</h2>
        <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
├── src/multimodel_rag_chatbot/     # Core package
│   ├── core/                       # Core functionality
│   │   ├── chatbot.py              # Main chatbot class
│   │   ├── document_processor.py   # Document processing
│   │   ├── vector_store.py         # Vector database
│   │   └── config.py               # Configuration
│   └── models/                     # AI model management
├── data/documents/                 # Sample documents
├── requirements.txt                # Dependencies
├── .env.example                    # Configuration template
└── README.md                       # Documentation
        </pre>
    </div>
</body>
</html>
    """
    
    # Save to file
    output_file = Path(__file__).parent.parent / "demo.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Demo page created: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    demo_file = create_simple_html_demo()
    print(f"🌐 Open in browser: file://{demo_file}")