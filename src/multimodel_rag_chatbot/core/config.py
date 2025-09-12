"""Configuration management for the multimodel RAG chatbot."""

import os
from typing import Dict, Any
try:
    from pydantic import BaseSettings, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    print("Warning: Pydantic not available. Using simple settings.")
    
    # Simple settings fallback
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
    
    def Field(default=None, env=None):
        return default


class Settings:
    """Application settings."""
    
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # Model Configuration
        self.default_model = os.getenv("DEFAULT_MODEL", "openai")
        self.default_embedding_model = os.getenv(
            "DEFAULT_EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector Database Configuration
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Web Interface Configuration
        self.web_port = int(os.getenv("WEB_PORT", "8501"))
        self.api_port = int(os.getenv("API_PORT", "8000"))

if HAS_PYDANTIC:
    class Settings(BaseSettings):
        """Application settings."""
        
        # API Keys
        openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
        anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
        
        # Model Configuration
        default_model: str = Field(default="openai", env="DEFAULT_MODEL")
        default_embedding_model: str = Field(
            default="sentence-transformers/all-MiniLM-L6-v2", 
            env="DEFAULT_EMBEDDING_MODEL"
        )
        
        # Vector Database Configuration
        vector_db_path: str = Field(default="./data/vectordb", env="VECTOR_DB_PATH")
        chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
        chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
        
        # Web Interface Configuration
        web_port: int = Field(default=8501, env="WEB_PORT")
        api_port: int = Field(default=8000, env="API_PORT")
        
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get configuration for available models."""
    return {
        "openai": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "api_key_env": "OPENAI_API_KEY",
            "description": "OpenAI GPT-3.5 Turbo"
        },
        "openai-gpt4": {
            "provider": "openai", 
            "model_name": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
            "description": "OpenAI GPT-4"
        },
        "anthropic": {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
            "api_key_env": "ANTHROPIC_API_KEY", 
            "description": "Anthropic Claude 3 Haiku"
        },
        "anthropic-sonnet": {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "api_key_env": "ANTHROPIC_API_KEY",
            "description": "Anthropic Claude 3 Sonnet"
        }
    }


# Global settings instance
settings = Settings()