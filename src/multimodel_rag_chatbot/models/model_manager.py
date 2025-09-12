"""Model implementations for different AI providers."""

import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    from langchain.llms.base import LLM
    from langchain.chat_models import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: LangChain OpenAI not available.")

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: LangChain Anthropic not available.")

# Simple model fallback
class SimpleModel:
    """Simple model fallback when APIs are not available."""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def predict(self, prompt):
        return f"[Mock Response from {self.model_name}] I'm a simple chatbot. The actual AI models are not available because API libraries are not installed. Please install the required dependencies to use real AI models."

from ..core.config import settings, get_available_models


class BaseModel(ABC):
    """Abstract base class for AI models."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available (API key configured)."""
        pass


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        self.client = None
        
        if not HAS_OPENAI:
            self.client = SimpleModel(model_name)
            return
        
        if self.is_available():
            try:
                self.client = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=settings.openai_api_key,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Error initializing OpenAI model: {str(e)}")
                self.client = SimpleModel(model_name)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.client:
            return "OpenAI model not available. Please check your API key."
        
        try:
            # Set temperature if provided
            if 'temperature' in kwargs:
                self.client.temperature = kwargs['temperature']
            
            response = self.client.predict(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return HAS_OPENAI and bool(settings.openai_api_key and settings.openai_api_key.strip())


class AnthropicModel(BaseModel):
    """Anthropic model implementation."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic model.
        
        Args:
            model_name: Name of the Anthropic model to use
        """
        self.model_name = model_name
        self.client = None
        
        if not HAS_ANTHROPIC:
            self.client = SimpleModel(model_name)
            return
        
        if self.is_available():
            try:
                self.client = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=settings.anthropic_api_key,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Error initializing Anthropic model: {str(e)}")
                self.client = SimpleModel(model_name)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.client:
            return "Anthropic model not available. Please check your API key."
        
        try:
            # Set temperature if provided
            if 'temperature' in kwargs:
                self.client.temperature = kwargs['temperature']
            
            response = self.client.predict(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        return HAS_ANTHROPIC and bool(settings.anthropic_api_key and settings.anthropic_api_key.strip())


class ModelManager:
    """Manage multiple AI models and provide unified interface."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.available_models = get_available_models()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models."""
        for model_id, config in self.available_models.items():
            try:
                if config["provider"] == "openai":
                    model = OpenAIModel(config["model_name"])
                elif config["provider"] == "anthropic":
                    model = AnthropicModel(config["model_name"])
                else:
                    continue
                
                if model.is_available():
                    self.models[model_id] = model
                    print(f"Initialized model: {model_id}")
                else:
                    print(f"Model {model_id} not available (missing API key)")
                    
            except Exception as e:
                print(f"Error initializing model {model_id}: {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a specific model by ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model instance or None if not available
        """
        return self.models.get(model_id)
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models.
        
        Returns:
            Dictionary of model_id -> description
        """
        available = {}
        for model_id in self.models.keys():
            if model_id in self.available_models:
                available[model_id] = self.available_models[model_id]["description"]
        return available
    
    def generate_response(self, prompt: str, model_id: str = None, **kwargs) -> str:
        """Generate response using specified model.
        
        Args:
            prompt: Input prompt
            model_id: ID of the model to use (defaults to configured default)
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Use default model if none specified
        if not model_id:
            model_id = settings.default_model
        
        model = self.get_model(model_id)
        if not model:
            # Try to use any available model
            if self.models:
                model_id = list(self.models.keys())[0]
                model = self.models[model_id]
                print(f"Using fallback model: {model_id}")
            else:
                return "No AI models available. Please configure API keys."
        
        return model.generate_response(prompt, **kwargs)
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is available.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        return model_id in self.models