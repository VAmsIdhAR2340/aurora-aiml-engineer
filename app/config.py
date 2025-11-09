"""Configuration management"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # Perplexity API Configuration
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
    AURORA_API_URL: str = os.getenv(
        "AURORA_API_URL", 
        "https://november7-730026606190.europe-west1.run.app"
    )
    
    # Model Configuration
    LLM_MODEL: str = "sonar"  #Perplexity offerd LLM Model name
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 200
    
    # Embeddings
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L3-v2"
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 10
    
    # Application Settings
    APP_NAME: str = "Aurora QA System"
    APP_VERSION: str = "1.0.0"
    
    @classmethod
    def validate(cls) -> None:
        if not cls.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
        
        if not cls.PERPLEXITY_API_KEY.startswith("pplx-"):
            raise ValueError("PERPLEXITY_API_KEY should start with 'pplx-'")


config = Config()
