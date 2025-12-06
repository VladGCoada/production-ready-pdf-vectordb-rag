from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

from openai import OpenAI  # Changed from AzureOpenAI
from sentence_transformers import SentenceTransformer
from typing import List
import os
import logging

logger = logging.getLogger(__name__)


class AIService:
    """AI service for embeddings and completions"""

    def __init__(self):
        """Initialize AI services with configuration"""
        try:
            # Regular OpenAI client
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Warm up model
            self._warmup()

        except Exception as e:
            logger.error(f"Error initializing AI service: {str(e)}")
            raise

    def _warmup(self):
        """Warm up the embedding model"""
        try:
            _ = self.embedding_model.encode("Warm up text")
        except Exception as e:
            logger.error(f"Error warming up model: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def get_completion(self, messages: List[dict],
                       max_tokens: int = 1000,
                       temperature: float = 0.2) -> str:
        """Get completion from OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Or "gpt-4", "gpt-3.5-turbo"
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise


# Initialize global AI service
ai_service = AIService()