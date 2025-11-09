"""Lightweight QA Engine - No heavy ML models"""
import requests
from typing import List, Dict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.data_loader import AuroraDataLoader
from app.config import config

logger = logging.getLogger(__name__)

class QAEngineError(Exception):
    pass

class RAGQAEngine:
    """Lightweight RAG using TF-IDF (no sentence transformers)"""
    
    def __init__(self, data_loader: AuroraDataLoader):
        self.data_loader = data_loader
        
        # Perplexity config
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        self.perplexity_api_key = config.PERPLEXITY_API_KEY
        self.model = config.LLM_MODEL
        
        # Lightweight search
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.documents = []
        self.doc_vectors = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        try:
            logger.info("Loading messages...")
            messages = self.data_loader.fetch_messages()
            
            logger.info(f"Processing {len(messages)} messages...")
            for msg in messages:
                content = f"User: {msg['user_name']}\nDate: {msg['timestamp']}\nMessage: {msg['message']}"
                self.documents.append({
                    'content': content,
                    'metadata': msg
                })
            
            logger.info("Creating search index...")
            contents = [doc['content'] for doc in self.documents]
            self.doc_vectors = self.vectorizer.fit_transform(contents)
            
            logger.info(f"QA Engine ready with {len(self.documents)} documents!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise QAEngineError(f"Failed to initialize: {e}")
    
    def _get_relevant_docs(self, question: str, k: int = 10):
        """Find relevant documents using TF-IDF similarity"""
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def _call_perplexity(self, prompt: str) -> str:
        """Call Perplexity API"""
        headers = {
            'Authorization': f'Bearer {self.perplexity_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for a luxury concierge service. Answer concisely based on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(
                self.perplexity_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Perplexity error: {e}")
            raise QAEngineError(f"API call failed: {e}")
    
    def answer_question(self, question: str) -> str:
        """Answer question using RAG"""
        try:
            if not question.strip():
                return "Please provide a valid question."
            
            # Get relevant context
            relevant_docs = self._get_relevant_docs(question, k=10)
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            
            # Build prompt
            prompt = f"""Answer based on this context:

{context}

Question: {question}

Answer in 1-3 sentences. Be specific. If not in context, say so."""
            
            # Get answer
            answer = self._call_perplexity(prompt)
            return answer
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return "Error processing question. Please try again."
    
    def get_relevant_context(self, question: str, k: int = 5) -> List[Dict]:
        """Get relevant messages for debugging"""
        try:
            docs = self._get_relevant_docs(question, k)
            return [{'content': doc['content'], 'metadata': doc['metadata']} for doc in docs]
        except:
            return []
    
    def refresh_data(self) -> None:
        """Refresh data"""
        self.data_loader.refresh_cache()
        self._initialize()
    
    def get_stats(self) -> Dict:
        """Get stats"""
        return {
            'initialized': self.doc_vectors is not None,
            'documents_indexed': len(self.documents),
            'llm_model': self.model,
            'llm_provider': 'Perplexity AI'
        }
