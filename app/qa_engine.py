"""RAG-based Question Answering Engine using Perplexity API"""
import os
from typing import List, Dict, Optional
import logging
import traceback
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.data_loader import AuroraDataLoader
from app.config import config

logger = logging.getLogger(__name__)


class QAEngineError(Exception):
    """Custom exception for QA Engine errors"""
    pass


class RAGQAEngine:
    """RAG QA Engine using Perplexity API directly"""
    
    def __init__(self, data_loader: AuroraDataLoader):
        self.data_loader = data_loader
        
        try:
            logger.info("Initializing embeddings model (HuggingFace)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Initializing Perplexity LLM ({config.LLM_MODEL})...")
            # Stoing Perplexity configuration
            self.perplexity_url = "https://api.perplexity.ai/chat/completions"
            self.perplexity_api_key = config.PERPLEXITY_API_KEY
            self.model = config.LLM_MODEL
            
            self.vectorstore: Optional[FAISS] = None
            self._initialize()
            
        except Exception as e:
            logger.error(f"Failed to initialize QA engine: {e}")
            logger.error(traceback.format_exc())
            raise QAEngineError(f"Initialization failed: {e}")
    
    def _initialize(self) -> None:
        try:
            logger.info("Initializing vector store...")
            self._build_vectorstore()
            logger.info("QA Engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise QAEngineError(f"Failed to initialize: {e}")
    
    def _build_vectorstore(self) -> None:
        try:
            messages = self.data_loader.fetch_messages()
            
            if not messages:
                raise QAEngineError("No messages available")
            
            documents = []
            skipped_count = 0
            
            for idx, msg in enumerate(messages):
                try:
                    user_name = msg.get('user_name', 'Unknown User')
                    timestamp = msg.get('timestamp', 'Unknown Date')
                    message_text = msg.get('message', '')
                    
                    if not message_text or not message_text.strip():
                        skipped_count += 1
                        continue
                    
                    content = (
                        f"User: {user_name}\n"
                        f"Date: {timestamp}\n"
                        f"Message: {message_text}"
                    )
                    
                    metadata = {
                        'user_name': user_name,
                        'user_id': msg.get('user_id', 'unknown'),
                        'timestamp': timestamp,
                        'message_id': msg.get('id', f'msg_{idx}')
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                    
                except Exception:
                    skipped_count += 1
                    continue
            
            if not documents:
                raise QAEngineError("No valid documents created")
            
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} messages")
            
            logger.info(f"Creating vector store with {len(documents)} documents...")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except QAEngineError:
            raise
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            logger.error(traceback.format_exc())
            raise QAEngineError(f"Failed to build vector store: {e}")
    
    def _call_perplexity(self, prompt: str) -> str:
        """Call Perplexity API directly (no LangChain)"""
        headers = {
            'Authorization': f'Bearer {self.perplexity_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                    "You are a helpful assistant for a luxury concierge service. "
                    "Answer questions about member messages accurately, concisely and shortly in one line. "
                    "Do NOT add citations, references, or [Context] markers. "
                    "Provide a clean, natural answer."
                )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS
        }
        
        try:
            logger.info("Calling Perplexity API...")
            response = requests.post(
                self.perplexity_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content']
            logger.info("Perplexity API call successful")
            return answer
            
        except requests.RequestException as e:
            logger.error(f"Perplexity API error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise QAEngineError(f"Perplexity API call failed: {e}")
    
    def answer_question(self, question: str) -> str:
        """Answer a question using RAG with Perplexity"""
        try:
            if not question or not question.strip():
                return "Please provide a valid question."
            
            if not self.vectorstore:
                raise QAEngineError("Vector store not initialized")
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get relevant context from vector store
            docs = self.vectorstore.similarity_search(
                question, 
                k=config.TOP_K_RESULTS
            )
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt with context
            prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Instructions:
- Answer in 1-3 clear sentences
- Include specific details (dates, numbers, names) when available
- Only use information from the provided context
- If the answer is not in the context, say "I don't have that information in the available messages"
- Be direct and professional

Answer:"""
            
            # Call Perplexity
            answer = self._call_perplexity(prompt)
            
            logger.info("Answer generated successfully")
            return answer.strip()
        
        except QAEngineError:
            logger.error("QA Engine error occurred")
            raise
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            logger.error(traceback.format_exc())
            
            if "rate limit" in str(e).lower():
                return "I'm experiencing high demand. Please try again in a moment."
            elif "timeout" in str(e).lower():
                return "The request took too long. Please try a simpler question."
            else:
                return "I encountered an error processing your question. Please try rephrasing."
    
    def get_relevant_context(self, question: str, k: int = 5) -> List[Dict]:
        """Get relevant messages for debugging"""
        try:
            if not self.vectorstore:
                raise QAEngineError("Vector store not initialized")
            
            if not question or not question.strip():
                return []
            
            k = max(1, min(k, 50))
            docs = self.vectorstore.similarity_search(question, k=k)
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in docs
            ]
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []
    
    def refresh_data(self) -> None:
        """Refresh the vector store with latest data"""
        try:
            logger.info("Refreshing QA engine...")
            self.data_loader.refresh_cache()
            self._build_vectorstore()
            logger.info("QA engine refreshed!")
        except Exception as e:
            logger.error(f"Error refreshing: {e}")
            raise QAEngineError(f"Failed to refresh: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the QA engine"""
        try:
            doc_count = 0
            if self.vectorstore:
                doc_count = self.vectorstore.index.ntotal
            
            cache_status = self.data_loader.get_cache_status()
            
            return {
                'initialized': self.vectorstore is not None,
                'documents_indexed': doc_count,
                'messages_cached': cache_status['message_count'],
                'embedding_model': config.EMBEDDING_MODEL,
                'llm_model': self.model,
                'llm_provider': 'Perplexity AI',
                'retrieval_k': config.TOP_K_RESULTS
            }
        except Exception as e:
            return {'error': str(e)}
