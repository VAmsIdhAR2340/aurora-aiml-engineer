"""Data loader for Aurora API - uses skip/limit parameters"""
import requests
from typing import List, Dict, Optional
from requests.exceptions import RequestException, JSONDecodeError, Timeout
import logging
import time

logger = logging.getLogger(__name__)


class AuroraDataLoader:
    """
    Fetches and caches member messages from Aurora API
    
    Features:
    - Automatic caching to minimize API calls
    - Robust error handling for network and JSON errors
    - Retry logic with exponential backoff
    
    Note: Aurora API uses skip/limit parameters (not page/page_size)
    """
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the data loader"""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: Optional[List[Dict]] = None
    
    def fetch_messages(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch ALL messages from Aurora API using skip/limit
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of all message dictionaries (all 3349 messages)
        """
        if self._cache and not force_refresh:
            logger.info(f"Using cached data ({len(self._cache)} messages)")
            return self._cache
        
        # Retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Fetching messages from {self.base_url}/messages "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                
                # Use skip=0 and limit=10000 to get ALL messages at once
                response = requests.get(
                    f"{self.base_url}/messages/",
                    params={'skip': 0, 'limit': 10000},  # Get all messages
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                # Parse JSON
                try:
                    data = response.json()
                except JSONDecodeError as json_err:
                    logger.error(f"Invalid JSON response: {json_err}")
                    raise ValueError("API returned invalid JSON")
                
                # Extract messages
                if isinstance(data, dict):
                    messages = data.get('items', [])
                    total = data.get('total', len(messages))
                    
                    logger.info(f"Fetched {len(messages)} messages (total: {total})")
                    
                    if not messages:
                        raise ValueError("No messages returned")
                    
                    # Cache and return
                    self._cache = messages
                    logger.info(f"Successfully fetched and cached {len(messages)} messages")
                    return messages
                    
                elif isinstance(data, list):
                    logger.info(f"Fetched {len(data)} messages")
                    
                    if not data:
                        raise ValueError("Empty list returned")
                    
                    self._cache = data
                    logger.info(f"Successfully fetched and cached {len(data)} messages")
                    return data
                    
                else:
                    raise ValueError(f"Unexpected response format: {type(data)}")
            
            except Timeout as timeout_err:
                logger.warning(f"Timeout (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception("Request timeout after all retries")
            
            except RequestException as req_err:
                logger.error(f"Request error (attempt {attempt}/{self.max_retries}): {req_err}")
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to fetch: {req_err}")
            
            except ValueError as val_err:
                logger.error(f"Data validation error: {val_err}")
                raise Exception(f"Invalid data: {val_err}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise Exception(f"Unexpected error: {e}")
        
        raise Exception("Failed to fetch messages")
    
    def get_messages_by_user(
        self,
        user_name: str,
        exact_match: bool = False
    ) -> List[Dict]:
        """Filter messages by specific user"""
        messages = self.fetch_messages()
        
        if exact_match:
            return [
                msg for msg in messages 
                if msg.get('user_name', '') == user_name
            ]
        else:
            user_lower = user_name.lower()
            return [
                msg for msg in messages 
                if user_lower in msg.get('user_name', '').lower()
            ]
    
    def get_messages_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Filter messages by date range"""
        messages = self.fetch_messages()
        filtered = messages
        
        if start_date:
            filtered = [
                msg for msg in filtered
                if msg.get('timestamp', '') >= start_date
            ]
        
        if end_date:
            filtered = [
                msg for msg in filtered
                if msg.get('timestamp', '') <= end_date
            ]
        
        return filtered
    
    def get_message_count(self) -> int:
        """Get total number of cached messages"""
        if self._cache:
            return len(self._cache)
        return 0
    
    def refresh_cache(self) -> None:
        """Force refresh the message cache"""
        logger.info("Forcing cache refresh")
        self.fetch_messages(force_refresh=True)
    
    def clear_cache(self) -> None:
        """Clear the message cache"""
        logger.info("Clearing message cache")
        self._cache = None
    
    def get_cache_status(self) -> Dict:
        """Get information about cache status"""
        return {
            'cached': self._cache is not None,
            'message_count': len(self._cache) if self._cache else 0,
            'api_url': f"{self.base_url}/messages"
        }
