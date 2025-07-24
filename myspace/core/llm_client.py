"""
LLM Client for OpenAI API integration
"""
import json
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import List, Dict, Any
import tiktoken


class LLMClient:
    """
    OpenAI API client with retry logic and token management
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the LLM client with configuration
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.embedding_model = config["model"]["EMBEDDING_MODEL"]
        self.gpt_model = config["model"]["GPT_MODEL"]
        self.batch_size = config["batch_size"]
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=config["api_key"],
            base_url=config["api_base"]
        )
        
        # Initialize tokenizer for the GPT model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.gpt_model)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Process in batches to avoid API limits
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_text(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.0,
                     max_tokens: int = 2048) -> str:
        """
        Generate text using the chat completion API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def filter_table_content(self, 
                           table_content: str, 
                           query: str, 
                           max_rows: int = 50) -> str:
        """
        Use LLM to filter table content based on query relevance
        
        Args:
            table_content: String representation of table
            query: User query
            max_rows: Maximum number of rows to keep
            
        Returns:
            Filtered table content
        """
        prompt = f"""
You are an expert at analyzing tables and filtering relevant information.

Given the following table and query, please filter the table to keep only the most relevant rows for answering the query.
Keep at most {max_rows} rows that are most likely to help answer the query.

Query: {query}

Table:
{table_content}

Please return the filtered table in the same format, keeping the header and the most relevant rows.
If no rows are relevant, return just the header.
"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that filters table data based on query relevance."},
            {"role": "user", "content": prompt}
        ]
        
        return self.generate_text(messages, temperature=0.0)
    
    def clarify_table_terms(self, table_content: str) -> str:
        """
        Use LLM to clarify domain-specific terms in the table
        
        Args:
            table_content: String representation of table
            
        Returns:
            Table with clarified terms
        """
        prompt = f"""
Analyze the following table and provide clarifications for any domain-specific terms, abbreviations, or ambiguous column names.
Add explanations in parentheses where necessary to make the content more understandable.

Table:
{table_content}

Please return the table with clarifications added where needed.
"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that clarifies technical terms in tables."},
            {"role": "user", "content": prompt}
        ]
        
        return self.generate_text(messages, temperature=0.0)
