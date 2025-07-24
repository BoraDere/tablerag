"""
Text processing utilities
"""
import re
import string
from typing import List, Dict, Any
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        
    Returns:
        List of keywords
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = text.split()
    
    # Filter by length and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    return keywords


def tokenize_text(text: str) -> List[str]:
    """
    Simple text tokenization
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Clean the text first
    text = clean_text(text)
    
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    return tokens


def highlight_matches(text: str, query: str) -> str:
    """
    Highlight query matches in text
    
    Args:
        text: Source text
        query: Query to highlight
        
    Returns:
        Text with highlighted matches
    """
    # Extract keywords from query
    query_keywords = extract_keywords(query)
    
    highlighted_text = text
    for keyword in query_keywords:
        # Case-insensitive replacement with highlighting
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{keyword}**", highlighted_text)
    
    return highlighted_text


def truncate_text_smart(text: str, max_length: int, preserve_words: bool = True) -> str:
    """
    Intelligently truncate text
    
    Args:
        text: Input text
        max_length: Maximum character length
        preserve_words: Whether to preserve word boundaries
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if preserve_words:
        # Find the last complete word within the limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Only if we don't lose too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
    else:
        return text[:max_length - 3] + "..."


def extract_date_patterns(text: str) -> List[str]:
    """
    Extract date patterns from text
    
    Args:
        text: Input text
        
    Returns:
        List of found date patterns
    """
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or M/D/YYYY
        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY or M-D-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD or YYYY-M-D
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
    ]
    
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_dates.extend(matches)
    
    return found_dates


def extract_time_patterns(text: str) -> List[str]:
    """
    Extract time patterns from text
    
    Args:
        text: Input text
        
    Returns:
        List of found time patterns
    """
    time_patterns = [
        r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?',  # HH:MM or HH:MM:SS with optional AM/PM
        r'\b(?:morning|afternoon|evening|night)\b',  # General time references
        r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',  # Simple hour with AM/PM
    ]
    
    found_times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_times.extend(matches)
    
    return found_times


def extract_numbers(text: str) -> List[str]:
    """
    Extract numeric values from text
    
    Args:
        text: Input text
        
    Returns:
        List of found numbers
    """
    # Pattern for various number formats
    number_patterns = [
        r'\b\d+\.?\d*\b',  # Basic numbers (integers and decimals)
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas
        r'\$\d+(?:\.\d{2})?\b',  # Currency
        r'\d+%\b',  # Percentages
    ]
    
    found_numbers = []
    for pattern in number_patterns:
        matches = re.findall(pattern, text)
        found_numbers.extend(matches)
    
    return found_numbers


def create_text_summary(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple summary of text
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Text summary
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return '. '.join(sentences) + '.'
    
    # Take first, middle, and last sentences for a basic summary
    if max_sentences == 3 and len(sentences) >= 3:
        selected = [sentences[0], sentences[len(sentences)//2], sentences[-1]]
    else:
        # Take first max_sentences
        selected = sentences[:max_sentences]
    
    return '. '.join(selected) + '.'


def preprocess_table_text(df: pd.DataFrame) -> str:
    """
    Preprocess DataFrame for text analysis
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed text representation
    """
    # Convert all columns to string
    df_str = df.astype(str)
    
    # Combine column names and values
    text_parts = []
    
    # Add column information
    text_parts.append(f"Table with columns: {', '.join(df.columns)}")
    
    # Add row data
    for _, row in df_str.iterrows():
        row_text = ' | '.join([f"{col}: {val}" for col, val in row.items()])
        text_parts.append(row_text)
    
    combined_text = '\n'.join(text_parts)
    return clean_text(combined_text)


def format_response_text(text: str, max_line_length: int = 80) -> str:
    """
    Format response text for better readability
    
    Args:
        text: Input text
        max_line_length: Maximum characters per line
        
    Returns:
        Formatted text
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    formatted_paragraphs = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_line_length:
            formatted_paragraphs.append(paragraph)
        else:
            # Wrap long paragraphs
            words = paragraph.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_line_length:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            formatted_paragraphs.append('\n'.join(lines))
    
    return '\n\n'.join(formatted_paragraphs)
