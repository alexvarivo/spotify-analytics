"""Utility functions for Spotify Analytics"""

import logging
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd

def setup_logging(level=logging.INFO, log_file='spotify_analytics.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_year(date_str: str) -> Optional[int]:
    """Extract year from date string"""
    if pd.isna(date_str):
        return None
    try:
        return int(date_str[:4])
    except:
        return None

def calculate_age_days(date_str: str) -> Optional[int]:
    """Calculate age in days from date string"""
    if pd.isna(date_str):
        return None
    try:
        release_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
        return (datetime.now() - release_date).days
    except:
        return None

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    if denominator == 0:
        return default
    return numerator / denominator

def flatten_genres(genre_lists: List[str]) -> List[str]:
    """Flatten and clean genre lists"""
    all_genres = []
    for genres in genre_lists:
        if genres and genres != 'Unknown':
            all_genres.extend(genres.split(', '))
    return all_genres

def calculate_diversity_index(items: List[str]) -> float:
    """Calculate Simpson's Diversity Index"""
    from collections import Counter
    
    counts = Counter(items)
    total = sum(counts.values())
    
    if total <= 1:
        return 0.0
    
    diversity = 1 - sum((count/total)**2 for count in counts.values())
    return diversity