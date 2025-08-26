"""Tests for Spotify Analytics Platform"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.analyzer import SpotifyAnalyzer
from src.utils import extract_year, calculate_diversity_index

class TestSpotifyAnalyzer:
    """Test cases for SpotifyAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mocked Spotify client"""
        with patch('src.analyzer.spotipy.Spotify'):
            analyzer = SpotifyAnalyzer()
            analyzer.sp = Mock()
            return analyzer
    
    def test_extract_year(self):
        """Test year extraction from date strings"""
        assert extract_year("2024-01-15") == 2024
        assert extract_year("2024") == 2024
        assert extract_year(None) is None
    
    def test_diversity_index(self):
        """Test diversity index calculation"""
        items = ['rock', 'rock', 'pop', 'jazz']
        diversity = calculate_diversity_index(items)
        assert 0.5 < diversity < 0.7
    
    def test_data_collection(self, analyzer):
        """Test data collection process"""
        # Mock Spotify API responses
        analyzer.sp.current_user.return_value = {'display_name': 'Test User'}
        analyzer.sp.current_user_top_tracks.return_value = {
            'items': [{'id': '1', 'name': 'Test Track'}]
        }
        
        # Test collection
        data = analyzer.collect_all_data()
        assert 'tracks' in data

if __name__ == "__main__":
    pytest.main([__file__])