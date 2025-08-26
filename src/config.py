"""Configuration management for Spotify Analytics"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SpotifyConfig:
    """Spotify API configuration"""
    client_id: str = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    scope: str = (
        "user-top-read user-library-read user-read-recently-played "
        "playlist-read-private user-follow-read"
    )
    cache_path: str = ".spotify_cache"
    
    def validate(self):
        """Validate configuration"""
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing Spotify credentials. Please set SPOTIFY_CLIENT_ID "
                "and SPOTIFY_CLIENT_SECRET in your .env file"
            )

@dataclass
class AnalyticsConfig:
    """Analytics configuration"""
    output_dir: Path = Path("spotify_analytics_output")
    data_dir: Path = Path("data")
    plots_dir: Path = Path("visualizations")
    max_tracks_per_range: int = 50
    max_artists_per_range: int = 50
    clustering_components: int = 5
    max_clusters: int = 10