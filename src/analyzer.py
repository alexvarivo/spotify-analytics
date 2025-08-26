"""
Spotify Music Analytics Platform - Core Analyzer Module
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter, defaultdict

# Data Processing
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Spotify API
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Internal imports
from config import SpotifyConfig, AnalyticsConfig
from utils import extract_year, calculate_age_days

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Core Spotify Analytics Engine
# ============================================================================

class SpotifyAnalyzer:
    """Main class for Spotify data analysis"""
    
    def __init__(self, config: SpotifyConfig = None):
        self.config = config or SpotifyConfig()
        self.config.validate()  # Validate credentials
        self.analytics_config = AnalyticsConfig()
        self.sp = None
        self.user_data = {}
        self.tracks_df = None
        self.artists_df = None
        self.recent_df = None
        
        # Create output directories
        self._setup_directories()
        
        # Initialize Spotify client
        self._init_spotify()
    
    def _setup_directories(self):
        """Create necessary directories for output"""
        for dir_path in [self.analytics_config.output_dir, 
                         self.analytics_config.data_dir,
                         self.analytics_config.plots_dir]:
            dir_path.mkdir(exist_ok=True)
            (self.analytics_config.output_dir / dir_path).mkdir(exist_ok=True)
    
    def _init_spotify(self):
        """Initialize Spotify API client with authentication"""
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                redirect_uri=self.config.redirect_uri,
                scope=self.config.scope,
                cache_path=self.config.cache_path
            ))
            
            # Test connection and get user info
            self.user_data = self.sp.current_user()
            logger.info(f"Connected as: {self.user_data['display_name']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise
    
    # ========================================================================
    # Data Collection Methods
    # ========================================================================
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all available Spotify data"""
        logger.info("Starting comprehensive data collection...")
        
        # Collect different data types
        tracks_data = self._collect_top_tracks()
        artists_data = self._collect_top_artists()
        recent_data = self._collect_recent_plays()
        playlists_data = self._collect_playlists()
        
        # Convert to DataFrames
        self.tracks_df = pd.DataFrame(tracks_data)
        self.artists_df = pd.DataFrame(artists_data)
        self.recent_df = pd.DataFrame(recent_data)
        self.playlists_df = pd.DataFrame(playlists_data)
        
        # Save raw data
        self._save_raw_data()
        
        logger.info("Data collection complete!")
        
        return {
            'tracks': self.tracks_df,
            'artists': self.artists_df,
            'recent': self.recent_df,
            'playlists': self.playlists_df
        }
    
    def _collect_top_tracks(self) -> List[Dict]:
        """Collect top tracks across different time ranges"""
        all_tracks = []
        time_ranges = {
            "short_term": "Last 4 weeks",
            "medium_term": "Last 6 months",
            "long_term": "All time"
        }
        
        for range_key, range_name in time_ranges.items():
            logger.info(f"Collecting top tracks for: {range_name}")
            
            try:
                results = self.sp.current_user_top_tracks(
                    limit=self.analytics_config.max_tracks_per_range,
                    time_range=range_key
                )
                
                for idx, track in enumerate(results['items']):
                    track_data = self._extract_track_features(track)
                    track_data['time_range'] = range_name
                    track_data['rank'] = idx + 1
                    track_data['range_key'] = range_key
                    all_tracks.append(track_data)
                    
            except Exception as e:
                logger.error(f"Error collecting tracks for {range_name}: {e}")
        
        return all_tracks
    
    def _extract_track_features(self, track: Dict) -> Dict:
        """Extract comprehensive features from a track object"""
        # Basic track info
        features = {
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_names': ', '.join([artist['name'] for artist in track['artists']]),
            'artist_ids': ', '.join([artist['id'] for artist in track['artists']]),
            'primary_artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
            'album_name': track['album']['name'],
            'album_id': track['album']['id'],
            'album_type': track['album']['album_type'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'duration_min': track['duration_ms'] / 60000,
            'popularity': track['popularity'],
            'explicit': track['explicit'],
            'preview_url': track['preview_url'],
            'track_number': track['track_number'],
            'disc_number': track['disc_number']
        }
        
        # Calculate additional metrics
        features['release_year'] = self._extract_year(features['release_date'])
        features['track_age_days'] = self._calculate_age_days(features['release_date'])
        
        # Get artist details if available
        if track['artists']:
            artist_id = track['artists'][0]['id']
            artist_info = self._get_artist_info(artist_id)
            features.update(artist_info)
        
        return features
    
    def _get_artist_info(self, artist_id: str) -> Dict:
        """Get additional artist information"""
        try:
            artist = self.sp.artist(artist_id)
            return {
                'artist_popularity': artist['popularity'],
                'artist_followers': artist['followers']['total'],
                'artist_genres': ', '.join(artist['genres'][:5]) if artist['genres'] else 'Unknown',
                'artist_genre_count': len(artist['genres'])
            }
        except:
            return {
                'artist_popularity': None,
                'artist_followers': None,
                'artist_genres': 'Unknown',
                'artist_genre_count': 0
            }
    
    def _collect_top_artists(self) -> List[Dict]:
        """Collect top artists across time ranges"""
        all_artists = []
        time_ranges = {
            "short_term": "Last 4 weeks",
            "medium_term": "Last 6 months",
            "long_term": "All time"
        }
        
        for range_key, range_name in time_ranges.items():
            logger.info(f"Collecting top artists for: {range_name}")
            
            try:
                results = self.sp.current_user_top_artists(
                    limit=self.analytics_config.max_artists_per_range,
                    time_range=range_key
                )
                
                for idx, artist in enumerate(results['items']):
                    artist_data = {
                        'artist_id': artist['id'],
                        'artist_name': artist['name'],
                        'genres': ', '.join(artist['genres'][:5]) if artist['genres'] else 'Unknown',
                        'genre_count': len(artist['genres']),
                        'popularity': artist['popularity'],
                        'followers': artist['followers']['total'],
                        'time_range': range_name,
                        'rank': idx + 1,
                        'range_key': range_key
                    }
                    all_artists.append(artist_data)
                    
            except Exception as e:
                logger.error(f"Error collecting artists for {range_name}: {e}")
        
        return all_artists
    
    def _collect_recent_plays(self) -> List[Dict]:
        """Collect recently played tracks with temporal analysis"""
        recent_tracks = []
        
        try:
            results = self.sp.current_user_recently_played(limit=50)
            
            for item in results['items']:
                track = item['track']
                played_at = datetime.fromisoformat(item['played_at'].replace('Z', '+00:00'))
                
                track_data = {
                    'track_name': track['name'],
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'played_at': played_at,
                    'played_date': played_at.date(),
                    'played_hour': played_at.hour,
                    'played_weekday': played_at.strftime('%A'),
                    'played_weekday_num': played_at.weekday(),
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                }
                recent_tracks.append(track_data)
                
        except Exception as e:
            logger.error(f"Error collecting recent plays: {e}")
        
        return recent_tracks
    
    def _collect_playlists(self) -> List[Dict]:
        """Collect user playlists data"""
        playlists_data = []
        
        try:
            playlists = self.sp.current_user_playlists(limit=50)
            
            for playlist in playlists['items']:
                playlist_data = {
                    'playlist_id': playlist['id'],
                    'playlist_name': playlist['name'],
                    'owner': playlist['owner']['display_name'],
                    'is_public': playlist['public'],
                    'is_collaborative': playlist['collaborative'],
                    'track_count': playlist['tracks']['total'],
                    'description': playlist.get('description', '')
                }
                playlists_data.append(playlist_data)
                
        except Exception as e:
            logger.error(f"Error collecting playlists: {e}")
        
        return playlists_data
    
    # ========================================================================
    # Advanced Analytics Methods
    # ========================================================================
    
    def perform_clustering_analysis(self) -> Dict:
        """Perform clustering analysis on music taste"""
        logger.info("Performing clustering analysis...")
        
        if self.tracks_df is None or self.tracks_df.empty:
            logger.warning("No tracks data available for clustering")
            return {}
        
        # Prepare features for clustering
        feature_cols = ['popularity', 'duration_min', 'artist_popularity', 
                       'track_age_days', 'explicit']
        
        # Filter and prepare data
        cluster_df = self.tracks_df[self.tracks_df['time_range'] == 'All time'].copy()
        cluster_df = cluster_df.dropna(subset=feature_cols)
        
        if len(cluster_df) < 10:
            logger.warning("Insufficient data for clustering")
            return {}
        
        # Convert boolean to numeric
        cluster_df['explicit'] = cluster_df['explicit'].astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_df[feature_cols])
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(3, len(feature_cols)))
        features_pca = pca.fit_transform(features_scaled)
        
        # Find optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(10, len(cluster_df) // 5))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            silhouette_avg = silhouette_score(features_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Use optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_df['cluster'] = kmeans.fit_predict(features_pca)
        
        # Analyze clusters
        cluster_summary = self._analyze_clusters(cluster_df, feature_cols)
        
        # Add PCA components for visualization
        cluster_df['pca_1'] = features_pca[:, 0]
        cluster_df['pca_2'] = features_pca[:, 1]
        if features_pca.shape[1] > 2:
            cluster_df['pca_3'] = features_pca[:, 2]
        
        return {
            'clustered_data': cluster_df,
            'cluster_summary': cluster_summary,
            'optimal_k': optimal_k,
            'pca_variance_ratio': pca.explained_variance_ratio_,
            'silhouette_scores': list(zip(K_range, silhouette_scores))
        }
    
    def _analyze_clusters(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Analyze characteristics of each cluster"""
        cluster_summary = df.groupby('cluster')[feature_cols].agg(['mean', 'std'])
        cluster_summary['size'] = df.groupby('cluster').size()
        
        # Add top artists per cluster
        top_artists = df.groupby('cluster')['primary_artist'].apply(
            lambda x: x.value_counts().head(3).index.tolist()
        )
        cluster_summary['top_artists'] = top_artists
        
        return cluster_summary
    
    def calculate_diversity_metrics(self) -> Dict:
        """Calculate music diversity metrics"""
        logger.info("Calculating diversity metrics...")
        
        metrics = {}
        
        if self.tracks_df is not None and not self.tracks_df.empty:
            # Artist diversity (Simpson's diversity index)
            artist_counts = self.tracks_df['primary_artist'].value_counts()
            total_tracks = len(self.tracks_df)
            simpson_index = 1 - sum((count/total_tracks)**2 for count in artist_counts)
            metrics['artist_diversity_simpson'] = simpson_index
            
            # Genre diversity
            all_genres = []
            for genres in self.tracks_df['artist_genres'].dropna():
                if genres != 'Unknown':
                    all_genres.extend(genres.split(', '))
            
            genre_counts = Counter(all_genres)
            metrics['unique_genres'] = len(genre_counts)
            metrics['top_genres'] = dict(genre_counts.most_common(10))
            
            # Temporal diversity (how spread out are release dates)
            if 'release_year' in self.tracks_df.columns:
                years = self.tracks_df['release_year'].dropna()
                metrics['release_year_std'] = years.std()
                metrics['release_year_range'] = years.max() - years.min()
                metrics['median_release_year'] = years.median()
            
            # Popularity variance
            metrics['popularity_std'] = self.tracks_df['popularity'].std()
            metrics['popularity_cv'] = self.tracks_df['popularity'].std() / self.tracks_df['popularity'].mean()
            
        return metrics
    
    def analyze_listening_patterns(self) -> Dict:
        """Analyze temporal listening patterns"""
        logger.info("Analyzing listening patterns...")
        
        patterns = {}
        
        if self.recent_df is not None and not self.recent_df.empty:
            # Hourly distribution
            hourly_counts = self.recent_df['played_hour'].value_counts().sort_index()
            patterns['hourly_distribution'] = hourly_counts.to_dict()
            
            # Peak listening hours
            peak_hours = hourly_counts.nlargest(3).index.tolist()
            patterns['peak_hours'] = peak_hours
            
            # Weekday distribution
            weekday_counts = self.recent_df['played_weekday_num'].value_counts().sort_index()
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            patterns['weekday_distribution'] = {weekday_names[i]: count 
                                               for i, count in weekday_counts.items()}
            
            # Calculate listening streaks
            dates = pd.to_datetime(self.recent_df['played_date']).unique()
            dates = np.sort(dates)
            
            if len(dates) > 1:
                date_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
                max_streak = self._find_longest_streak(date_diffs)
                patterns['longest_streak_days'] = max_streak
                patterns['average_tracks_per_day'] = len(self.recent_df) / len(dates)
            
        return patterns
    
    def _find_longest_streak(self, diffs: np.ndarray) -> int:
        """Find longest consecutive listening streak"""
        streak = 1
        max_streak = 1
        
        for diff in diffs:
            if diff == 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        
        return max_streak
    
    def calculate_evolution_metrics(self) -> Dict:
        """Calculate how music taste has evolved over time"""
        logger.info("Calculating evolution metrics...")
        
        evolution = {}
        
        if self.tracks_df is None or self.tracks_df.empty:
            return evolution
        
        time_ranges = ['Last 4 weeks', 'Last 6 months', 'All time']
        
        for i in range(len(time_ranges) - 1):
            recent = self.tracks_df[self.tracks_df['time_range'] == time_ranges[i]]
            older = self.tracks_df[self.tracks_df['time_range'] == time_ranges[i + 1]]
            
            if recent.empty or older.empty:
                continue
            
            # Artist overlap
            recent_artists = set(recent['primary_artist'].unique())
            older_artists = set(older['primary_artist'].unique())
            
            if older_artists:
                artist_retention = len(recent_artists & older_artists) / len(older_artists)
                new_artist_ratio = len(recent_artists - older_artists) / len(recent_artists) if recent_artists else 0
                
                evolution[f'artist_retention_{time_ranges[i]}'] = artist_retention
                evolution[f'new_artist_ratio_{time_ranges[i]}'] = new_artist_ratio
            
            # Popularity shift
            popularity_change = recent['popularity'].mean() - older['popularity'].mean()
            evolution[f'popularity_shift_{time_ranges[i]}'] = popularity_change
            
            # Genre evolution
            recent_genres = Counter()
            older_genres = Counter()
            
            for genres in recent['artist_genres'].dropna():
                if genres != 'Unknown':
                    recent_genres.update(genres.split(', '))
            
            for genres in older['artist_genres'].dropna():
                if genres != 'Unknown':
                    older_genres.update(genres.split(', '))
            
            if older_genres:
                # Calculate Jaccard similarity for genres
                all_genres = set(recent_genres.keys()) | set(older_genres.keys())
                genre_similarity = len(set(recent_genres.keys()) & set(older_genres.keys())) / len(all_genres)
                evolution[f'genre_similarity_{time_ranges[i]}'] = genre_similarity
        
        return evolution
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def create_comprehensive_visualizations(self):
        """Create all visualizations"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create different types of visualizations
        self._create_overview_dashboard()
        self._create_temporal_analysis()
        self._create_artist_network()
        self._create_clustering_visualization()
        self._create_evolution_charts()
        
        logger.info("Visualizations complete!")
    
    def _create_overview_dashboard(self):
        """Create main dashboard with key metrics"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Top Artists', 'Genre Distribution', 
                        'Popularity Distribution', 'Listening Hours',
                        'Track Release Years', 'Key Metrics'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'histogram'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # Top Artists
        if self.artists_df is not None and not self.artists_df.empty:
            top_artists = self.artists_df[self.artists_df['time_range'] == 'All time'].head(10)
            fig.add_trace(
                go.Bar(
                    x=top_artists['artist_name'], 
                    y=top_artists['popularity'],
                    marker_color='lightblue',
                    name='Artist Popularity'
                ),
                row=1, col=1
            )
        
        # Genre Distribution
        if self.tracks_df is not None and not self.tracks_df.empty:
            all_genres = []
            for genres in self.tracks_df['artist_genres'].dropna():
                if genres != 'Unknown':
                    all_genres.extend(genres.split(', '))
            
            genre_counts = Counter(all_genres)
            top_genres = dict(genre_counts.most_common(8))
            
            if top_genres:
                fig.add_trace(
                    go.Pie(
                        labels=list(top_genres.keys()), 
                        values=list(top_genres.values()),
                        name='Genres'
                    ),
                    row=1, col=2
                )
        
        # Popularity Distribution
        if self.tracks_df is not None and not self.tracks_df.empty:
            fig.add_trace(
                go.Histogram(
                    x=self.tracks_df['popularity'],
                    nbinsx=20,
                    marker_color='coral',
                    name='Popularity'
                ),
                row=2, col=1
            )
        
        # Listening Hours
        if self.recent_df is not None and not self.recent_df.empty:
            hourly_counts = self.recent_df['played_hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    marker_color='lightgreen',
                    name='Tracks by Hour'
                ),
                row=2, col=2
            )
        
        # Track Release Years
        if self.tracks_df is not None and 'release_year' in self.tracks_df.columns:
            years = self.tracks_df['release_year'].dropna()
            fig.add_trace(
                go.Histogram(
                    x=years,
                    nbinsx=20,
                    marker_color='purple',
                    name='Release Years'
                ),
                row=3, col=1
            )
        
        # Key Metrics (as scatter points for visualization)
        if self.tracks_df is not None and not self.tracks_df.empty:
            # Calculate diversity metrics
            diversity = self.calculate_diversity_metrics()
            
            # Create a simple metrics visualization
            metrics_names = ['Diversity Score', 'Avg Popularity', 'Unique Artists', 'Unique Genres']
            metrics_values = [
                diversity.get('artist_diversity_simpson', 0) * 100,
                self.tracks_df['popularity'].mean(),
                len(self.tracks_df['primary_artist'].unique()),
                diversity.get('unique_genres', 0)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_names,
                    y=metrics_values,
                    mode='markers+text',
                    marker=dict(size=20, color=metrics_values, colorscale='Viridis'),
                    text=[f'{v:.1f}' if v < 100 else f'{int(v)}' for v in metrics_values],
                    textposition='top center',
                    name='Metrics'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000, 
            showlegend=False, 
            title_text="Spotify Analytics Dashboard",
            title_font_size=24
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Artist", row=1, col=1)
        fig.update_yaxes(title_text="Popularity", row=1, col=1)
        
        fig.update_xaxes(title_text="Popularity Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
        fig.update_yaxes(title_text="Tracks Played", row=2, col=2)
        
        fig.update_xaxes(title_text="Release Year", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        
        fig.update_xaxes(title_text="Metric", row=3, col=2)
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        # Save dashboard
        output_path = self.analytics_config.output_dir / self.analytics_config.plots_dir / 'dashboard.html'
        fig.write_html(str(output_path))
        logger.info(f"Dashboard saved to {output_path}")
    
    def _create_temporal_analysis(self):
        """Create temporal analysis visualizations"""
        if self.recent_df is None or self.recent_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hourly distribution
        hourly_counts = self.recent_df['played_hour'].value_counts().sort_index()
        axes[0, 0].bar(hourly_counts.index, hourly_counts.values, color='skyblue')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Tracks')
        axes[0, 0].set_title('Listening Activity by Hour')
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Weekday distribution
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = self.recent_df['played_weekday'].value_counts()
        weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)
        
        axes[0, 1].bar(range(7), weekday_counts.values, color='lightcoral')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels([day[:3] for day in weekday_order], rotation=45)
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Tracks')
        axes[0, 1].set_title('Listening Activity by Weekday')
        
        # Heatmap of hour vs weekday
        pivot_table = self.recent_df.pivot_table(
            values='track_name', 
            index='played_hour', 
            columns='played_weekday_num',
            aggfunc='count',
            fill_value=0
        )
        
        # All 7 days represented
        for i in range(7):
            if i not in pivot_table.columns:
                pivot_table[i] = 0
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)
        
        sns.heatmap(pivot_table, cmap='YlOrRd', cbar_kws={'label': 'Tracks Played'},
                ax=axes[1, 0])
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Hour of Day')
        axes[1, 0].set_title('Listening Patterns Heatmap')
        # Fix the x-axis labels
        axes[1, 0].set_xticklabels([day[:3] for day in weekday_order], rotation=0)
        
        # Daily trend
        daily_counts = self.recent_df.groupby('played_date').size()
        axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Tracks Played')
        axes[1, 1].set_title('Daily Listening Trend')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.analytics_config.output_dir / self.analytics_config.plots_dir / 'temporal_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Temporal analysis saved to {output_path}")
    
    def _create_artist_network(self):
        """Create artist collaboration network"""
        # This would require additional logic to identify collaborations
        # For now, we'll create a co-occurrence matrix based on listening sessions
        pass
    
    def _create_clustering_visualization(self):
        """Visualize clustering results"""
        clustering_results = self.perform_clustering_analysis()
        
        if not clustering_results:
            return
        
        cluster_df = clustering_results['clustered_data']
        
        # Create 2D scatter plot of clusters
        fig = px.scatter(
            cluster_df,
            x='pca_1',
            y='pca_2',
            color='cluster',
            hover_data=['track_name', 'primary_artist', 'popularity'],
            title='Music Taste Clusters (PCA Projection)',
            labels={'pca_1': 'First Principal Component', 'pca_2': 'Second Principal Component'}
        )
        
        output_path = self.analytics_config.output_dir / self.analytics_config.plots_dir / 'clusters.html'
        fig.write_html(str(output_path))
        logger.info(f"Clustering visualization saved to {output_path}")
    
    def _create_evolution_charts(self):
        """Create charts showing taste evolution"""
        if self.tracks_df is None or self.tracks_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Popularity over time ranges
        time_order = ['Last 4 weeks', 'Last 6 months', 'All time']
        avg_popularity = self.tracks_df.groupby('time_range')['popularity'].mean()
        avg_popularity = avg_popularity.reindex(time_order)
        
        axes[0, 0].plot(time_order, avg_popularity.values, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Time Range')
        axes[0, 0].set_ylabel('Average Popularity')
        axes[0, 0].set_title('Popularity Trend Over Time Ranges')
        axes[0, 0].grid(True, alpha=0.3)
        
        # New vs recurring artists
        evolution_metrics = self.calculate_evolution_metrics()
        
        if evolution_metrics:
            metrics_to_plot = [m for m in evolution_metrics.keys() if 'new_artist_ratio' in m]
            if metrics_to_plot:
                values = [evolution_metrics[m] * 100 for m in metrics_to_plot]
                labels = [m.replace('new_artist_ratio_', '') for m in metrics_to_plot]
                
                axes[0, 1].bar(labels, values, color='teal')
                axes[0, 1].set_xlabel('Time Period')
                axes[0, 1].set_ylabel('New Artist Discovery Rate (%)')
                axes[0, 1].set_title('Music Discovery Patterns')
        
        # Release year distribution by time range
        for time_range in time_order:
            subset = self.tracks_df[self.tracks_df['time_range'] == time_range]
            if 'release_year' in subset.columns:
                subset['release_year'].hist(bins=20, alpha=0.5, label=time_range, ax=axes[1, 0])
        
        axes[1, 0].set_xlabel('Release Year')
        axes[1, 0].set_ylabel('Number of Tracks')
        axes[1, 0].set_title('Release Year Distribution by Time Range')
        axes[1, 0].legend()
        
        # Genre evolution
        genre_data = defaultdict(lambda: defaultdict(int))
        
        for time_range in time_order:
            subset = self.tracks_df[self.tracks_df['time_range'] == time_range]
            for genres in subset['artist_genres'].dropna():
                if genres != 'Unknown':
                    for genre in genres.split(', ')[:1]:  # Take primary genre
                        genre_data[genre][time_range] += 1
        
        # Plot top evolving genres
        top_genres = sorted(genre_data.keys(), 
                          key=lambda x: sum(genre_data[x].values()), 
                          reverse=True)[:5]
        
        x_pos = np.arange(len(time_order))
        width = 0.15
        
        for i, genre in enumerate(top_genres):
            values = [genre_data[genre].get(tr, 0) for tr in time_order]
            axes[1, 1].bar(x_pos + i * width, values, width, label=genre)
        
        axes[1, 1].set_xlabel('Time Range')
        axes[1, 1].set_ylabel('Track Count')
        axes[1, 1].set_title('Genre Evolution Over Time')
        axes[1, 1].set_xticks(x_pos + width * 2)
        axes[1, 1].set_xticklabels(time_order)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        output_path = self.analytics_config.output_dir / self.analytics_config.plots_dir / 'evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Evolution charts saved to {output_path}")
    
    # ========================================================================
    # Reporting Methods
    # ========================================================================
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive markdown report"""
        logger.info("Generating comprehensive report...")
        
        report = []
        report.append("# Spotify Music Analytics Report")
        report.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**User:** {self.user_data.get('display_name', 'Unknown')}")
        report.append("\n---\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        
        if self.tracks_df is not None and not self.tracks_df.empty:
            total_tracks = len(self.tracks_df['track_id'].unique())
            total_artists = len(self.tracks_df['primary_artist'].unique())
            avg_popularity = self.tracks_df['popularity'].mean()
            
            report.append(f"- **Total unique tracks analyzed:** {total_tracks}")
            report.append(f"- **Total unique artists:** {total_artists}")
            report.append(f"- **Average track popularity:** {avg_popularity:.1f}/100")
        
        # Diversity Metrics
        diversity = self.calculate_diversity_metrics()
        if diversity:
            report.append("\n## Music Diversity Analysis\n")
            report.append(f"- **Artist diversity (Simpson's Index):** {diversity.get('artist_diversity_simpson', 0):.3f}")
            report.append(f"- **Number of unique genres:** {diversity.get('unique_genres', 0)}")
            report.append(f"- **Popularity variance coefficient:** {diversity.get('popularity_cv', 0):.3f}")
            
            if 'release_year_range' in diversity:
                report.append(f"- **Release year range:** {diversity['release_year_range']:.0f} years")
        
        # Listening Patterns
        patterns = self.analyze_listening_patterns()
        if patterns:
            report.append("\n## Listening Patterns\n")
            
            if 'peak_hours' in patterns:
                peak_hours_str = ', '.join([f"{h}:00" for h in patterns['peak_hours']])
                report.append(f"- **Peak listening hours:** {peak_hours_str}")
            
            if 'longest_streak_days' in patterns:
                report.append(f"- **Longest listening streak:** {patterns['longest_streak_days']} days")
            
            if 'average_tracks_per_day' in patterns:
                report.append(f"- **Average tracks per day:** {patterns['average_tracks_per_day']:.1f}")
        
        # Evolution Metrics
        evolution = self.calculate_evolution_metrics()
        if evolution:
            report.append("\n## Taste Evolution\n")
            
            for key, value in evolution.items():
                if 'artist_retention' in key:
                    period = key.replace('artist_retention_', '')
                    report.append(f"- **Artist retention ({period}):** {value*100:.1f}%")
                elif 'new_artist_ratio' in key:
                    period = key.replace('new_artist_ratio_', '')
                    report.append(f"- **New artist discovery rate ({period}):** {value*100:.1f}%")
        
        # Top Content
        report.append("\n## Top Content\n")
        
        if self.tracks_df is not None and not self.tracks_df.empty:
            report.append("\n### Top 10 Tracks (All Time)\n")
            top_tracks = self.tracks_df[self.tracks_df['time_range'] == 'All time'].head(10)
            
            for idx, track in top_tracks.iterrows():
                report.append(f"{track['rank']}. **{track['track_name']}** by {track['primary_artist']}")
        
        if self.artists_df is not None and not self.artists_df.empty:
            report.append("\n### Top 10 Artists (All Time)\n")
            top_artists = self.artists_df[self.artists_df['time_range'] == 'All time'].head(10)
            
            for idx, artist in top_artists.iterrows():
                genres = artist['genres'].split(', ')[:2]
                genres_str = ', '.join(genres) if genres[0] != 'Unknown' else 'Unknown'
                report.append(f"{artist['rank']}. **{artist['artist_name']}** ({genres_str})")
        
        # Clustering Results
        clustering = self.perform_clustering_analysis()
        if clustering and 'optimal_k' in clustering:
            report.append("\n## Music Taste Clusters\n")
            report.append(f"- **Optimal number of clusters found:** {clustering['optimal_k']}")
            report.append("- Clusters represent different 'moods' or 'styles' in your music taste")
            
            if 'pca_variance_ratio' in clustering:
                var_explained = sum(clustering['pca_variance_ratio']) * 100
                report.append(f"- **Variance explained by principal components:** {var_explained:.1f}%")
        
        # Save report
        report_text = '\n'.join(report)
        output_path = self.analytics_config.output_dir / 'spotify_analytics_report.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def _save_raw_data(self):
        """Save all raw data to CSV files"""
        output_dir = self.analytics_config.output_dir / self.analytics_config.data_dir
        
        data_frames = {
            'tracks': self.tracks_df,
            'artists': self.artists_df,
            'recent': self.recent_df,
            'playlists': self.playlists_df
        }
        
        for name, df in data_frames.items():
            if df is not None and not df.empty:
                output_path = output_dir / f'{name}_data.csv'
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {name} data to {output_path}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string"""
        if pd.isna(date_str):
            return None
        try:
            return int(date_str[:4])
        except:
            return None
    
    def _calculate_age_days(self, date_str: str) -> Optional[int]:
        """Calculate age of track in days"""
        if pd.isna(date_str):
            return None
        try:
            release_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            return (datetime.now() - release_date).days
        except:
            return None