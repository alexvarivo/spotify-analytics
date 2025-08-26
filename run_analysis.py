#!/usr/bin/env python
"""
Main entry point for Spotify Analytics Platform
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analyzer import SpotifyAnalyzer, SpotifyConfig
from config import SpotifyConfig  # Import from config.py
import logging

def main():
    parser = argparse.ArgumentParser(description='Spotify Music Analytics Platform')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', type=str, default='spotify_analytics_output', 
                       help='Output directory')
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print(" SPOTIFY ANALYTICS PLATFORM ")
    print("="*60 + "\n")
    
    try:
        # Initialize and run analyzer
        analyzer = SpotifyAnalyzer()
        
        # Collect data
        print("Collecting Spotify data...")
        data = analyzer.collect_all_data()
        
        # Perform analytics
        print("Running analytics...")
        analyzer.calculate_diversity_metrics()
        analyzer.analyze_listening_patterns()
        analyzer.calculate_evolution_metrics()
        analyzer.perform_clustering_analysis()
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.create_comprehensive_visualizations()
        
        # Generate report
        print("Generating report...")
        analyzer.generate_comprehensive_report()
        
        print(f"\nAnalysis complete! Check {args.output}/ for results")
        
    except Exception as e:
        print(f"\n Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()