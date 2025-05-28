"""
Content utilities for FAISS-based content system
Provides helper functions for content similarity and recommendations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def search_similar_content(query_vector, faiss_index, metadata, top_k=10):
    """Search for similar content using FAISS index"""
    try:
        distances, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata["articles"]):
                article = metadata["articles"][idx]
                results.append({
                    "rank": i + 1,
                    "similarity": float(dist),
                    "article_id": article.get("newsID", f"article_{idx}"),
                    "title": article.get("title", "Unknown"),
                    "engagement_probability": article.get("engagement_probability", 0),
                    "editorial_recommendation": article.get("editorial_recommendation", "Unknown")
                })
        
        return results
    except Exception as e:
        print(f"Error in content search: {e}")
        return []

def get_top_engaging_content(metadata, threshold=0.6, limit=50):
    """Get top engaging content based on class-weighted predictions"""
    try:
        articles = metadata.get("articles", [])
        
        # Filter by engagement threshold
        engaging_articles = [
            article for article in articles 
            if article.get("engagement_probability", 0) >= threshold
        ]
        
        # Sort by engagement probability
        engaging_articles.sort(
            key=lambda x: x.get("engagement_probability", 0), 
            reverse=True
        )
        
        return engaging_articles[:limit]
    except Exception as e:
        print(f"Error getting engaging content: {e}")
        return []

def filter_by_editorial_recommendation(metadata, recommendation="Prioritize"):
    """Filter articles by editorial recommendation"""
    try:
        articles = metadata.get("articles", [])
        
        filtered = [
            article for article in articles
            if article.get("editorial_recommendation") == recommendation
        ]
        
        return filtered
    except Exception as e:
        print(f"Error filtering by recommendation: {e}")
        return []
