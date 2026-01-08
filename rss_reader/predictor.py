#!/usr/bin/env python3
"""
HN Success Predictor Classes
Shared module for training and inference.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict
from urllib.parse import urlparse
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression


class FeatureExtractor:
    """Extract features from titles and URLs."""
    
    # Common HN prefixes
    PREFIXES = ['show hn', 'ask hn', 'tell hn', 'launch hn']
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )
        self.domain_encoder = None
        self.top_domains = None
        
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        if not url:
            return "self_post"
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain if domain else "self_post"
        except Exception:
            return "self_post"
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title."""
        if not title:
            return ""
        title = title.lower()
        for prefix in self.PREFIXES:
            if title.startswith(prefix + ':'):
                title = title[len(prefix)+1:].strip()
            elif title.startswith(prefix):
                title = title[len(prefix):].strip()
        return title
    
    def _extract_meta_features(self, row: pd.Series) -> Dict:
        """Extract handcrafted meta-features."""
        title = str(row.get('title', '')).lower()
        url = str(row.get('url', '') or row.get('link', '') or '')
        domain = self._extract_domain(url)
        
        features = {
            'is_show_hn': 1 if title.startswith('show hn') else 0,
            'is_ask_hn': 1 if title.startswith('ask hn') else 0,
            'is_tell_hn': 1 if title.startswith('tell hn') else 0,
            'is_self_post': 1 if domain == 'self_post' else 0,
            'title_length': len(title),
            'word_count': len(title.split()),
            'has_year': 1 if re.search(r'\b(19|20)\d{2}\b', title) else 0,
            'has_number': 1 if re.search(r'\b\d+\b', title) else 0,
            'has_question': 1 if '?' in title else 0,
            'has_exclamation': 1 if '!' in title else 0,
            'has_colon': 1 if ':' in title else 0,
            'has_dash': 1 if '–' in title or '-' in title else 0,
            'has_pdf': 1 if '[pdf]' in title.lower() else 0,
            'has_video': 1 if '[video]' in title.lower() or 'video' in title.lower() else 0,
            'is_github': 1 if 'github' in domain else 0,
            'is_arxiv': 1 if 'arxiv' in domain else 0,
            'is_medium': 1 if 'medium' in domain else 0,
            'is_twitter': 1 if 'twitter' in domain or 'x.com' in domain else 0,
            'is_youtube': 1 if 'youtube' in domain else 0,
            'is_substack': 1 if 'substack' in domain else 0,
            'is_major_news': 1 if any(d in domain for d in ['nytimes', 'bbc', 'theguardian', 'reuters', 'wsj']) else 0,
            'is_tech_blog': 1 if any(d in domain for d in ['arstechnica', 'theverge', 'techcrunch', 'wired']) else 0,
        }
        
        return features
    
    def fit(self, df: pd.DataFrame):
        """Fit the feature extractors."""
        titles_clean = df['title'].apply(self._clean_title)
        self.tfidf.fit(titles_clean)
        
        url_col = 'url' if 'url' in df.columns else 'link'
        domains = df[url_col].fillna('').apply(self._extract_domain)
        domain_counts = Counter(domains)
        self.top_domains = [d for d, c in domain_counts.most_common(200)]
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data into feature matrix."""
        titles_clean = df['title'].apply(self._clean_title)
        tfidf_features = self.tfidf.transform(titles_clean).toarray()
        
        meta_features = df.apply(self._extract_meta_features, axis=1)
        meta_df = pd.DataFrame(list(meta_features))
        
        url_col = 'url' if 'url' in df.columns else 'link'
        domains = df[url_col].fillna('').apply(self._extract_domain)
        domain_features = np.zeros((len(df), len(self.top_domains)))
        for i, domain in enumerate(domains):
            if domain in self.top_domains:
                domain_features[i, self.top_domains.index(domain)] = 1
        
        all_features = np.hstack([
            tfidf_features,
            meta_df.values,
            domain_features
        ])
        
        return all_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform."""
        self.fit(df)
        return self.transform(df)


class SuccessPredictor:
    """Combined classifier and regressor for predicting post success."""
    
    def __init__(self, hit_threshold: int = 100):
        self.hit_threshold = hit_threshold
        self.feature_extractor = FeatureExtractor()
        self.classifier = LogisticRegression(
            C=1.0, 
            max_iter=2000,
            class_weight='balanced'
        )
        self.regressor = Ridge(alpha=10.0)
        
    def fit(self, df: pd.DataFrame):
        """Fit both models."""
        X = self.feature_extractor.fit_transform(df)
        y_class = (df['points'] >= self.hit_threshold).astype(int)
        y_reg = np.log1p(df['points'])
        
        print(f"Features shape: {X.shape}")
        print(f"Hits (>={self.hit_threshold} pts): {y_class.sum()} ({y_class.mean()*100:.1f}%)")
        
        print("\nTraining classifier...")
        self.classifier.fit(X, y_class)
        
        print("Training regressor...")
        self.regressor.fit(X, y_reg)
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of being a hit."""
        X = self.feature_extractor.transform(df)
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict_points(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected points."""
        X = self.feature_extractor.transform(df)
        log_points = self.regressor.predict(X)
        return np.expm1(log_points)
    
    def predict(self, titles: List[str], urls: List[str] = None) -> pd.DataFrame:
        """Predict for new titles."""
        if urls is None:
            urls = [''] * len(titles)
        
        df = pd.DataFrame({
            'title': titles,
            'url': urls,
            'link': urls  # Support both column names
        })
        
        hit_prob = self.predict_proba(df)
        expected_points = self.predict_points(df)
        
        return pd.DataFrame({
            'title': titles,
            'hit_probability': hit_prob,
            'expected_points': expected_points,
            'recommendation': pd.cut(
                hit_prob, 
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['Skip', 'Maybe', 'Good', 'Excellent']
            )
        })
