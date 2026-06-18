"""
Data Loader for Bengali Author Stories
Handles data loading, preprocessing, and organization
"""

import os
from pathlib import Path
import numpy as np
from feature_extractor import BengaliFeatureExtractor

class DataLoader:
    """Load and organize author story data"""
    
    def __init__(self, data_dir='.'):
        """
        Args:
            data_dir: Root directory containing Female Stories and Male Stories folders
        """
        self.data_dir = data_dir
        self.feature_extractor = BengaliFeatureExtractor()
        self.data = []
        self.authors = []
        self.genders = []
        self.filenames = []
    
    def load_data(self):
        """
        Load all story files from Female Stories and Male Stories directories
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Gender labels (0=Female, 1=Male)
            metadata: Dict with author and filename info
        """
        X = []
        y = []
        metadata = {
            'authors': [],
            'filenames': [],
            'genders': [],
            'text_lengths': []
        }
        
        # Load female stories
        female_dir = os.path.join(self.data_dir, 'Female Stories')
        if os.path.exists(female_dir):
            print(f"Loading from {female_dir}...")
            X_female, metadata_female = self._load_from_directory(female_dir, gender='F')
            X.extend(X_female)
            metadata['authors'].extend(metadata_female['authors'])
            metadata['filenames'].extend(metadata_female['filenames'])
            metadata['genders'].extend([0] * len(X_female))  # 0 for Female
            metadata['text_lengths'].extend(metadata_female['text_lengths'])
            y.extend([0] * len(X_female))
        
        # Load male stories
        male_dir = os.path.join(self.data_dir, 'Male Stories')
        if os.path.exists(male_dir):
            print(f"Loading from {male_dir}...")
            X_male, metadata_male = self._load_from_directory(male_dir, gender='M')
            X.extend(X_male)
            metadata['authors'].extend(metadata_male['authors'])
            metadata['filenames'].extend(metadata_male['filenames'])
            metadata['genders'].extend([1] * len(X_male))  # 1 for Male
            metadata['text_lengths'].extend(metadata_male['text_lengths'])
            y.extend([1] * len(X_male))
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nData loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Female stories: {sum(1 for g in y if g == 0)}")
        print(f"Male stories: {sum(1 for g in y if g == 1)}")
        print(f"Features per sample: {X.shape[1] if X.ndim > 1 else 1}")
        
        return X, y, metadata
    
    def _load_from_directory(self, directory, gender='F'):
        """
        Load all text files from a directory
        
        Args:
            directory: Path to directory containing author subdirectories
            gender: 'F' or 'M'
        
        Returns:
            X: Feature vectors
            metadata: File and author information
        """
        X = []
        metadata = {
            'authors': [],
            'filenames': [],
            'text_lengths': []
        }
        
        # Check if directory contains author subdirectories or files directly
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        if subdirs:
            # Author subdirectories
            for author_name in sorted(subdirs):
                author_path = os.path.join(directory, author_name)
                
                # Load all text files from author directory
                for filename in sorted(os.listdir(author_path)):
                    if filename.endswith(('.txt', '.md')):
                        filepath = os.path.join(author_path, filename)
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read()
                            
                            # Extract features
                            features, feature_names = self.feature_extractor.extract_features_vector(text)
                            X.append(features)
                            
                            metadata['authors'].append(author_name)
                            metadata['filenames'].append(filename)
                            metadata['text_lengths'].append(len(text))
                            
                            print(f"  Loaded: {author_name}/{filename} ({len(text)} chars)")
                        
                        except Exception as e:
                            print(f"  Error loading {filename}: {e}")
        else:
            # Files directly in directory
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(('.txt', '.md')):
                    filepath = os.path.join(directory, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Extract features
                        features, feature_names = self.feature_extractor.extract_features_vector(text)
                        X.append(features)
                        
                        # Extract author name from filename (e.g., "AuthorName_Story1.txt")
                        author_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                        
                        metadata['authors'].append(author_name)
                        metadata['filenames'].append(filename)
                        metadata['text_lengths'].append(len(text))
                        
                        print(f"  Loaded: {filename} ({len(text)} chars)")
                    
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
        
        return X, metadata


def load_single_text(filepath):
    """
    Load and extract features from a single text file
    
    Args:
        filepath: Path to text file
    
    Returns:
        features: Feature vector
        feature_names: Names of features
    """
    feature_extractor = BengaliFeatureExtractor()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        features, feature_names = feature_extractor.extract_features_vector(text)
        return features, feature_names
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None
