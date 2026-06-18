"""
Feature Extractor for Bengali Author Gender Classification
No pretrained models or embeddings - hand-crafted features
"""

import numpy as np
from collections import Counter
from bengali_text_normalizer import BengaliNormalizer, TextPreprocessor

class BengaliFeatureExtractor:
    """Extract linguistic features from Bengali text"""
    
    def __init__(self):
        self.normalizer = BengaliNormalizer()
        self.preprocessor = TextPreprocessor()
        
        # Common Bengali vocabulary for feature extraction
        self.common_verbs = {
            'বল', 'ভাব', 'কর', 'দেখ', 'আছ', 'এস', 'যা', 'দ', 'থাক', 'নি',
            'হ', 'রাখ', 'চল', 'লাগ', 'নিয়', 'তুল', 'বসা', 'ঠাই', 'লি', 'সাহায়
        }
        
        self.common_pronouns = {
            'আমি', 'তুমি', 'সে', 'তিনি', 'আমরা', 'তারা', 'আপনি', 'যিনি', 'কে', 'যা'
        }
        
        self.common_adjectives = {
            'বড়', 'ছোট', 'ভাল', 'মন্দ', 'সুন্দর', 'কদর্থ', 'নতুন', 'পুরাতন', 'প্রথম', 'শেষ'
        }
    
    def extract_features(self, text):
        """Extract all linguistic features"""
        tokens = self.normalizer.tokenize(text)
        sentences = self.preprocessor.get_sentence_tokens(text)
        pos_tags = self.normalizer.get_tokens_with_pos(text)
        
        features = {}
        
        # 1. Lexical features
        features.update(self._extract_lexical_features(tokens))
        
        # 2. Syntactic features
        features.update(self._extract_syntactic_features(tokens, sentences))
        
        # 3. POS usage features
        features.update(self._extract_pos_features(pos_tags))
        
        # 4. Character-level features
        features.update(self._extract_character_features(text))
        
        # 5. Morphological features
        features.update(self._extract_morphological_features(tokens))
        
        return features
    
    def _extract_lexical_features(self, tokens):
        """Extract vocabulary-based features"""
        features = {}
        
        if not tokens:
            return features
        
        # Vocabulary size
        features['vocab_size'] = len(set(tokens))
        features['total_tokens'] = len(tokens)
        
        # Type-token ratio (vocabulary richness)
        features['type_token_ratio'] = len(set(tokens)) / len(tokens) if tokens else 0
        
        # Unique word usage ratio
        word_freq = Counter(tokens)
        hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
        features['hapax_ratio'] = hapax_legomena / len(set(tokens)) if set(tokens) else 0
        
        # Average word length
        features['avg_word_length'] = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Pronoun frequency
        pronoun_count = sum(1 for t in tokens if t in self.common_pronouns)
        features['pronoun_freq'] = pronoun_count / len(tokens) if tokens else 0
        
        # Verb frequency
        verb_count = sum(1 for t in tokens if t in self.common_verbs)
        features['verb_freq'] = verb_count / len(tokens) if tokens else 0
        
        # Adjective frequency
        adj_count = sum(1 for t in tokens if t in self.common_adjectives)
        features['adjective_freq'] = adj_count / len(tokens) if tokens else 0
        
        # Word frequency distribution
        features['avg_word_freq'] = np.mean(list(word_freq.values())) if word_freq else 0
        
        return features
    
    def _extract_syntactic_features(self, tokens, sentences):
        """Extract sentence structure features"""
        features = {}
        
        if not sentences:
            return features
        
        # Sentence length statistics
        sent_lengths = [len(s) for s in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0
        features['max_sentence_length'] = np.max(sent_lengths) if sent_lengths else 0
        features['min_sentence_length'] = np.min(sent_lengths) if sent_lengths else 0
        features['sentence_length_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # Number of sentences
        features['num_sentences'] = len(sentences)
        
        # Sentence complexity (simple heuristic)
        complex_sentences = sum(1 for s in sent_lengths if s > 20)
        features['complex_sentence_ratio'] = complex_sentences / len(sentences) if sentences else 0
        
        return features
    
    def _extract_pos_features(self, pos_tags):
        """Extract Part-of-Speech usage features"""
        features = {}
        
        if not pos_tags:
            return features
        
        pos_counts = Counter([tag for _, tag in pos_tags])
        total_tags = len(pos_tags)
        
        # POS frequencies
        features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tags if total_tags else 0
        features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tags if total_tags else 0
        features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tags if total_tags else 0
        features['pron_ratio'] = pos_counts.get('PRON', 0) / total_tags if total_tags else 0
        
        return features
    
    def _extract_character_features(self, text):
        """Extract character-level features"""
        features = {}
        
        if not text:
            return features
        
        # Count vowels and consonants
        vowel_count = sum(1 for c in text if c in self.normalizer.vowels)
        consonant_count = sum(1 for c in text if c in self.normalizer.consonants)
        
        features['vowel_count'] = vowel_count
        features['consonant_count'] = consonant_count
        features['vowel_consonant_ratio'] = vowel_count / consonant_count if consonant_count > 0 else 0
        
        # Character diversity
        unique_chars = len(set(c for c in text if c in (self.normalizer.vowels | self.normalizer.consonants)))
        features['character_diversity'] = unique_chars
        
        return features
    
    def _extract_morphological_features(self, tokens):
        """Extract morphological features"""
        features = {}
        
        if not tokens:
            return features
        
        # Suffix patterns (common Bengali morphology)
        suffixes = {}
        for token in tokens:
            # 2-character suffixes
            if len(token) >= 2:
                suffix = token[-2:]
                suffixes[suffix] = suffixes.get(suffix, 0) + 1
        
        # Most common suffixes
        most_common_suffixes = sorted(suffixes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (suffix, count) in enumerate(most_common_suffixes):
            features[f'suffix_{i}_freq'] = count / len(tokens)
        
        # Padding with zeros for consistency
        for i in range(len(most_common_suffixes), 5):
            features[f'suffix_{i}_freq'] = 0
        
        return features
    
    def get_feature_names(self):
        """Get names of all extracted features"""
        # Extract features from a dummy text to get all feature names
        dummy_features = self.extract_features("এটি একটি পরীক্ষা পাঠ।")
        return sorted(dummy_features.keys())
    
    def extract_features_vector(self, text):
        """Extract features and return as ordered vector"""
        features_dict = self.extract_features(text)
        feature_names = self.get_feature_names()
        
        # Create vector ensuring all features are present
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features_dict.get(feature_name, 0))
        
        return np.array(feature_vector), feature_names
