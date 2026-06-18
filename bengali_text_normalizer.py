"""
Bengali Text Normalizer - Shadhu-Cholito normalization
Handles Bengali script normalization without pretrained models
"""

import re
import unicodedata
from collections import Counter

class BengaliNormalizer:
    """
    Normalizes Bengali text with Shadhu-Cholito standardization.
    Shadhu = Formal Bengali (standard)
    Cholito = Colloquial Bengali (spoken)
    """
    
    # Shadhu (formal) to standard mappings
    FORMAL_VARIANTS = {
        'ও': 'এবং',  # Formal conjunction to standard
        'যাহা': 'যা',  # Formal pronoun
        'কাহার': 'কার',  # Formal possessive
        'তাহার': 'তার',  # Formal possessive
        'এহেন': 'এমন',  # Formal adjective
    }
    
    def __init__(self):
        self.vowels = set('অআইঈউঊঋএঐওঔ')
        self.consonants = set('কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়')
        self.stops = set('।॥')
        
    def normalize(self, text):
        """Complete normalization pipeline"""
        if not text:
            return ""
        
        # Step 1: Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Step 2: Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 3: Normalize character variations
        text = text.replace('ঃ', ':')
        text = text.replace('ঁ', 'ং')
        text = text.replace('়', '')
        
        # Step 4: Normalize formal variants (Shadhu to cholito)
        for formal, standard in self.FORMAL_VARIANTS.items():
            text = text.replace(formal, standard)
        
        # Step 5: Remove diacritics (matras) for feature extraction
        text = self._remove_diacritics(text)
        
        # Step 6: Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Step 7: Lowercase (Bengali maintains case but normalize symbols)
        text = text.lower()
        
        return text
    
    def _remove_diacritics(self, text):
        """Remove Bengali diacritical marks (matras)"""
        matras = 'ািীুূৃেৈোৌ়ৗ'
        for matra in matras:
            text = text.replace(matra, '')
        return text
    
    def _normalize_punctuation(self, text):
        """Normalize punctuation marks"""
        text = text.replace('।', '।')
        text = text.replace('॥', '।।')
        text = re.sub(r'[.!?]{2,}', '.', text)
        return text
    
    def tokenize(self, text):
        """Tokenize into words (hand-crafted, no pretrained tokenizer)"""
        text = self.normalize(text)
        words = re.findall(r'[অ-ঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়]+', text)
        return [w for w in words if w]
    
    def get_tokens_with_pos(self, text):
        """Get tokens with POS-inspired tagging (hand-crafted)"""
        tokens = self.tokenize(text)
        pos_tags = []
        
        for token in tokens:
            tag = self._tag_pos(token)
            pos_tags.append((token, tag))
        
        return pos_tags
    
    def _tag_pos(self, token):
        """Simple hand-crafted POS tagging based on morphology"""
        # Common suffixes indicating parts of speech
        if token.endswith('ছি') or token.endswith('ছে') or token.endswith('ছিল'):
            return 'VERB'
        elif token.endswith('া') or token.endswith('ে') or token.endswith('ো'):
            return 'ADJ'  # Adjective ending
        elif token.endswith('নি') or token.endswith('বি') or token.endswith('গি'):
            return 'NOUN'
        elif token.endswith('র') or token.endswith('দের'):
            return 'PRON'  # Pronoun/possessive
        else:
            return 'NOUN'  # Default


class TextPreprocessor:
    """Complete preprocessing pipeline"""
    
    def __init__(self):
        self.normalizer = BengaliNormalizer()
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.normalizer.normalize(text)
        tokens = self.normalizer.tokenize(text)
        return tokens, text
    
    def get_sentence_tokens(self, text):
        """Split into sentences then tokenize"""
        text = self.normalizer.normalize(text)
        sentences = re.split(r'[।॥]', text)
        
        sentence_tokens = []
        for sent in sentences:
            tokens = self.normalizer.tokenize(sent)
            if tokens:
                sentence_tokens.append(tokens)
        
        return sentence_tokens
