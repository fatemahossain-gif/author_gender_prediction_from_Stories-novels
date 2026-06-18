"""
Bengali Text Normalizer - Shadhu-Cholito normalization
Handles Bengali script normalization without pretrained models
"""

import re
import unicodedata

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
        'সেই': 'সেই',  # Keep formal
        'এই': 'এই',   # Keep formal
    }
    
    # Cholito (colloquial) common contractions
    COLLOQUIAL_TO_STANDARD = {
        'র': 'রা',      # Plural marker variant
        'ে': 'তে',      # Location marker variant
        'দেখছি': 'দেখছ', # Verb variant
        'বলছি': 'বলছ',  # Verb variant
    }
    
    # Character normalizations
    CHAR_NORMALIZATION = {
        'ঃ': ':',       # Visarga (colon)
        'ঁ': 'ং',      # Anusvara variant
        '়': '',        # Nukta (diacritic)
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
        for old, new in self.CHAR_NORMALIZATION.items():
            text = text.replace(old, new)
        
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
        # Vowel signs (matras)
        matras = 'ািীুূৃেৈোৌ়ৗ'
        for matra in matras:
            text = text.replace(matra, '')
        return text
    
    def _normalize_punctuation(self, text):
        """Normalize punctuation marks"""
        # Common normalizations
        text = text.replace('।', '।')  # Danda
        text = text.replace('॥', '।।')  # Double danda
        
        # Remove extra punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        
        return text
    
    def tokenize(self, text):
        """Tokenize into words (hand-crafted, no pretrained tokenizer)"""
        # Normalize first
        text = self.normalize(text)
        
        # Split by spaces and punctuation, but keep word boundaries
        words = re.findall(r'[অ-ঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়]+', text)
        
        return [w for w in words if w]  # Remove empty strings
    
    def get_tokens_with_info(self, text):
        """Get tokens with their properties"""
        tokens = self.tokenize(text)
        token_info = []
        
        for token in tokens:
            info = {
                'token': token,
                'length': len(token),
                'has_vowel': any(c in self.vowels for c in token),
                'has_consonant': any(c in self.consonants for c in token),
            }
            token_info.append(info)
        
        return token_info


class TextPreprocessor:
    """Complete preprocessing pipeline"""
    
    def __init__(self):
        self.normalizer = BengaliNormalizer()
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        # Normalize
        text = self.normalizer.normalize(text)
        
        # Tokenize
        tokens = self.normalizer.tokenize(text)
        
        return tokens, text
    
    def get_sentence_tokens(self, text):
        """Split into sentences then tokenize"""
        text = self.normalizer.normalize(text)
        
        # Split by sentence-ending markers
        sentences = re.split(r'[।॥]', text)
        
        sentence_tokens = []
        for sent in sentences:
            tokens = self.normalizer.tokenize(sent)
            if tokens:
                sentence_tokens.append(tokens)
        
        return sentence_tokens
