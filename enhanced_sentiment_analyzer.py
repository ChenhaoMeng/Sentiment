"""
Enhanced sentiment analyzer with multiple models and game-specific sentiment
"""
import jieba
import nltk
from snownlp import SnowNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import GAME_SENTIMENT_WORDS
import re

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer that combines multiple models and game-specific sentiment
    """
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.textblob_available = True
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
        except ImportError:
            self.textblob_available = False

    def analyze_sentiment_ensemble(self, text, language='chinese'):
        """
        Analyze sentiment using ensemble of models
        """
        if not text or len(text.strip()) == 0:
            return 0.0
            
        scores = []
        
        if language == 'chinese' or language == 'schinese':
            # SnowNLP for Chinese
            s = SnowNLP(text)
            snow_score = (s.sentiments - 0.5) * 2  # Convert to -1 to 1 scale
            scores.append(snow_score)
        else:
            # VADER for English
            vader_score = self.vader_analyzer.polarity_scores(text)['compound']
            scores.append(vader_score)
            
            # TextBlob if available
            if self.textblob_available:
                blob = self.TextBlob(text)
                textblob_score = blob.sentiment.polarity
                scores.append(textblob_score)
        
        # Calculate weighted average
        return sum(scores) / len(scores) if scores else 0.0

    def analyze_sentiment_with_domain_knowledge(self, text, language='chinese'):
        """
        Analyze sentiment with game-specific domain knowledge
        """
        base_score = self.analyze_sentiment_ensemble(text, language)
        
        # Apply game-specific sentiment adjustments
        text_lower = text.lower()
        adjustment = 0.0
        
        for word, weight in GAME_SENTIMENT_WORDS.items():
            if word in text_lower:
                adjustment += weight * 0.1  # Small adjustment factor
        
        # Clamp the result to [-1, 1] range
        final_score = max(-1.0, min(1.0, base_score + adjustment))
        
        return final_score

    def analyze_sentiment(self, text, language='chinese'):
        """
        Main sentiment analysis method that uses domain knowledge
        """
        return self.analyze_sentiment_with_domain_knowledge(text, language)

# Example usage and testing
if __name__ == "__main__":
    analyzer = EnhancedSentimentAnalyzer()
    
    # Test with various game-related texts
    test_texts = [
        ("This game is amazing and beautiful!", "english"),
        ("画面太棒了，游戏性很强", "chinese"),
        ("The game keeps crashing, very frustrating", "english"),
        ("优化太差了，老是卡顿", "chinese"),
        ("Challenging but fair gameplay", "english"),
        ("很有挑战性，但很公平", "chinese")
    ]
    
    print("Enhanced Sentiment Analysis Results:")
    print("-" * 50)
    
    for text, lang in test_texts:
        score = analyzer.analyze_sentiment(text, lang)
        print(f"Text: {text}")
        print(f"Language: {lang}")
        print(f"Score: {score:.3f}")
        print("-" * 30)