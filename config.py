"""
Configuration file for Black Myth: Wukong Steam reviews analysis project
"""

CONFIG = {
    'app_id': '2358720',  # Black Myth: Wukong
    'languages': {
        'chinese': 'schinese',
        'english': 'english'
    },
    'review_count': 2000,
    'num_per_page': 100,
    'request_delay': 1,  # seconds between requests
    'max_retries': 3,
    
    'output_paths': {
        'raw_data': 'data/raw_reviews.xlsx',
        'result_data': 'result_data.xlsx',
        'sentiment_plot': 'sentiment_distribution.png',
        'chinese_wordcloud': 'chinese_wordcloud.png',
        'english_wordcloud': 'english_wordcloud.png',
        'comprehensive_analysis': 'comprehensive_analysis.png'
    },
    
    'sentiment': {
        'vader_enabled': True,
        'snownlp_enabled': True
    },
    
    'topic_modeling': {
        'n_topics': 5,
        'max_features': 1000
    },
    
    'visualization': {
        'figure_size': (12, 8),
        'dpi': 300,
        'font_family': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    }
}

# Game-specific sentiment words
GAME_SENTIMENT_WORDS = {
    # Positive
    'amazing': 0.8, 'brilliant': 0.8, 'incredible': 0.8, 'fantastic': 0.8,
    'beautiful': 0.7, 'gorgeous': 0.7, 'stunning': 0.7, 'impressive': 0.7,
    'fun': 0.6, 'enjoyable': 0.6, 'engaging': 0.6, 'addictive': 0.6,
    'masterpiece': 0.9, 'excellent': 0.8, 'outstanding': 0.8, 'remarkable': 0.7,
    
    # Negative
    'buggy': -0.7, 'crash': -0.7, 'lag': -0.7, 'glitch': -0.7,
    'frustrating': -0.6, 'annoying': -0.6, 'difficult': -0.5, 'hard': -0.4,
    'terrible': -0.8, 'awful': -0.8, 'horrible': -0.8, 'disappointing': -0.7,
    'boring': -0.6, 'broke': -0.7, 'broken': -0.7, 'unplayable': -0.8,
    
    # Game-specific terms
    'hardcore': 0.2, 'challenging': 0.3, 'soulslike': 0.2, 'grindy': -0.4,
    'polish': 0.5, 'optimization': 0.2, 'performance': 0.3, 'immersive': 0.6,
    'story': 0.4, 'art': 0.5, 'graphics': 0.6, 'atmosphere': 0.5,
    'mechanics': 0.3, 'controls': 0.2, 'balance': 0.4, 'difficulty': 0.1
}