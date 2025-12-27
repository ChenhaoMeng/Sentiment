"""
中国文化软实力的数字化反馈——基于《黑神话：悟空》Steam评论的中外情感倾向与文本挖掘对比分析
Author: [Your Name]
Major: [Your Major]
"""

import requests
import pandas as pd
import numpy as np
import re
import time
import jieba
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from snownlp import SnowNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SteamScraper:
    """
    Steam评论爬虫类
    """
    def __init__(self):
        self.app_id = "2358720"  # Black Myth: Wukong
        self.base_url = "https://store.steampowered.com/appreviews/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_reviews(self, language, count=2000):
        """
        获取指定语言的评论
        """
        reviews = []
        cursor = "*"
        num_per_page = 100
        
        while len(reviews) < count:
            params = {
                'json': 1,
                'filter': 'recent',
                'language': language,
                'cursor': cursor,
                'num_per_page': num_per_page
            }
            
            try:
                response = requests.get(f"{self.base_url}{self.app_id}", params=params, headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                
                if 'reviews' in data and data['reviews']:
                    reviews.extend(data['reviews'])
                    cursor = data.get('cursor', '')
                    
                    # 如果没有更多评论，跳出循环
                    if not data.get('cursor', ''):
                        break
                        
                    print(f"Fetched {len(reviews)} reviews in {language} so far...")
                    
                    # 避免请求过于频繁
                    time.sleep(1)
                else:
                    break
                    
            except Exception as e:
                print(f"Error fetching reviews: {e}")
                break
        
        # 限制数量
        reviews = reviews[:count]
        return reviews

class DataPreprocessor:
    """
    数据预处理类
    """
    def __init__(self):
        # 加载中英文停用词
        self.chinese_stopwords = self.load_chinese_stopwords()
        self.english_stopwords = set(nltk_stopwords.words('english'))
        
        # 游戏领域相关停用词
        self.game_stopwords = {
            'game', 'play', 'player', 'players', 'playing', 'played', 'character', 
            'characters', 'story', 'storyline', 'quest', 'quests', 'npc', 'items',
            'item', 'inventory', 'level', 'levels', 'character', 'characters'
        }
        
        self.english_stopwords.update(self.game_stopwords)
        
        # 初始化词形还原器
        self.lemmatizer = WordNetLemmatizer()
    
    def load_chinese_stopwords(self):
        """
        加载中文停用词表
        """
        # 如果停用词文件不存在，创建一个基础的
        stopwords_path = "/workspace/data/stopwords.txt"
        if not os.path.exists(stopwords_path):
            chinese_stopwords = [
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                '自己', '这', '那', '里', '就是', '还是', '为了', '都', '把', '这个', '什么',
                '非常', '可以', '这个', '还是', '然后', '但是', '因为', '所以', '如果', '虽然',
                '真的', '很', '太', '了', '吧', '啊', '呢', '吗', '哈', '哦', '嗯', '额', '嘿'
            ]
            with open(stopwords_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(chinese_stopwords))
        else:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                chinese_stopwords = [line.strip() for line in f if line.strip()]
        
        return set(chinese_stopwords)
    
    def clean_text(self, text):
        """
        清洗文本
        """
        if pd.isna(text) or text is None:
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除特殊字符，保留中英文字符、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:\-_\'\"]', ' ', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_clean(self, text, language='chinese'):
        """
        分词和去停用词
        """
        if not text or len(text.strip()) == 0:
            return []
        
        if language == 'chinese':
            # 中文分词
            tokens = jieba.lcut(text)
            # 去除停用词和长度小于2的词
            tokens = [token for token in tokens if token not in self.chinese_stopwords and len(token.strip()) > 1]
        else:
            # 英文分词
            tokens = word_tokenize(text.lower())
            # 去除停用词、标点符号和长度小于2的词
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.english_stopwords and token.isalpha() and len(token) > 1]
        
        return tokens

class SentimentAnalyzer:
    """
    情感分析类
    """
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text, language='chinese'):
        """
        分析情感极性
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        if language == 'chinese':
            # 使用SnowNLP进行中文情感分析
            try:
                s = SnowNLP(text)
                sentiment_score = s.sentiments  # 返回0-1之间的值，0.5为中性
                # 转换到-1到1的范围
                sentiment_score = (sentiment_score - 0.5) * 2
                return sentiment_score
            except:
                return 0.0
        else:
            # 使用VADER进行英文情感分析
            scores = self.vader_analyzer.polarity_scores(text)
            return scores['compound']

class TopicModeler:
    """
    主题建模类
    """
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.lda_model = None
        self.vectorizer = None
    
    def fit_transform(self, texts, max_features=1000):
        """
        训练LDA模型并转换文本
        """
        # 构建文档-词矩阵
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            max_df=0.8,  # 忽略出现在80%以上文档中的词
            min_df=2,    # 忽略出现在少于2个文档中的词
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # 训练LDA模型
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        
        topic_distributions = self.lda_model.fit_transform(doc_term_matrix)
        
        return topic_distributions
    
    def get_top_words_per_topic(self, top_n=10):
        """
        获取每个主题的top词
        """
        if self.lda_model is None or self.vectorizer is None:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f'Topic {topic_idx+1}'] = top_words
        
        return topics

class DataVisualizer:
    """
    数据可视化类
    """
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def plot_sentiment_distribution(self, df, output_path):
        """
        绘制情感分布图
        """
        plt.figure(figsize=(12, 6))
        
        # 按语言分组
        languages = df['language'].unique()
        
        for lang in languages:
            lang_data = df[df['language'] == lang]['sentiment_score']
            plt.hist(lang_data, alpha=0.7, label=f'{lang}', bins=30)
        
        plt.xlabel('情感得分')
        plt.ylabel('频次')
        plt.title('中英文评论情感得分分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_wordcloud(self, texts, language, output_path):
        """
        生成词云
        """
        # 合并所有文本
        combined_text = ' '.join(texts)
        
        # 根据语言设置词云参数
        if language == 'chinese':
            # 中文词云
            wc = WordCloud(
                font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # 可能需要调整字体路径
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
        else:
            # 英文词云
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(combined_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{language} 词云图')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    主函数 - 执行完整的分析流程
    """
    print("开始分析《黑神话：悟空》Steam评论...")
    
    # 1. 数据获取
    print("\n1. 开始爬取Steam评论数据...")
    scraper = SteamScraper()
    
    # 获取中文评论
    print("获取中文评论...")
    chinese_reviews = scraper.get_reviews('schinese', count=2000)
    print(f"获取到 {len(chinese_reviews)} 条中文评论")
    
    # 获取英文评论
    print("获取英文评论...")
    english_reviews = scraper.get_reviews('english', count=2000)
    print(f"获取到 {len(english_reviews)} 条英文评论")
    
    # 2. 数据预处理
    print("\n2. 开始数据预处理...")
    preprocessor = DataPreprocessor()
    
    # 创建DataFrame
    data = []
    
    # 处理中文评论
    for review in chinese_reviews:
        review_id = review.get('recommendationid', '')
        review_text = preprocessor.clean_text(review.get('review', ''))
        voted_up = review.get('voted_up', False)
        timestamp = review.get('timestamp_created', 0)
        playtime = review.get('author', {}).get('playtime_forever', 0)
        
        data.append({
            'review_id': review_id,
            'language': 'Chinese',
            'review_text': review_text,
            'voted_up': voted_up,
            'timestamp_created': timestamp,
            'playtime_forever': playtime,
            'tokens': ' '.join(preprocessor.tokenize_and_clean(review_text, 'chinese'))
        })
    
    # 处理英文评论
    for review in english_reviews:
        review_id = review.get('recommendationid', '')
        review_text = preprocessor.clean_text(review.get('review', ''))
        voted_up = review.get('voted_up', False)
        timestamp = review.get('timestamp_created', 0)
        playtime = review.get('author', {}).get('playtime_forever', 0)
        
        data.append({
            'review_id': review_id,
            'language': 'English',
            'review_text': review_text,
            'voted_up': voted_up,
            'timestamp_created': timestamp,
            'playtime_forever': playtime,
            'tokens': ' '.join(preprocessor.tokenize_and_clean(review_text, 'english'))
        })
    
    df = pd.DataFrame(data)
    print(f"预处理完成，共 {len(df)} 条评论")
    
    # 保存原始数据
    raw_data_path = "/workspace/data/raw_reviews.xlsx"
    df.to_excel(raw_data_path, index=False)
    print(f"原始数据已保存至 {raw_data_path}")
    
    # 3. 情感分析
    print("\n3. 开始情感分析...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # 为每条评论计算情感得分
    sentiment_scores = []
    for idx, row in df.iterrows():
        score = sentiment_analyzer.analyze_sentiment(row['review_text'], row['language'].lower())
        sentiment_scores.append(score)
        
        if idx % 500 == 0:
            print(f"已处理 {idx}/{len(df)} 条评论")
    
    df['sentiment_score'] = sentiment_scores
    print("情感分析完成")
    
    # 4. 主题建模
    print("\n4. 开始主题建模...")
    topic_modeler = TopicModeler(n_topics=5)
    
    # 按语言分组进行主题建模
    for lang in df['language'].unique():
        lang_texts = df[df['language'] == lang]['tokens'].tolist()
        
        if len(lang_texts) > 0:
            print(f"对 {lang} 评论进行主题建模...")
            topic_distributions = topic_modeler.fit_transform(lang_texts)
            
            # 将主题分布添加到DataFrame
            for i in range(topic_modeler.n_topics):
                df[f'{lang}_topic_{i+1}_prob'] = topic_distributions[:, i]
            
            # 输出主题词
            topics = topic_modeler.get_top_words_per_topic(top_n=10)
            print(f"{lang} 主题建模结果:")
            for topic, words in topics.items():
                print(f"  {topic}: {', '.join(words[:5])}")
    
    # 5. 数据可视化
    print("\n5. 开始数据可视化...")
    visualizer = DataVisualizer()
    
    # 绘制情感分布图
    sentiment_dist_path = "/workspace/sentiment_distribution.png"
    visualizer.plot_sentiment_distribution(df, sentiment_dist_path)
    print(f"情感分布图已保存至 {sentiment_dist_path}")
    
    # 生成词云
    for lang in df['language'].unique():
        lang_texts = df[df['language'] == lang]['tokens'].tolist()
        if len(lang_texts) > 0:
            wordcloud_path = f"/workspace/{lang.lower()}_wordcloud.png"
            visualizer.generate_wordcloud(lang_texts, lang, wordcloud_path)
            print(f"{lang} 词云图已保存至 {wordcloud_path}")
    
    # 6. 结果导出
    print("\n6. 导出分析结果...")
    result_path = "/workspace/result_data.xlsx"
    df.to_excel(result_path, index=False)
    print(f"分析结果已保存至 {result_path}")
    
    # 7. 打印统计摘要
    print("\n7. 统计摘要:")
    for lang in df['language'].unique():
        lang_data = df[df['language'] == lang]
        avg_sentiment = lang_data['sentiment_score'].mean()
        print(f"{lang}评论平均情感得分: {avg_sentiment:.3f}")
        
        # 计算推荐率
        recommend_rate = lang_data['voted_up'].mean()
        print(f"{lang}评论推荐率: {recommend_rate:.3f}")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
