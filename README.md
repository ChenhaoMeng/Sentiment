# Sentiment Analysis Project

This project performs sentiment analysis and cultural feedback research on Chinese cultural products using text mining techniques, with a focus on analyzing Steam reviews for "Black Myth: Wukong".

## Project Overview

This is a comprehensive sentiment analysis project that includes:
- Data preprocessing and cleaning
- Multilingual sentiment analysis (Chinese/English)
- Topic modeling for both languages
- Visualization of results
- Word cloud generation
- Statistical analysis

The project analyzes player reviews of "Black Myth: Wukong" on Steam to compare sentiment and focus differences between domestic and international players, evaluating the effectiveness of Chinese cultural products in international communication.

## Project Goals

1. Analyze sentiment differences between domestic and international players for "Black Myth: Wukong"
2. Mine focus differences between players from different cultural backgrounds
3. Evaluate the cultural acceptance of Chinese game products in international markets
4. Explore the role of digital entertainment products in cultural output

## Deployment to GitHub

To deploy this project to GitHub, follow these steps:

1. Run the deployment script:
   ```bash
   ./deploy_to_github.sh
   ```

2. Follow the interactive prompts to create a GitHub repository and push the code.

## Files Included

- `updated_analysis.py` - Main analysis script
- `enhanced_sentiment_analyzer.py` - Enhanced sentiment analysis implementation
- `config.py` - Configuration settings
- `main.py` - Entry point for the application
- `requirements.txt` - Python dependencies
- Various data files in the `/data` directory
- Documentation files in various formats

## Project Structure

```
/workspace/
├── BlackMythAnalysis.py                    # Original analysis code
├── updated_analysis.py                     # Updated analysis code (using config and enhanced sentiment analysis)
├── WangXiaoming_CS_BlackMythAnalysis.docx  # Project report
├── create_files.py                         # Data file creation script
├── create_doc.py                           # Word document creation script
├── config.py                               # Configuration file
├── enhanced_sentiment_analyzer.py          # Enhanced sentiment analyzer
├── IMPROVEMENTS_RECOMMENDATIONS.md         # Improvement recommendations document
├── requirements.txt                        # Dependencies list
├── result_data.xlsx                        # Analysis result data
├── data/                                   # Data directory
│   ├── raw_reviews.xlsx                    # Raw review data
│   └── stopwords.txt                       # Stopwords list
├── deploy_to_github.sh                     # GitHub deployment script
└── DEPLOY_TO_GITHUB.md                     # GitHub deployment guide
```

## Tech Stack

- Python 3.x
- pandas - Data processing
- numpy - Numerical computing
- matplotlib/seaborn - Data visualization
- jieba - Chinese text segmentation
- nltk - English text processing
- snownlp - Chinese sentiment analysis
- scikit-learn - Machine learning algorithms
- wordcloud - Word cloud generation
- openpyxl - Excel file processing
- python-docx - Word document processing

## Features

1. **Data crawling and preprocessing** - Simulate acquisition of Steam review data and cleaning
2. **Chinese/English text segmentation** - Use jieba for Chinese, nltk for English
3. **Sentiment analysis** - Perform sentiment polarity analysis on reviews
4. **Topic modeling** - Use LDA model to mine review topics
5. **Visualization** - Generate sentiment distribution charts, word clouds, etc.
6. **Comparative analysis** - Compare reviews between domestic and international players

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the original analysis program:
   ```bash
   python BlackMythAnalysis.py
   ```

3. Or run the updated analysis program (recommended):
   ```bash
   python updated_analysis.py
   ```

4. Check results:
   - Analysis results saved in `result_data.xlsx`
   - Generated charts saved in the project directory

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Project Significance

This project explores the role of popular culture products (such as AAA games) as carriers of Chinese culture abroad from a digital humanities perspective. Through quantitative analysis of sentiment differences and focus points between domestic and international players, it provides data support and practical references for "telling Chinese stories well".

## Research Hypotheses

1. The sentiment mean of Chinese reviews is higher than that of English reviews, because domestic players have national pride
2. English reviews focus more on "Performance" and "Difficulty", while Chinese reviews focus more on "Story" and "Art"

## Data Source

This project uses simulated data. In practical applications, it can be replaced with real Steam review data.