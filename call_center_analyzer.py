import pandas as pd
import numpy as np
import re
import torch
import logging
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect_langs, LangDetectException
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(word.strip().lower() for word in f.read().split() if word.strip())
    except FileNotFoundError:
        print(f"Stopwords file not found: {filepath}")
        return set()
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()

class CallCenterDialogProcessor:
    def __init__(self):
        self.operator_patterns = [
            r'меня зовут сара', r'я буду рада вам помочь', r'здравствуйте.*казахтелеком',
            r'сәлеметсіз бе.*қазақтелеком', r'я переключаю вас', r'пожалуйста.*задайте.*вопрос',
            r'в целях контроля качества', r'для получения.*консультации',
        ]
        
        self.customer_emotion_indicators = {
            'negative': ['не работает', 'не могу', 'проблема', 'сбой', 'не помогли', 'не подключается', 'не дозвонилась'],
            'positive': ['спасибо', 'хорошо', 'отлично', 'рахмет', 'жарайды', 'помогло'],
            'frustrated': ['соедините с оператором', 'связаться с оператором', 'свяжите меня с оператором', 'я уже все проверила'],
            'polite': ['пожалуйста', 'извините', 'кешіріңіз', 'будьте добры']
        }
    
    def parse_dialog(self, dialog_text):
        if not isinstance(dialog_text, str):
            return []
        
        turns = re.split(r'\n?(\d+): -', dialog_text)
        parsed_turns = []
        
        for i in range(1, len(turns), 2):
            if i + 1 < len(turns):
                turn_number = int(turns[i])
                turn_text = turns[i + 1].strip()
                
                if turn_text and turn_text != '-':
                    parts = turn_text.split('\n-')
                    
                    for j, part in enumerate(parts):
                        part = part.strip(' -\n')
                        if part:
                            speaker = self._identify_speaker(part, turn_number, j)
                            parsed_turns.append({
                                'turn': turn_number,
                                'part': j,
                                'speaker': speaker,
                                'text': part,
                                'length': len(part)
                            })
        
        return parsed_turns
    
    def _identify_speaker(self, text, turn_number, part_index):
        text_lower = text.lower()
        
        for pattern in self.operator_patterns:
            if re.search(pattern, text_lower):
                return 'operator'
        
        if turn_number == 1 and part_index == 0:
            return 'operator'
        
        if len(text) < 20 and text_lower in ['да', 'нет', '1', '2', 'на русском', 'как физическое']:
            return 'customer'
        
        tech_keywords = ['проверьте', 'нажмите', 'введите', 'переподключите', 'подключение']
        if any(keyword in text_lower for keyword in tech_keywords):
            return 'operator'
        
        return 'customer' if part_index == 0 else 'operator'
    
    def analyze_customer_sentiment(self, customer_text):
        if not customer_text or len(customer_text.strip()) < 3:
            return 'neutral', 0.0, []
        
        text_lower = customer_text.lower()
        found_emotions = []
        emotion_scores = defaultdict(int)
        
        for emotion, indicators in self.customer_emotion_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    emotion_scores[emotion] += 1
                    found_emotions.append(indicator)
        
        if emotion_scores['frustrated'] > 0:
            return 'negative', emotion_scores['frustrated'] / len(text_lower.split()), found_emotions
        elif emotion_scores['negative'] > 0:
            return 'negative', emotion_scores['negative'] / len(text_lower.split()), found_emotions
        elif emotion_scores['positive'] > 0:
            return 'positive', emotion_scores['positive'] / len(text_lower.split()), found_emotions
        elif emotion_scores['polite'] > 0:
            return 'neutral', emotion_scores['polite'] / len(text_lower.split()), found_emotions
        else:
            return 'neutral', 0.0, []

class SentimentAnalyzer:
    def __init__(self, LOCAL_MODELS):
        self.device = 0 if torch.cuda.is_available() else -1
        self.LOCAL_MODELS = LOCAL_MODELS
        self.models = self._load_models()
        self.vader = SentimentIntensityAnalyzer()
        self.error_log = []

    def _load_models(self):
        models = {}
        for name, path in self.LOCAL_MODELS.items():
            if name == "embedding":
                continue
            try:
                models[name] = pipeline("sentiment-analysis", model=path, device=self.device, 
                                      truncation=True, max_length=512, batch_size=8)
                print(f"loaded {name} model")
            except Exception as e:
                models[name] = None
                print(f"failed to load {name} model: {e}")
        return models

    def _clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _truncate_text(self, text, max_words=400):
        words = text.split()
        return " ".join(words[:max_words])

    def analyze(self, text: str, lang: str) -> dict:
        base_result = {'sentiment': 'neutral', 'confidence': 0.0, 'method': 'fallback'}
        
        try:
            text = self._clean_text(text)
            if not text:
                return base_result
            
            text = self._truncate_text(text)
            
            if lang == 'auto':
                try:
                    lang = detect_langs(text)[0].lang
                except LangDetectException:
                    lang = 'en'
            
            if lang == 'ru' and self.models.get('russian'):
                return self._analyze_ru(text)
            elif lang == 'en' and self.models.get('english'):
                return self._analyze_en(text)
            elif lang in ('kk', 'kz'):
                return self._analyze_kz(text)
            elif self.models.get('multilingual'):
                return self._analyze_multilingual(text)
            else:
                return self._analyze_fallback(text, lang)
        
        except Exception as e:
            self.error_log.append(f"{text[:50]}... | Error: {str(e)}")
            return base_result

    def _safe_model_predict(self, model, text):
        try:
            result = model(text)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
            return None
        except Exception as e:
            return None

    def _analyze_multilingual(self, text: str) -> dict:
        try:
            result = self._safe_model_predict(self.models['multilingual'], text)
            if result and isinstance(result, list):
                scores = {item['label']: item['score'] for item in result}
                sentiment = max(scores, key=scores.get)
                sentiment = sentiment.lower()
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                return {'sentiment': sentiment, 'confidence': float(scores[max(scores, key=scores.get)]), 'method': 'multilingual'}
            return self._analyze_fallback(text, 'en')
        except Exception as e:
            return self._analyze_fallback(text, 'en')

    def _analyze_ru(self, text: str) -> dict:
        try:
            result = self._safe_model_predict(self.models['russian'], text)
            if result and isinstance(result, list):
                scores = {item['label'].lower(): item['score'] for item in result}
                sentiment = max(scores, key=scores.get)
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                return {'sentiment': sentiment, 'confidence': float(scores[max(scores, key=scores.get)]), 'method': 'ru'}
            return self._analyze_fallback(text, 'ru')
        except Exception as e:
            return self._analyze_fallback(text, 'ru')

    def _analyze_en(self, text: str) -> dict:
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {'sentiment': sentiment, 'confidence': float(abs(compound)), 'method': 'vader'}
        except Exception as e:
            return self._analyze_fallback(text, 'en')

    def _normalize_kz_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    def _analyze_kz(self, text: str) -> dict:
        try:
            norm_text = self._normalize_kz_text(text)
            
            if self.models.get('multilingual') is not None:
                result = self._safe_model_predict(self.models['multilingual'], norm_text)
                if result and isinstance(result, list):
                    scores = {item['label'].lower(): item['score'] for item in result}
                    sentiment = max(scores, key=scores.get)
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'
                    return {'sentiment': sentiment, 'confidence': float(scores[max(scores, key=scores.get)]), 'method': 'kk-multilingual+norm'}
            
            return self._analyze_fallback(norm_text, 'kk')
        except Exception as e:
            return self._analyze_fallback(text, 'kk')

    def _analyze_fallback(self, text: str, lang: str) -> dict:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {'sentiment': sentiment, 'confidence': float(abs(polarity)), 'method': 'textblob'}
        except Exception as e:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'method': 'fallback'}

class DialogTextClusterAnalyzer:
    def __init__(self, LOCAL_MODELS, stopwords_file='Stop Words List - Kazakh.txt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LOCAL_MODELS = LOCAL_MODELS
        self.stopwords = load_stopwords(stopwords_file)
        
        try:
            self.embedder = SentenceTransformer(LOCAL_MODELS["embedding"], device=self.device)
            self.use_embeddings = True
            print(f"Loaded embeddings")
        except Exception as e:
            print(f"Failed to load embeddings, fallback to TF-IDF")
            self.use_embeddings = False
            self.vectorizer = TfidfVectorizer(max_features=5000, 
                                           stop_words=list(self.stopwords) if self.stopwords else None)
        
        self.sentiment_analyzer = SentimentAnalyzer(LOCAL_MODELS)
        self.dialog_processor = CallCenterDialogProcessor()

    def _remove_stopwords(self, text: str) -> str:
        return " ".join([w for w in text.split() if w.lower() not in self.stopwords])

    def vectorize(self, texts: list) -> np.ndarray:
        cleaned_texts = [self._remove_stopwords(t) for t in texts]
        
        if self.use_embeddings:
            return self.embedder.encode(cleaned_texts, show_progress_bar=True, 
                                      batch_size=64, convert_to_numpy=True)
        else:
            return self.vectorizer.fit_transform(cleaned_texts).toarray()

    def cluster(self, features: np.ndarray, n_clusters=None) -> tuple:
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(features)
        
        if len(features) > 10000:
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        return model.fit_predict(features), n_clusters

    def _find_optimal_clusters(self, features: np.ndarray, max_k=8) -> int:
        if len(features) < 10:
            return min(3, len(features))
        
        scores = []
        range_k = range(2, min(max_k + 1, len(features)))
        
        for k in tqdm(range_k, desc="Finding optimal clusters"):
            try:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
                labels = kmeans.fit_predict(features)
                
                if len(set(labels)) > 1:
                    scores.append(silhouette_score(features, labels))
                else:
                    scores.append(-1)
            except Exception:
                scores.append(-1)
        
        return int(np.argmax(scores) + 2) if max(scores) > 0 else 3

    def reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            return pca.fit_transform(features)
        return features

    def process_dialogs(self, df: pd.DataFrame, dialog_col='dialog_text') -> tuple:
        print("Processing dialogs into turns...")
        
        all_turns = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dialogs"):
            dialog_text = row[dialog_col]
            turns = self.dialog_processor.parse_dialog(dialog_text)
            
            for turn in turns:
                turn['dialog_id'] = idx
                
                if turn['speaker'] == 'customer':
                    emotion, confidence, indicators = self.dialog_processor.analyze_customer_sentiment(turn['text'])
                    turn['emotion'] = emotion
                    turn['emotion_confidence'] = confidence
                    turn['emotion_indicators'] = indicators
                
                all_turns.append(turn)
        
        turns_df = pd.DataFrame(all_turns)
        print(f"Extracted {len(turns_df)} turns from {len(df)} dialogs")
        return turns_df

    def analyze_turns_sentiment(self, turns_df: pd.DataFrame) -> pd.DataFrame:
        print("Analyzing sentiment for customer turns...")
        
        results = []
        
        for idx, row in tqdm(turns_df.iterrows(), total=len(turns_df), desc="Sentiment analysis"):
            if row['speaker'] == 'customer':
                try:
                    lang = 'auto'
                    if any(char in row['text'] for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                        lang = 'ru'
                    elif any(char in row['text'] for char in 'әғқңөұүһі'):
                        lang = 'kk'
                    
                    res = self.sentiment_analyzer.analyze(row['text'], lang)
                except Exception as e:
                    res = {'sentiment': 'neutral', 'confidence': 0.0, 'method': 'error'}
            else:
                res = {'sentiment': 'neutral', 'confidence': 0.0, 'method': 'operator'}
            
            results.append(res)
        
        turns_df = turns_df.copy()
        turns_df['ml_sentiment'] = [r['sentiment'] for r in results]
        turns_df['ml_confidence'] = [r['confidence'] for r in results]
        turns_df['ml_method'] = [r['method'] for r in results]
        
        return turns_df

    def cluster_customer_turns(self, turns_df: pd.DataFrame) -> pd.DataFrame:
        print("Clustering customer turns...")
        
        customer_turns = turns_df[turns_df['speaker'] == 'customer'].copy()
        
        if len(customer_turns) < 10:
            print("Not enough customer turns for clustering")
            return customer_turns
        
        texts = customer_turns['text'].tolist()
        features = self.vectorize(texts)
        
        clusters, n_clusters = self.cluster(features)
        customer_turns['cluster'] = clusters
        
        print(f"Found {n_clusters} customer turn clusters")
        
        return customer_turns

    def cluster_complaint_reasons(self, customer_turns: pd.DataFrame) -> pd.DataFrame:
        print("Clustering complaint reasons...")
        
        negative_turns = customer_turns[customer_turns['ml_sentiment'] == 'negative'].copy()
        
        if len(negative_turns) < 5:
            print("Not enough negative turns for complaint clustering")
            return negative_turns
        
        texts = negative_turns['text'].tolist()
        features = self.vectorize(texts)
        
        clusters, n_clusters = self.cluster(features, n_clusters=min(5, len(negative_turns)))
        negative_turns['complaint_cluster'] = clusters
        
        print(f"Found {n_clusters} complaint reason clusters")
        
        return negative_turns

    def visualize_results(self, customer_turns: pd.DataFrame, complaint_turns: pd.DataFrame = None):
        if 'cluster' not in customer_turns.columns or len(customer_turns) < 10:
            print("Insufficient data for visualization")
            return
        
        features = self.vectorize(customer_turns['text'].tolist())
        points = self.reduce_dimensions(features)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.scatter(points[:, 0], points[:, 1], c=customer_turns['cluster'], cmap='tab20', alpha=0.7)
        plt.title('Customer Turn Clusters')
        plt.colorbar()
        
        plt.subplot(232)
        color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
        colors = [color_map.get(s, 'gray') for s in customer_turns.get('ml_sentiment', ['neutral']*len(customer_turns))]
        plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
        plt.title('Customer Sentiment')
        
        plt.subplot(233)
        if 'ml_sentiment' in customer_turns.columns:
            sentiment_counts = customer_turns['ml_sentiment'].value_counts()
            sentiment_counts.plot(kind='bar', color=[color_map.get(x, 'gray') for x in sentiment_counts.index])
            plt.title('Sentiment Distribution')
            plt.xticks(rotation=45)
        
        plt.subplot(234)
        if 'cluster' in customer_turns.columns:
            cluster_sizes = customer_turns['cluster'].value_counts().sort_index()
            cluster_sizes.plot(kind='bar')
            plt.title('Cluster Sizes')
            plt.xlabel('Cluster')
            plt.ylabel('Count')
        
        if complaint_turns is not None and 'complaint_cluster' in complaint_turns.columns:
            plt.subplot(235)
            complaint_features = self.vectorize(complaint_turns['text'].tolist())
            complaint_points = self.reduce_dimensions(complaint_features)
            plt.scatter(complaint_points[:, 0], complaint_points[:, 1], 
                       c=complaint_turns['complaint_cluster'], cmap='tab20', alpha=0.7)
            plt.title('Complaint Reason Clusters')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def generate_reports(self, customer_turns: pd.DataFrame, turns_df: pd.DataFrame, complaint_turns: pd.DataFrame = None) -> tuple:
        print("Generating reports...")
        
        cluster_report = []
        if 'cluster' in customer_turns.columns:
            for cluster_id in sorted(customer_turns['cluster'].unique()):
                cluster_data = customer_turns[customer_turns['cluster'] == cluster_id]
                
                sentiment_dist = cluster_data.get('ml_sentiment', pd.Series(['neutral']*len(cluster_data))).value_counts()
                dominant_sentiment = sentiment_dist.idxmax() if not sentiment_dist.empty else 'neutral'
                
                sample_texts = cluster_data['text'].head(3).tolist()
                
                cluster_report.append({
                    'cluster': int(cluster_id),
                    'size': int(len(cluster_data)),
                    'dominant_sentiment': dominant_sentiment,
                    'sentiment_ratio': float(sentiment_dist.max() / len(cluster_data)) if not sentiment_dist.empty else 0.0,
                    'avg_length': float(cluster_data['length'].mean()),
                    'sample_texts': sample_texts
                })
        
        complaint_report = []
        if complaint_turns is not None and 'complaint_cluster' in complaint_turns.columns:
            for cluster_id in sorted(complaint_turns['complaint_cluster'].unique()):
                cluster_data = complaint_turns[complaint_turns['complaint_cluster'] == cluster_id]
                
                sample_texts = cluster_data['text'].head(3).tolist()
                
                complaint_report.append({
                    'complaint_cluster': int(cluster_id),
                    'size': int(len(cluster_data)),
                    'avg_length': float(cluster_data['length'].mean()),
                    'sample_texts': sample_texts
                })
        
        dialog_stats = []
        for dialog_id in turns_df['dialog_id'].unique():
            dialog_turns = turns_df[turns_df['dialog_id'] == dialog_id]
            customer_dialog_turns = dialog_turns[dialog_turns['speaker'] == 'customer']
            
            sentiments = customer_dialog_turns.get('ml_sentiment', pd.Series(['neutral']*len(customer_dialog_turns))).tolist()
            
            dialog_stats.append({
                'dialog_id': int(dialog_id),
                'total_turns': int(len(dialog_turns)),
                'customer_turns': int(len(customer_dialog_turns)),
                'sentiments': sentiments,
                'has_negative': 'negative' in sentiments
            })
        
        return pd.DataFrame(cluster_report), pd.DataFrame(complaint_report), pd.DataFrame(dialog_stats)

def analyze_call_center_dialogs(df: pd.DataFrame, LOCAL_MODELS: dict, dialog_col='dialog_text', 
                               stopwords_file='Stop Words List - Kazakh.txt'):
    if df.empty:
        print("Empty dataframe")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df = df.copy()
    df[dialog_col] = df[dialog_col].fillna('').astype(str)
    
    print("Initializing dialog analyzer...")
    analyzer = DialogTextClusterAnalyzer(LOCAL_MODELS, stopwords_file=stopwords_file)
    
    turns_df = analyzer.process_dialogs(df, dialog_col)
    
    if not turns_df.empty:
        turns_df = analyzer.analyze_turns_sentiment(turns_df)
        customer_turns = analyzer.cluster_customer_turns(turns_df)
        complaint_turns = analyzer.cluster_complaint_reasons(customer_turns)
        
        analyzer.visualize_results(customer_turns, complaint_turns)
        
        cluster_report, complaint_report, dialog_report = analyzer.generate_reports(customer_turns, turns_df, complaint_turns)
        
        print("\nCluster Report:")
        print(cluster_report)
        
        if not complaint_report.empty:
            print("\nComplaint Report:")
            print(complaint_report)
        
        print("\nDialog Statistics:")
        print(f"Total dialogs: {len(dialog_report)}")
        print(f"Dialogs with negative sentiment: {dialog_report['has_negative'].sum()}")
        
        if analyzer.sentiment_analyzer.error_log:
            print(f"\nSentiment errors: {len(analyzer.sentiment_analyzer.error_log)}")
        
        print("Analysis completed")
        return df, turns_df, customer_turns, cluster_report, complaint_report
    
    else:
        print("No turns extracted")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()