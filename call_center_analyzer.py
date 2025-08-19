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
        
        self.complaint_indicators = {
            'technical': ['не работает', 'не подключается', 'сбой', 'отключился', 'нет связи', 'медленно работает'],
            'billing': ['счет', 'оплата', 'деньги', 'тариф', 'абонентская плата', 'списали'],
            'service': ['не помогли', 'плохое обслуживание', 'долго ждал', 'не дозвонилась', 'соедините с оператором'],
            'equipment': ['роутер', 'модем', 'приставка', 'оборудование', 'не включается']
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
            return 'neutral', 0.0
        
        text_lower = customer_text.lower()
        
        negative_indicators = ['не работает', 'не могу', 'проблема', 'сбой', 'не помогли', 'не подключается', 'не дозвонилась']
        positive_indicators = ['спасибо', 'хорошо', 'отлично', 'рахмет', 'жарайды', 'помогло']
        
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        
        if negative_score > positive_score:
            return 'negative', negative_score / len(text_lower.split())
        elif positive_score > negative_score:
            return 'positive', positive_score / len(text_lower.split())
        else:
            return 'neutral', 0.0
    
    def identify_complaint_type(self, customer_text):
        text_lower = customer_text.lower()
        complaint_scores = {}
        
        for complaint_type, indicators in self.complaint_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                complaint_scores[complaint_type] = score
        
        if complaint_scores:
            return max(complaint_scores, key=complaint_scores.get)
        return 'general'

class SentimentAnalyzer:
    def __init__(self, LOCAL_MODELS):
        self.device = 0 if torch.cuda.is_available() else -1
        self.LOCAL_MODELS = LOCAL_MODELS
        self.models = self._load_models()
        self.vader = SentimentIntensityAnalyzer()

    def _load_models(self):
        models = {}
        for name, path in self.LOCAL_MODELS.items():
            if name == "embedding":
                continue
            try:
                models[name] = pipeline("sentiment-analysis", model=path, device=self.device, 
                                      truncation=True, max_length=512, batch_size=8)
                print(f"Loaded {name} model")
            except Exception as e:
                models[name] = None
                print(f"Failed to load {name}: {e}")
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
        try:
            text = self._clean_text(text)
            if not text:
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
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
            elif self.models.get('multilingual'):
                return self._analyze_multilingual(text)
            else:
                return self._analyze_fallback(text)
        
        except Exception as e:
            return {'sentiment': 'neutral', 'confidence': 0.0}

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

    def _normalize_sentiment(self, sentiment):
        sentiment = sentiment.lower()
        if sentiment in ['positive', 'pos']:
            return 'positive'
        elif sentiment in ['negative', 'neg']:
            return 'negative'
        else:
            return 'neutral'

    def _analyze_multilingual(self, text: str) -> dict:
        try:
            result = self._safe_model_predict(self.models['multilingual'], text)
            if result and isinstance(result, list):
                scores = {item['label']: item['score'] for item in result}
                sentiment = max(scores, key=scores.get)
                return {'sentiment': self._normalize_sentiment(sentiment), 'confidence': float(scores[sentiment])}
            return self._analyze_fallback(text)
        except Exception as e:
            return self._analyze_fallback(text)

    def _analyze_ru(self, text: str) -> dict:
        try:
            result = self._safe_model_predict(self.models['russian'], text)
            if result and isinstance(result, list):
                scores = {item['label'].lower(): item['score'] for item in result}
                sentiment = max(scores, key=scores.get)
                return {'sentiment': self._normalize_sentiment(sentiment), 'confidence': float(scores[sentiment])}
            return self._analyze_fallback(text)
        except Exception as e:
            return self._analyze_fallback(text)

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
            
            return {'sentiment': sentiment, 'confidence': float(abs(compound))}
        except Exception as e:
            return self._analyze_fallback(text)

    def _analyze_fallback(self, text: str) -> dict:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {'sentiment': sentiment, 'confidence': float(abs(polarity))}
        except Exception as e:
            return {'sentiment': 'neutral', 'confidence': 0.0}

class ComplaintClusterAnalyzer:
    def __init__(self, LOCAL_MODELS, stopwords_file='Stop Words List - Kazakh.txt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LOCAL_MODELS = LOCAL_MODELS
        self.stopwords = load_stopwords(stopwords_file)
        
        try:
            self.embedder = SentenceTransformer(LOCAL_MODELS["embedding"], device=self.device)
            self.use_embeddings = True
            print(f"Loaded embeddings model")
        except Exception as e:
            print(f"Using TF-IDF fallback")
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
        
        for k in range_k:
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
        print("Parsing dialogs into turns...")
        
        all_turns = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            dialog_text = row[dialog_col]
            turns = self.dialog_processor.parse_dialog(dialog_text)
            
            for turn in turns:
                turn['dialog_id'] = idx
                all_turns.append(turn)
        
        turns_df = pd.DataFrame(all_turns)
        print(f"Extracted {len(turns_df)} turns from {len(df)} dialogs")
        return turns_df

    def analyze_customer_sentiment(self, turns_df: pd.DataFrame) -> pd.DataFrame:
        print("Analyzing customer sentiment...")
        
        customer_turns = turns_df[turns_df['speaker'] == 'customer'].copy()
        
        results = []
        
        for idx, row in tqdm(customer_turns.iterrows(), total=len(customer_turns), desc="Sentiment"):
            try:
                lang = 'auto'
                if any(char in row['text'] for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                    lang = 'ru'
                elif any(char in row['text'] for char in 'әғқңөұүһі'):
                    lang = 'kk'
                
                res = self.sentiment_analyzer.analyze(row['text'], lang)
                rule_sentiment, rule_confidence = self.dialog_processor.analyze_customer_sentiment(row['text'])
                
                final_sentiment = res['sentiment']
                if rule_sentiment != 'neutral' and rule_confidence > res['confidence']:
                    final_sentiment = rule_sentiment
                
            except Exception as e:
                final_sentiment = 'neutral'
            
            results.append(final_sentiment)
        
        customer_turns['sentiment'] = results
        return customer_turns

    def cluster_complaints(self, customer_turns: pd.DataFrame) -> pd.DataFrame:
        print("Clustering complaint reasons...")
        
        complaint_turns = customer_turns[customer_turns['sentiment'] == 'negative'].copy()
        
        if len(complaint_turns) < 5:
            print("Not enough complaints for clustering")
            return complaint_turns
        
        for idx, row in complaint_turns.iterrows():
            complaint_type = self.dialog_processor.identify_complaint_type(row['text'])
            complaint_turns.loc[idx, 'complaint_type'] = complaint_type
        
        texts = complaint_turns['text'].tolist()
        features = self.vectorize(texts)
        
        clusters, n_clusters = self.cluster(features)
        complaint_turns['cluster'] = clusters
        
        print(f"Found {n_clusters} complaint clusters")
        
        return complaint_turns

    def visualize_results(self, customer_turns: pd.DataFrame, complaint_turns: pd.DataFrame):
        if len(customer_turns) < 10:
            print("Insufficient data for visualization")
            return
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(231)
        sentiment_counts = customer_turns['sentiment'].value_counts()
        colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'blue' for s in sentiment_counts.index]
        sentiment_counts.plot(kind='bar', color=colors)
        plt.title('Customer Sentiment Distribution')
        plt.xticks(rotation=45)
        
        if len(complaint_turns) > 0 and 'cluster' in complaint_turns.columns:
            features = self.vectorize(complaint_turns['text'].tolist())
            points = self.reduce_dimensions(features)
            
            plt.subplot(232)
            plt.scatter(points[:, 0], points[:, 1], c=complaint_turns['cluster'], cmap='tab20', alpha=0.7)
            plt.title('Complaint Clusters')
            plt.colorbar()
            
            plt.subplot(233)
            cluster_sizes = complaint_turns['cluster'].value_counts().sort_index()
            cluster_sizes.plot(kind='bar')
            plt.title('Complaint Cluster Sizes')
            
            plt.subplot(234)
            if 'complaint_type' in complaint_turns.columns:
                type_counts = complaint_turns['complaint_type'].value_counts()
                type_counts.plot(kind='bar')
                plt.title('Complaint Types')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    def generate_reports(self, customer_turns: pd.DataFrame, complaint_turns: pd.DataFrame, turns_df: pd.DataFrame) -> tuple:
        print("Generating reports...")
        
        sentiment_report = customer_turns['sentiment'].value_counts().to_dict()
        
        complaint_cluster_report = []
        if len(complaint_turns) > 0 and 'cluster' in complaint_turns.columns:
            for cluster_id in sorted(complaint_turns['cluster'].unique()):
                cluster_data = complaint_turns[complaint_turns['cluster'] == cluster_id]
                
                complaint_types = cluster_data.get('complaint_type', pd.Series(['general']*len(cluster_data))).value_counts()
                dominant_type = complaint_types.idxmax() if not complaint_types.empty else 'general'
                
                sample_texts = cluster_data['text'].head(3).tolist()
                
                complaint_cluster_report.append({
                    'cluster': int(cluster_id),
                    'size': int(len(cluster_data)),
                    'dominant_complaint_type': dominant_type,
                    'avg_length': float(cluster_data['length'].mean()),
                    'sample_texts': sample_texts
                })
        
        dialog_stats = []
        for dialog_id in turns_df['dialog_id'].unique():
            dialog_turns = turns_df[turns_df['dialog_id'] == dialog_id]
            customer_dialog_turns = dialog_turns[dialog_turns['speaker'] == 'customer']
            
            if len(customer_dialog_turns) > 0:
                customer_sentiment_turns = customer_turns[customer_turns['dialog_id'] == dialog_id]
                sentiments = customer_sentiment_turns['sentiment'].tolist() if len(customer_sentiment_turns) > 0 else ['neutral']
                
                dialog_stats.append({
                    'dialog_id': int(dialog_id),
                    'total_turns': int(len(dialog_turns)),
                    'customer_turns': int(len(customer_dialog_turns)),
                    'customer_sentiments': sentiments,
                    'has_negative': 'negative' in sentiments,
                    'has_positive': 'positive' in sentiments
                })
        
        return pd.DataFrame(complaint_cluster_report), pd.DataFrame(dialog_stats), sentiment_report

def analyze_call_center_dialogs(df: pd.DataFrame, LOCAL_MODELS: dict, dialog_col='dialog_text', 
                               stopwords_file='Stop Words List - Kazakh.txt'):
    if df.empty:
        print("Empty dataframe")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df = df.copy()
    df[dialog_col] = df[dialog_col].fillna('').astype(str)
    
    print("Initializing analyzer...")
    analyzer = ComplaintClusterAnalyzer(LOCAL_MODELS, stopwords_file=stopwords_file)
    
    turns_df = analyzer.process_dialogs(df, dialog_col)
    
    if not turns_df.empty:
        customer_turns = analyzer.analyze_customer_sentiment(turns_df)
        complaint_turns = analyzer.cluster_complaints(customer_turns)
        
        analyzer.visualize_results(customer_turns, complaint_turns)
        
        cluster_report, dialog_report, sentiment_summary = analyzer.generate_reports(customer_turns, complaint_turns, turns_df)
        
        print(f"\nSentiment Summary: {sentiment_summary}")
        print(f"Total dialogs: {len(dialog_report)}")
        print(f"Dialogs with complaints: {dialog_report['has_negative'].sum()}")
        print(f"Complaint clusters found: {len(cluster_report)}")
        
        return df, turns_df, customer_turns, complaint_turns
    
    else:
        print("No turns extracted")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()