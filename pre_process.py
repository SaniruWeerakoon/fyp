import pandas as pd
import numpy as np
import json
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from sentence_transformers import SentenceTransformer, util

# Load trend scores
with open("./files/combined_trends_by_date.json", "r") as f:
    trend_scores = json.load(f)

# Sentiment model
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# Topic model
topic_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", device=device, local_files_only=True
)
candidate_labels = [
    "climate change",
    "stock market trends",
    "mental health",
    "entrepreneurship",
    "machine learning",
    "remote work",
    "personal finance",
    "movie recommendations",
    "mental health awareness",
    "recipe ideas",
    "budget travel tips",
    "video game reviews",
    "book recommendations",
    "gen z",
    "trump",
    "democrat",
    "parenting advice",
    "fitness journey",
    "pet care tips",
    "mobile app development",
    "self improvement",
    "elon musk",
    "tesla",
    "gaming",
    "crypto",
    "freedom of speech",
    "mass surveillance",
    "education reform",
    "internet censorship",
    "affordable housing",
    "job interview tips",
    "work life balance",
    "resume tips",
    "learning to code",
    "netflix recommendations",
    "wedding planning",
    "instagram marketing",
    "passive income",
    "financial independence",
    "ukraine updates",
]
topic_embeddings = topic_model.encode(candidate_labels, convert_to_tensor=True)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


def classify_topic(text):
    tweet_embedding = topic_model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(tweet_embedding, topic_embeddings).squeeze()

    best_idx = similarities.argmax().item()
    best_label = candidate_labels[best_idx]
    best_score = similarities[best_idx].item()

    return best_label, best_score


def enrich_topic_with_embeddings(row):
    try:
        text = row["cleaned_text"]
        best_topic, topic_score = classify_topic(text)
        created_date = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d")
        trend_value = trend_scores.get(created_date, {}).get(best_topic, 0)

        return pd.Series(
            {
                "top_topic": best_topic,
                "topic_score": topic_score,
                "trend_score": trend_value,
            }
        )

    except Exception as e:
        print(f"Error processing row {row['id']}: {e}")
        return pd.Series({"top_topic": "skipped", "topic_score": 0, "trend_score": 0})


def add_sentiment(row):
    try:
        text = row["cleaned_text"]
        # Get sentiment
        sentiment = sentiment_pipeline(text)[0]

        return pd.Series(
            {
                "sentiment": sentiment["label"],
                "sentiment_score": sentiment["score"],
            }
        )
    except Exception as e:
        # Skip row safely
        print(f"Error processing row {row['id']}: {e}")
        return pd.Series(
            {
                "sentiment": "skipped",
                "sentiment_score": 0,
            }
        )


def select_sentiment(row):
    threshold = 0.4
    score = row["sentiment_score"]
    label = row["sentiment"]

    if score >= threshold:
        if label == "POSITIVE":
            return pd.Series({"sentiment_positive": score, "sentiment_negative": 0})
        elif label == "NEGATIVE":
            return pd.Series({"sentiment_positive": 0, "sentiment_negative": score})

    return pd.Series({"sentiment_positive": 0, "sentiment_negative": 0})


def classify_best_hashtag(hashtags):
    if not hashtags:
        return "none", 0.0

    best_score = 0.0
    best_label = "none"

    for tag in hashtags:
        tag_text = tag.replace("#", "").replace("_", " ").lower()
        tag_emb = topic_model.encode(tag_text, convert_to_tensor=True)
        sim_scores = util.cos_sim(tag_emb, topic_embeddings).squeeze()

        max_idx = sim_scores.argmax().item()
        max_score = sim_scores[max_idx].item()

        if max_score > best_score:
            best_score = max_score
            best_label = candidate_labels[max_idx]

    return best_label, best_score


def enrich_hashtag_score(row):
    try:
        hashtags = row["hashtags"]
        trending_hashtag, hashtag_score = classify_best_hashtag(hashtags)

        created_date = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d")
        trend_score = trend_scores.get(created_date, {}).get(trending_hashtag, 0)

        return pd.Series(
            {
                "trending_hashtag": trending_hashtag,
                "hashtag_score": hashtag_score,
                "hashtag_trend_score": trend_score,
            }
        )
    except Exception as e:
        print(f"Error processing row {row['id']}: {e}")
        return pd.Series(
            {
                "trending_hashtag": "skipped",
                "hashtag_score": 0,
                "hashtag_trend_score": 0,
            }
        )


def preprocess_input(data: dict):
    required_keys = [
        "tweet_text",
        "timestamp",
        "user_followers",
        "user_account_age_days",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required input keys: {missing}")

    df = pd.DataFrame([data])
    df["id"] = df.index
    df["cleaned_text"] = df["tweet_text"].apply(clean_text)
    df["text_length"] = df["cleaned_text"].str.len()

    df["date"] = pd.to_datetime(df["timestamp"], utc=True)
    df["created_at"] = df["date"]
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday
    df["is_peak_hour"] = df["hour"].between(18, 21).astype(int)

    # Sentiment
    sentiment_columns = df.apply(add_sentiment, axis=1)
    df = pd.concat([df, sentiment_columns], axis=1)
    df[["sentiment_positive", "sentiment_negative"]] = df.apply(
        select_sentiment, axis=1
    )

    # Topic + Trend
    topic_columns = df.apply(enrich_topic_with_embeddings, axis=1)
    df = pd.concat([df, topic_columns], axis=1)

    # Hashtag topic + trend
    hashtag_columns = df.apply(enrich_hashtag_score, axis=1)
    df = pd.concat([df, hashtag_columns], axis=1)

    # Additional numeric
    df["followers"] = data.get("user_followers", 0)
    df["account_age_days"] = data.get("user_account_age_days", 0)
    df["log_followers"] = np.log1p(df["followers"])
    df["log_account_age_days"] = np.log1p(df["account_age_days"])
    df["hashtag_count"] = len(data.get("hashtags", []))
    df["blue"] = int(data.get("user_blue", False))

    # One-hot hour/weekday
    df = pd.get_dummies(
        df, columns=["hour", "weekday"], prefix=["hour", "weekday"]
    )

    # Ensure all expected features exist
    required = [
        "text_length",
        "is_peak_hour",
        "blue",
        "trend_score",
        "hashtag_trend_score",
        "log_followers",
        "log_account_age_days",
        "sentiment_positive",
        "sentiment_negative",
        "hashtag_count",
    ]
    required += [f"hour_{i}" for i in range(1, 24)]  # Skip hour_0
    required += [f"weekday_{i}" for i in range(1, 7)]  # Skip weekday_0

    for col in required:
        if col not in df:
            print(f"Warning: Missing required column '{col}', initializing to 0")
            df[col] = 0
            
    
    return df[required]
