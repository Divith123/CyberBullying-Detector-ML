import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

# Download NLTK resources
nltk.download('stopwords')

# Load datasets
aggression_data = pd.read_csv("datasets/aggression_parsed_dataset.csv")
attack_data = pd.read_csv("datasets/attack_parsed_dataset.csv")
kaggle_data = pd.read_csv("datasets/kaggle_parsed_dataset.csv")
racism_data = pd.read_csv("datasets/twitter_racism_parsed_dataset.csv")
sexism_data = pd.read_csv("datasets/twitter_sexism_parsed_dataset.csv")
youtube_data = pd.read_csv("datasets/youtube_parsed_dataset.csv")

# Rename columns to match (if necessary)
aggression_data = aggression_data.rename(columns={"Text": "text", "oh_label": "label"})
attack_data = attack_data.rename(columns={"Text": "text", "oh_label": "label"})
kaggle_data = kaggle_data.rename(columns={"Text": "text", "oh_label": "label"})
racism_data = racism_data.rename(columns={"Text": "text", "oh_label": "label"})
sexism_data = sexism_data.rename(columns={"Text": "text", "oh_label": "label"})
youtube_data = youtube_data.rename(columns={"Text": "text", "oh_label": "label"})

# Combine datasets
combined_data = pd.concat([
    aggression_data[["text", "label"]],
    attack_data[["text", "label"]],
    kaggle_data[["text", "label"]],
    racism_data[["text", "label"]],
    sexism_data[["text", "label"]],
    youtube_data[["text", "label"]]
], ignore_index=True)

# Drop rows with missing or invalid text
combined_data = combined_data.dropna(subset=["text"])  # Drop rows where 'text' is NaN
combined_data = combined_data[combined_data["text"].apply(lambda x: isinstance(x, str))]  # Keep only string values

# Text preprocessing function
def preprocess_text(text):
    try:
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)
        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        
        # Apply stemming with error handling
        stemmer = PorterStemmer()
        stemmed_tokens = []
        for word in tokens:
            try:
                stemmed_tokens.append(stemmer.stem(word))
            except RecursionError:
                # Skip problematic words
                continue
        
        return " ".join(stemmed_tokens)
    except Exception as e:
        # Log errors and return empty string for problematic rows
        print(f"Error processing text: {e}", file=sys.stderr)
        return ""

# Apply preprocessing
combined_data["cleaned_text"] = combined_data["text"].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
combined_data["label"] = label_encoder.fit_transform(combined_data["label"])

# Save combined dataset
combined_data.to_csv("datasets/combined_dataset.csv", index=False)
print("Combined dataset saved successfully!")