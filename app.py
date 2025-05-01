from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NLTK data path to a writable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK data
try:
    for resource in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' already downloaded")
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"NLTK resource '{resource}' downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

app = Flask(__name__)

# Text preprocessing function
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logger.error(f"Error loading stopwords: {e}")
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as'])

@lru_cache(maxsize=128)
def preprocess_text(text):
    try:
        # Lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        # Return original text with basic cleaning as fallback
        return re.sub(r'[^a-zA-Z\s]', '', text.lower())

# Load model and vectorizer
model_path = 'sentiment_model.pkl'
vectorizer_path = 'vectorizer.pkl'

# Force retraining in production environment (Render)
if os.environ.get('RENDER') or not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    logger.info("Training new model on deployment")
    
    # Create a more comprehensive sample dataset with balanced classes
    sample_data = {
        'reviewText': [
            # Positive examples
            "This product is amazing! I love it.",
            "Great product, works as expected.",
            "Exceeded my expectations, highly recommend!",
            "Perfect solution for my needs, very satisfied.",
            "Works flawlessly, best purchase I've made all year.",
            "The quality is outstanding, definitely worth the price.",
            "Incredible performance, couldn't be happier.",
            "Exactly what I was looking for, excellent quality.",
            "This is a fantastic product, I use it every day.",
            "Superior to all competing products on the market.",
            "The customer service was excellent and the product works great.",
            "I would recommend this to all my friends and family.",
            "This product changed my life, so convenient.",
            "Beautifully designed and works perfectly.",
            "Amazing value for the price, extremely satisfied.",
            "I'm impressed with how well this works.",
            "Five stars! Absolutely love this product.",
            "Works better than advertised, very pleased.",
            "Exceptional quality and performance.",
            "Reliable, efficient, and worth every penny.",
            
            # Neutral examples
            "It was okay, nothing special.",
            "The product is decent for the price.",
            "Average performance, nothing extraordinary.",
            "It works, but I've seen better.",
            "Not bad, not great, just average.",
            "Serves its purpose, nothing more.",
            "Functional but lacks some features.",
            "It's fine I guess, does what it's supposed to.",
            "Mediocre quality but gets the job done.",
            "I have mixed feelings about this product.",
            "Some good features, some disappointing ones.",
            "Moderately satisfied with this purchase.",
            "It's alright, wouldn't rave about it though.",
            "Not terrible but not impressive either.",
            "Basic functionality works as expected.",
            "It's adequate for occasional use.",
            "I neither love it nor hate it.",
            "Does what it claims, nothing more, nothing less.",
            "Acceptable quality for the price point.",
            "It's a middle-of-the-road product.",
            
            # Negative examples
            "Not worth the money, very disappointed.",
            "Terrible experience, avoid at all costs.",
            "Broke after a week, poor quality.",
            "Doesn't work as advertised, complete waste.",
            "Poor design, difficult to use.",
            "The worst product I've ever purchased.",
            "Save your money, this is junk.",
            "Completely useless, do not buy.",
            "Fell apart after minimal use.",
            "False advertising, product is nothing like described.",
            "I regret this purchase immensely.",
            "It stopped working after two days.",
            "Extremely disappointed with this product.",
            "The quality is subpar at best.",
            "Would give zero stars if I could.",
            "Don't waste your time or money on this.",
            "Frustrating to use and poorly made.",
            "Doesn't perform any of the functions properly.",
            "Cheaply made and breaks easily.",
            "This product is a complete scam."
        ],
        'overall': [
            # Ratings for positive examples
            5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 4,
            # Ratings for neutral examples
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            # Ratings for negative examples
            1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1
        ]
    }
    df = pd.DataFrame(sample_data)
    
    # Select relevant columns
    df = df[['reviewText', 'overall']]
    df.dropna(inplace=True)
    
    # Convert ratings to sentiment labels
    def get_sentiment_label(rating):
        if rating <= 2:
            return "Negative"  # 1 or 2 stars
        elif rating == 3:
            return "Neutral"   # 3 stars
        else:
            return "Positive"  # 4 or 5 stars
    
    df['sentiment'] = df['overall'].apply(get_sentiment_label)
    logger.info(f"Class distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Preprocess text
    df['cleaned_text'] = df['reviewText'].apply(preprocess_text)
    
    # Split data
    X = df['cleaned_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize text with better parameters
    vectorizer = TfidfVectorizer(max_features=3000, 
                               min_df=2, 
                               ngram_range=(1, 2),
                               sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model with optimized parameters
    model = LogisticRegression(multi_class='multinomial', 
                             solver='lbfgs', 
                             max_iter=500,
                             C=1.5,
                             class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    # Evaluate on test set
    X_test_vec = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vec, y_test)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    
    # Save model and vectorizer
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info("Model and vectorizer saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    # Clear memory
    del df, X, y, X_train, X_test, y_train, y_test, X_train_vec, X_test_vec
else:
    # Load existing model and vectorizer
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("Existing model and vectorizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
        # If loading fails, train a simple fallback model
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        # Simple fallback data
        fallback_data = {
            'text': ["great", "excellent", "amazing", "bad", "terrible", "awful", "okay", "fine", "average"],
            'sentiment': ["Positive", "Positive", "Positive", "Negative", "Negative", "Negative", "Neutral", "Neutral", "Neutral"]
        }
        fallback_df = pd.DataFrame(fallback_data)
        
        # Simple vectorizer and model
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(fallback_df['text'])
        model = LogisticRegression()
        model.fit(X_vec, fallback_df['sentiment'])
        logger.info("Fallback model trained due to loading error")
        
        # Clear memory
        del fallback_df, X_vec

# Prediction cache
prediction_cache = {}

# Prediction function
def predict_sentiment(review):
    try:
        # Check cache first
        if review in prediction_cache:
            logger.info(f"Cache hit for text: '{review[:50]}...'")
            return prediction_cache[review]
        
        processed_review = preprocess_text(review)
        vectorized_review = vectorizer.transform([processed_review])
        prediction = model.predict(vectorized_review)[0]
        
        # Get confidence scores
        probabilities = model.predict_proba(vectorized_review)[0]
        sentiment_classes = model.classes_
        confidence_scores = {sentiment: round(prob * 100, 2) for sentiment, prob in zip(sentiment_classes, probabilities)}
        
        logger.info(f"Prediction for text: '{review[:50]}...' - {prediction} with confidence: {confidence_scores[prediction]}")
        
        result = {
            'sentiment': prediction,
            'confidence': confidence_scores
        }
        
        # Store in cache (limit cache size to 1000 entries)
        if len(prediction_cache) < 1000:
            prediction_cache[review] = result
        
        return result
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            'sentiment': 'Error',
            'confidence': {'Error': 100, 'Positive': 0, 'Neutral': 0, 'Negative': 0}
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        try:
            data = request.json
            review_text = data.get('text', '')
            
            if not review_text:
                return jsonify({'error': 'No text provided'})
            
            result = predict_sentiment(review_text)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in /analyze endpoint: {e}")
            return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 