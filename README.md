# Sentiment Analysis Web Application

A web application that performs sentiment analysis on text input, classifying it as positive, neutral, or negative. This project was built as part of a portfolio to demonstrate machine learning and web development skills.

## Features

- Real-time sentiment analysis of user-provided text
- Classification into Positive, Neutral, or Negative sentiment
- Confidence scores for each sentiment category
- Responsive, modern UI built with Bootstrap
- Pre-trained machine learning model

## Technologies Used

- **Backend**: Flask, scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **ML Model**: Logistic Regression with TF-IDF vectorization
- **Dataset**: Amazon Electronics Reviews dataset

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/AashishChauhan0207/Sentiment-Analysis-Web-App.git
   cd sentiment-analysis-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## How It Works

1. The application preprocesses text using NLP techniques (tokenization, removing stopwords, lemmatization)
2. Text is vectorized using TF-IDF
3. A pre-trained Logistic Regression model predicts the sentiment
4. Results are displayed with confidence scores for each category

## Usage in Portfolio

This project demonstrates skills in:
- Machine Learning and Natural Language Processing
- Full-stack web development
- User interface design
- Data processing and model deployment

## Future Improvements

- Add support for more languages
- Implement a more sophisticated model (BERT, RoBERTa)
- Add user feedback mechanism to improve the model
- Create an API endpoint for integration with other services 
