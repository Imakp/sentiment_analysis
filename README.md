# Sentiment Analysis Project

This project focuses on sentiment analysis of movie reviews using various machine learning models. It includes data preprocessing, model training, hyperparameter tuning, and a web interface for users to input their reviews and receive sentiment predictions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Usage](#usage)
- [Web Application](#web-application)
- [Results](#results)

## Overview

The goal of this project is to classify movie reviews as positive or negative. Various models like Naive Bayes, Support Vector Machine (SVM), Random Forest, Logistic Regression, and Gradient Boosting are trained and evaluated. The best models are saved, and an ensemble model is created for better performance.

## Dataset

The dataset used in this project is a subset of the IMDB movie reviews dataset, which includes both positive and negative reviews. The data is cleaned, tokenized, lemmatized, and vectorized using methods like Bag of Words, TF-IDF, and Word2Vec.

dataset: [IMDB Dataset of Movie Review](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sentiment-analysis.git
    cd sentiment-analysis
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK data:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

## Project Structure

The project is organized as follows:

sentiment_analysis/
│
├── saved_models/           # Directory containing the trained machine learning models
├── IMDB.csv                # Dataset file with movie reviews for sentiment analysis
├── sentiment_analysis.ipynb # Jupyter Notebook with the complete sentiment analysis code
├── README.md               # Project overview and instructions
├── app.py                  # Flask application script for web interface
├── templates/              # Directory containing HTML templates for the web app
└── static/                 # Directory containing static files like CSS for the web app

## Models Used

The following models were trained and evaluated:

- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**

Additionally, a Voting Classifier (Ensemble Model) was created by combining the predictions of the above models.

## Usage

### Data Preprocessing

The dataset is preprocessed using techniques like:
- **Text cleaning**: Removing HTML tags, punctuation, and stopwords.
- **Tokenization**: Splitting text into individual words.
- **Lemmatization**: Reducing words to their base forms.
- **Vectorization**: Converting text into numerical features using Bag of Words, TF-IDF, and Word2Vec.

### Model Evaluation

The performance of each model is assessed on the test set using metrics such as accuracy, precision, recall, and F1-score.

## Web Application

A Flask web app is available for users to input their reviews and view the sentiment predictions from each model.

### Running the Web App

To run the web application:

1. Navigate to the project directory.
2. Start the Flask app by running the following command:

   ```bash
   python app.py

3. Open a browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
4. Enter a review in the text box and submit it to see the sentiment predictions.

## Results

The project includes the following visualizations:

- **Word Cloud:** Displays common words in the dataset.
- **Bar Plot:** Shows the most frequent words.
- **Confusion Matrices:** Visualizes model performance.

The ensemble model achieved the highest accuracy and is used as the default model in the web application.
