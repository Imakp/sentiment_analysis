from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models and vectorizer
vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
svm_model = joblib.load('saved_models/svm_model.pkl')
random_forest_model = joblib.load('saved_models/random_forest_model.pkl')
naive_bayes_model = joblib.load('saved_models/naive_bayes_model.pkl')
logistic_regression_model = joblib.load('saved_models/logistic_regression_model.pkl')
gradient_boosting_model = joblib.load('saved_models/gradient_boosting_model.pkl')
ensemble_model = joblib.load('saved_models/ensemble_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        
        # Transform the review using the vectorizer
        review_tfidf = vectorizer.transform([review])
        print("Transformed review:", review_tfidf)
        
        # Predict the sentiment using all models
        svm_sentiment = svm_model.predict(review_tfidf)
        random_forest_sentiment = random_forest_model.predict(review_tfidf)
        naive_bayes_sentiment = naive_bayes_model.predict(review_tfidf)
        logistic_regression_sentiment = logistic_regression_model.predict(review_tfidf)
        gradient_boosting_sentiment = gradient_boosting_model.predict(review_tfidf)
        ensemble_sentiment = ensemble_model.predict(review_tfidf)
        
        # Determine the sentiment label for each model
        sentiments = {
            'SVM': 'Positive' if svm_sentiment == 1 else 'Negative',
            'Random Forest': 'Positive' if random_forest_sentiment == 1 else 'Negative',
            'Naive Bayes': 'Positive' if naive_bayes_sentiment == 1 else 'Negative',
            'Logistic Regression': 'Positive' if logistic_regression_sentiment == 1 else 'Negative',
            'Gradient Boosting': 'Positive' if gradient_boosting_sentiment == 1 else 'Negative',
            'Ensemble': 'Positive' if ensemble_sentiment == 1 else 'Negative',
        }
        
        return render_template('index.html', review=review, sentiments=sentiments)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
