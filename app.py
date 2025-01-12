from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.parse
import os

# Initialize Flask app
app = Flask(__name__)

# Load your model (assuming you've trained the model as in your previous code)
def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.readlines()
    except OSError as e:
        print(f"Error opening file {filepath}: {e}")
        return []
    
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   # Converting URL-encoded data to simple string
        result.append(d)
    return result

# Load queries
badQueries = loadFile('badqueries.txt')
validQueries = loadFile('goodqueries.txt')
allQueries = badQueries + validQueries

# Labels: 1 for malicious, 0 for clean
yBad = [1 for _ in range(len(badQueries))]
yGood = [0 for _ in range(len(validQueries))]
y = yBad + yGood
queries = allQueries

# Convert data to vectors using TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(queries)

# Train Logistic Regression model with class weights to address imbalance
lgs = LogisticRegression(class_weight={1: 2 * len(validQueries) / len(badQueries), 0: 1.0})
lgs.fit(X, y)

# Define function to classify query
def classify_query(query):
    query_vectorized = vectorizer.transform([query])
    prediction = lgs.predict(query_vectorized)
    
    if prediction == 1:
        return "Malicious (Bad Query)"
    else:
        return "Valid (Good Query)"

# Home route to show the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to process the query
@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    result = classify_query(query)
    return render_template('index.html', query=query, result=result)

if __name__ == '__main__':
    app.run(debug=True)
