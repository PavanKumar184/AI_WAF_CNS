from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split  # Updated import
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
import matplotlib.pyplot as plt

def loadFile(name):
    directory = str(os.getcwd())
    # Use raw string to avoid issues with backslashes in Windows paths
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

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries

# Labels: 1 for malicious, 0 for clean
yBad = [1 for _ in range(len(badQueries))]
yGood = [0 for _ in range(len(validQueries))]
y = yBad + yGood
queries = allQueries

# Convert data to vectors
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(queries)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Count of bad and valid queries
badCount = len(badQueries)
validCount = len(validQueries)

# Train Logistic Regression model with class weights for imbalance
lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0})
lgs.fit(X_train, y_train)

# Evaluation
predicted = lgs.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

# Print metrics
print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + badCount)))
print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  # Accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)
