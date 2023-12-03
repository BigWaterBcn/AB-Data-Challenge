from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import joblib

# Load the dataset
file_path = './consumption_clusters.csv'
data = pd.read_csv(file_path)

# Separate features and labels
X = data.drop(['ID', 'Cluster'], axis=1)  # Features: hourly consumption data
y = data['Cluster']  # Labels: cluster assignments

# Split the dataset into training and testing sets (if needed for initial evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average score: {scores.mean()}")

# Evaluate the model on the test set (if you've split your dataset)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Assuming 'classifier' is your trained RandomForestClassifier model
joblib.dump(classifier, './classifier.joblib')

# The model is now trained and can be used to predict the cluster of a new household