import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Randomforest(data):
    # Assuming 'features' contains your independent variables and 'target' contains the target variable
    # Adjust this according to your dataset's columns
    features = data.drop('target_column', axis=1)
    target = data['target_column']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = forest_classifier.predict(X_test)

    # Calculate accuracy without reject option
    accuracy_without_reject = accuracy_score(y_test, y_pred)
    print(f'Accuracy without reject option: {accuracy_without_reject}')

    # Implementing a rejection option by considering prediction probabilities
    # You can adjust the threshold according to your confidence level for the rejection
    threshold = 0.7
    y_prob = forest_classifier.predict_proba(X_test)
    y_pred_with_reject = []

    for probs in y_prob:
        if max(probs) < threshold:
            y_pred_with_reject.append('Reject')  # Assign 'Reject' class for uncertain predictions
        else:
            y_pred_with_reject.append(forest_classifier.classes_[probs.argmax()])

    # Calculate accuracy with reject option
    accuracy_with_reject = accuracy_score(y_test, y_pred_with_reject)
    print(f'Accuracy with reject option: {accuracy_with_reject}')