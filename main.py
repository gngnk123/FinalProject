import sys
import time
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset into Pandas DataFrame
df = pd.read_csv("laptops.csv")

# iloc selects by position, ':' selects all rows, and number selects the column
first_column = df.iloc[:, 0]
second_column = df.iloc[:, 1]
third_column = df.iloc[:, 2]
forth_column = df.iloc[:, 3]
fifth_column = df.iloc[:, 4]
sixth_column = df.iloc[:, 5]

'''This function replaces the string value with an integer for the decision tree'''

def Average(lst):
    return sum(lst) / len(lst)


avarageprc = (Average(sixth_column))


def my_function(n_column, changearg):  # n_column is list of provided column values and changearg is the column names
    unique_strings = set(n_column)
    all_unique_str = []  # Initialize a list to store all unique strings
    ''' Iterating through a range of numbers to print unique elements '''
    for i in range(len(unique_strings)):
        unique_str = unique_strings.pop()
        all_unique_str.append(unique_str)

    # Printing all the unique strings after the loop
    # print(all_unique_str)

    i1 = 0  # increment to move between elements in list
    l = {}  # Dictionary to store replacing values
    for i in all_unique_str:  # going through all unique elements and setting numeric values for all elements in data
        if changearg == 'Price':  # Price has to be filtered by more than average and less
            if i >= avarageprc:
                l[i] = 0
            else:
                l[i] = 1
        else:
            l[i] = i1  # for example {Asus: 0}
            i1 += 1

    df[changearg] = df[changearg].map(l)  # Replace strings with numbers
    # print(l)


print("Chose the algorithm you want to use for this dataset:"
      "\n 1 : decision trees"
      "\n 2 : random forests"
      "\n type 1 or 2 :")
alg_tree = input()

if alg_tree == "1":
    print("doing decision tree")
    time.sleep(1)  # to clarify output visualisation

    '''Calling a functions for all columns'''
    my_function(first_column, 'CompanyName')
    my_function(second_column, 'Cpu')
    my_function(third_column, 'Ram')
    my_function(forth_column, 'Memory')
    my_function(fifth_column, 'Gpu')
    my_function(sixth_column, 'Price')

    print(df)  # print dataset with numeric values
    time.sleep(1)
    features = ['CompanyName', 'Cpu', 'Ram', 'Memory', 'Gpu']

    X = df[features]
    y = df['Price']  # target column

    '''plotting the tree'''
    # Pre-pruning: Limit the depth of the decision tree
    max_depth = 3
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree = dtree.fit(X, y)

    # Get probability estimates for each observation
    probabilities = dtree.predict_proba(X)

    # Set an ambiguity rejection threshold
    ambiguity_threshold = 0.7

    # Identify indices where the maximum probability is below the threshold
    ambiguous_indices = [i for i, prob in enumerate(probabilities.max(axis=1)) if prob < ambiguity_threshold]

    # Reject predictions for ambiguous instances
    y_pred = dtree.predict_proba(X)
    y_pred[ambiguous_indices] = None  # Use a special value (e.g., None) to indicate rejection

    # Plotting the pruned tree
    plt.figure(figsize=(15, 10))
    plot_tree(dtree, feature_names=features, filled=True, rounded=True, class_names=['below_avg', 'above_avg'])
    plt.savefig('pruned_decision_tree_plot.png')
    plt.show()
    # Display predictions with rejection
    print("Predictions with Ambiguity Rejection:")
    #  print(y_pred)

    '''predict price using other parameters and trained data'''
    new_data = pd.DataFrame([[1, 2, 2, 0, 0]], columns=features)
    print(new_data)
    predictions = dtree.predict(new_data)
    print('Price {}'.format(predictions))

    # random forest
elif alg_tree == "2":
    print("doing random forests")
    time.sleep(1)  # to clarify output visualisation

    '''Calling a functions for all columns'''
    my_function(first_column, 'CompanyName')
    my_function(second_column, 'Cpu')
    my_function(third_column, 'Ram')
    my_function(forth_column, 'Memory')
    my_function(fifth_column, 'Gpu')
    my_function(sixth_column, 'Price')

    print(df)  # print dataset with numeric values
    time.sleep(1)
    features = ['CompanyName', 'Cpu', 'Ram', 'Memory', 'Gpu']

    X = df[features]
    y = df['Price']  # target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    classifier = RandomForestClassifier(n_estimators=50)
    classifier.fit(X_test, y_test)

    '''predict price using other parameters and trained data'''

    y_pred = classifier.predict(X_test)
    print(y_pred)
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:", )
    print(result1)
    result2 = accuracy_score(y_test, y_pred)
    print("Accuracy:", result2)
    new_data = pd.DataFrame([[1, 2, 2, 0, 0]], columns=features)


    # Get probability estimates for each class
    probabilities = classifier.predict_proba(new_data)

    # Set an ambiguity rejection threshold (adjust as needed)
    ambiguity_threshold = 0.7

    # Identify indices where the maximum probability is below the threshold
    ambiguous_indices = [i for i, prob in enumerate(probabilities.max(axis=1)) if prob < ambiguity_threshold]

    # Reject predictions for ambiguous instances
    y_pred_new = classifier.predict(new_data)
    y_pred_new[ambiguous_indices] = -1  # Use a placeholder value to indicate rejection

    print("\nNew Data:")
    print(new_data)
    print('Price:', y_pred_new)
else:
    print("unknown command")
