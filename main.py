import sys
import time
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

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

l = {}
i1 = 0


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
      "\n 2 : random forests")
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

    '''Buildin Decision Tree'''
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    '''ploting tree'''
    tree.plot_tree(dtree, feature_names=features)
    plt.figure(figsize=(15, 10))  # Adjust the figure size if needed
    plot_tree(dtree, feature_names=features, filled=True)
    plt.show()  # showing during the code is running
    plt.savefig('decision_tree_plot.png')  # Saves the plot as a PNG file
    sys.stdout.flush()

    '''predict price using other parameters and trained data'''
    new_data = pd.DataFrame([[2, 1, 2, 1, 0]], columns=features)
    predictions = dtree.predict(new_data)
    print(predictions)


elif alg_tree == "2":
    print("doing random forests")
else:
    print("unknown command")