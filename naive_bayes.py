# Author: Kyle Arick Kassen

# Import statements
import pandas as pd
import numpy as np


print('Retrieving Data...')
# Terms data
# Let V be the vocabulary of all words in the documents D
file = open('newsgroups5-terms.txt', 'r')
V = [line.strip() for line in file]
file.close()

# Training data
# Putting the Training data into a dataframe structure
trainData = pd.read_csv('newsgroups5-train.csv', header=None)
trainData.columns = V
file = open('newsgroups5-train-labels.csv', 'r')
trainLabels = [line.strip() for line in file]
trainData.index = ['D' + str(i+1) for i in range(len(trainData))]
trainData.index.name = 'Document No.'
trainData.insert(len(V), column='Category', value=trainLabels[1:])
file.close()


# Simple function that allows the user to view the Training Data Table in the output terminal
def trainDataTable():
    print()
    print('Training Data Table -->')
    return trainData

# Testing data
# Putting the Testing data into a dataframe structure
testData = pd.read_csv('newsgroups5-test.csv', header=None)
testData.columns = V
file = open('newsgroups5-test-labels.csv')
testLabels = [line.strip() for line in file]
testData.index = ['D' + str(i+1) for i in range(len(testData))]
testData.index.name = 'Document No.'
testData.insert(len(V), column='Category', value=testLabels[1:])
file.close()

# Simple function that allows the user to view the Testing Data Table in the output terminal
def testingDataTable():
    print()
    print('Testing Data Table -->')
    return testData

# Naive Bayes Algorithm: Training Component
def NB_Train():
    # For each category c_i in C
    uniques = trainData['Category'].nunique()
    # A Dictionary that holds information about our categories (class labels)...
    # ...including the numerical category label, category count, and category prior probability
    c_dict = {}
    for i in range(uniques):
        C = trainData['Category'].value_counts().index, trainData['Category'].value_counts().values, \
            trainData['Category'].value_counts().values/len(trainData)
        c_dict[C[0][i]] = C[1][i], C[2][i]

    # Splitting the c_dict to view these structures separately
    categories = C[0]
    # Let D_i be the subset of documents in D in Category c_i
    D_i = C[1]
    # Priors: P(C_i) = |D_i|/|D|
    priors = C[2]

    # Dictionary Data Structure: {keys = category numbers: values = (Index [term objects]), (array [term counts])}
    dict_ = {}
    for i in range(len(C[0])):
        y = trainData[trainData['Category'] == C[0][i]]
        y = y.iloc[:, :-1]
        y = y.sum(axis=0).index, y.sum(axis=0).values
        dict_[C[0][i]] = y

    # Posteriors: P(w_i | c_i) = (n_ij + 1) / (n_i + |V|)
    posteriors = [[] for _ in range(len(C[0]))]
    for i in range(len(C[0])):
        for j in range(len(dict_[str(i)][1])):
            # LaPlace Smoothing
            n_ij = dict_[str(i)][1][j] + 1
            n_i = np.sum(dict_[str(i)][1]) + len(V)
            p = n_ij/n_i
            posteriors[i].append(p)
    return c_dict, posteriors, dict_, priors

print('Training in progress...')
NB_Train()
# Bringing the results returned by our NB_Train() function into the outer scope...
# ...for use in the remainder of the program (More specifically, the NB_Classify() function)
c_dict = NB_Train()[0]
posteriors = NB_Train()[1]
dict_ = NB_Train()[2]
priors = NB_Train()[3]

# Naive Bayes Algorithm: Testing Component
# Uses the posterior and prior probabilities returned returned by our NB_Train() function...
# ...to classify a test instance that was not used for training
def NB_Classify(X):
    # Suppressing a numpy warning
    np.seterr(divide='ignore')
    # Let n be the number of word occurrences in X
    n = 0
    # Dropping the 0s to reduce computation
    invertedIndexDict = {}
    for i in range(len(X)-1):
        if X[i] > 0:
            n += X[i]
            invertedIndexDict[i] = X[i]
    logProbs = []
    categoryInfo = []
    # Loop through our categories
    for c in c_dict.keys():
        # Adding the prior probability for category (c)
        c_prob = [c_dict[c][1]]
        # Loop through our 'inverted index' dictionary
        for term in invertedIndexDict.keys():
            p_ac = posteriors[int(c)][term]
            p_ac = p_ac ** invertedIndexDict[term]
            c_prob.append(p_ac)
        z = c, np.sum(np.log(c_prob))
        categoryInfo.append(z)
        # Sum of Log Probabilities
        logProbs.append(np.sum(np.log(c_prob)))
    sumLogProbs = max(logProbs)
    for i in range(len(categoryInfo)):
        if sumLogProbs == categoryInfo[i][1]:
            category = categoryInfo[i][0]
    return category, sumLogProbs

# Evaluation Function: measures accuracy of the trained classifier on the test data
def evaluation():
    accurate = 0
    wrong = 0
    actuals = testLabels[1:]
    for i in range(len(testData)):
        X = testData.iloc[i]
        X = X.values.tolist()
        if NB_Classify(X)[0] == actuals[i]:
            accurate += 1
        else:
            wrong += 1
    return accurate/(accurate+wrong)

# A simple function that returns Predicted Vs. Actual class labels and log probability values...
# ...for a range of document instances. Note: the range is specified by the user; captured...
# ...from the simple text interface
def docInstances(x ,y):
    for i in range(x, y):
        X = testData.iloc[i]
        X = X.values.tolist()
        print('Test Item ' + str(i) + ' Predicted Class: ' + str(NB_Classify(X)[0]) + ' Actual Class: '
              + testData['Category'][i] + ' Log Prob: ' + str(NB_Classify(X)[1]))

# Function that returns P(term | class) for each class
# Note: the term is specified by the user; captured from the simple text interface
def termPosteriors(word):
    uniques = trainData['Category'].nunique()
    words = []
    for i in range(uniques):
        index = np.where(dict_[str(i)][0] == word)
        z = index[0]
        words.append(posteriors[i][int(z)])
    print('Posteriors for feature: ' + word)
    for i in range(len(words)):
        print('Class Label ' + str(i) + ': ' + str(words[i]))

# The Interface: a simple text interface for interacting with the Naive Bayes Algorithm for Text Classification
while True:
    # Retrieving input from the user
    userInput = input('Menu Options:\n'
                    '0) Display Training Data Table\n'
                    '1) Display Testing Data Table\n'
                    '2) Display Prior Probabilities of Class Labels\n'
                    '3) Display Posteriors for Terms\n'
                    '4) Display Predicted Vs. Actual and Log Probability Values\n'
                    '5) Display Classification Accuracy Score\n'
                    '*) Exit\n'
                    'Please type the corresponding number: ')
    # Allows the user to view the Training Data Table in the output terminal
    if userInput == '0':
        print(trainDataTable())
        print()
    # Allows the user to view the Testing Data Table in the output terminal
    elif userInput == '1':
        print(testingDataTable())
        print()
    # Returns the Class Labels and Associated Prior Probabilities to the output terminal
    elif userInput == '2':
        print()
        i = 0
        print('Class Labels and Associated Prior Probabilities -->')
        for key in c_dict.keys():
            print('Class Label: ' + str(key) + '  ' + 'Prior: ' + str(priors[i]))
            i += 1
        print()
    # Provides the posteriors for each term and the associated class label
    # Note: the term is specified by the user
    elif userInput == '3':
        print()
        i = input('Please enter a term: ')
        print()
        i.lower()
        word = i
        termPosteriors(word)
        print()
    # Outputs the predicted, actual, and log probability values for a range of values (specified by the user)
    elif userInput == '4':
        print()
        x = input('Please type starting range: ')
        x = int(x)
        y = input('Please type ending range: ')
        y = int(y)
        print()
        print('Predicted, Actual, and Log Probability Values for Range ' + str(x) + ' to ' + str(y-1) + ' -->')
        docInstances(x, y)
        print()
    # Outputs the overall classification accuracy across all test instances
    elif userInput == '5':
        print()
        print('Calculating model accuracy score...')
        print()
        print('Accuracy Score: ' + str(evaluation()))
        print()
    # Exit the program gracefully
    elif userInput == '*':
        quit()

