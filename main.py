#classification problem
#3 classes
#4 features
#shuffle first
#sklearn for splitting
#3 distance measures
#experiment different values for k
#count number of correct and wrong predictions
#calc accuracy score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt

def majority(distances):
    labels = [d[1] for d in distances]

    setosa = labels.count("Iris-setosa")
    virginica = labels.count("Iris-virginica")
    versicolor = labels.count("Iris-versicolor")
    # print(setosa, virginica, versicolor)
    if setosa > virginica and setosa > versicolor:
        return "Iris-setosa"
    elif virginica > setosa and virginica > versicolor:
        return "Iris-virginica"
    elif versicolor > setosa and versicolor > virginica:
        return "Iris-versicolor"
    else:
        return "undefined" ##handle later
###
def euclidean(x, k: int): #returns an array of sorted distances
    distances = []
    for i in range(len(X_train)):
        dist = ((x[0] - X_train[i][0]) ** 2 +
                (x[1] - X_train[i][1]) ** 2 +
                (x[2] - X_train[i][2]) ** 2 +
                (x[3] - X_train[i][3]) ** 2)
        distances.append((sqrt(dist),Y_train[i]))
    distances.sort(key=lambda x:x[0]) #sort based on dist...
    return majority(distances[:k])
####
def manhattan(x, k: int): #returns an array of sorted distances
    distances = []
    for i in range(len(X_train)):
        dist = (abs((x[0] - X_train[i][0])) +
                abs((x[1] - X_train[i][1])) +
                abs((x[2] - X_train[i][2])) +
                abs((x[3] - X_train[i][3])))
        distances.append((dist,Y_train[i]))
    distances.sort(key=lambda x:x[0]) #sort based on dist...
    return majority(distances[:k])
###

def cosine_similarity(x, k:int):
    distances = []
    for i in range(len(X_train)):
        dot = x[0]*X_train[i][0] + x[1]*X_train[i][1] + x[2]*X_train[i][2] + x[3]*X_train[i][3]
        mag_x = sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)
        mag_x_train = sqrt(X_train[i][0]**2 + X_train[i][1]**2 + X_train[i][2]**2 + X_train[i][3]**2)
        distances.append((dot/(mag_x*mag_x_train), Y_train[i]))
    distances.sort(key=lambda x:x[0], reverse=True)
    return majority(distances[:k])
#if 0 -> euclidean
#if 1 -> manhattan
#if 2 -> cosine similarity
def KNeighborsClassifier(k:int,dist_type: int):
    ans = ""
    predictions = []
    correct = 0
    for i in range(len(X_test)):
        match dist_type:
            case 0:
                ans = euclidean(X_test[i], k)
            case 1:
                ans = manhattan(X_test[i], k)
            case 2:
                ans = cosine_similarity(X_test[i], k)
        ##
        # print(X_test[i], ans, Y_test[i])
        ##
        predictions.append(ans)
        if ans == Y_test[i]:
            correct += 1
    print("correct: "+str(correct))
    print("wrong: "+str(len(X_test)-correct))
    predictions = np.array(predictions)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Accuracy: {round(accuracy*100,2)}%")

df = pd.read_csv('D:\Y3_S2\SL\Assignment1\iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(df.head())
#count is the # of non null values in that column
print(df.describe())
# print(df.info())
sns.lmplot(x='sepal_length', y='sepal_width', data=df, hue='species', fit_reg=False)
plt.show()
sns.lmplot(x='petal_length', y='petal_width', data=df, hue='species', fit_reg=False)
plt.show()
######################
df = shuffle(df, random_state=42)
X = df.iloc[:, :-1] #extracting all cols
Y = df.iloc[:, -1] #extracting the species column (output)
###split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
###
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()
# print(Y_train)
###
#testing out all possible combinations of hyperparameters
algorithms = {0:"Euclidean Distance", 1:"Manhattan Distance", 2:"Cosine Similarity"}
for i in range(1,5):
    for j in range(3):
        print(f"Results for k = {i} and using {algorithms[j]}:")
        KNeighborsClassifier(i, j)









