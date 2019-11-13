
import string

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pickle

import matplotlib.pyplot as plt
import seaborn as sn

from preprocessing import get_data, pca_transform


def knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k, probability=True)
    model.fit(X_train, y_train)
    save_model(model,'knn')
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    score(model, X_test, y_test)


def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, probability=True)
    model.fit(X_train, y_train)
    save_model(model,'rf')
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    score(model, X_test, y_test)


def nn(X_train, X_test, y_train, y_test):
    model = MLPClassifier(max_iter=500, hidden_layer_sizes=(200), probability=True)
    model.fit(X_train, y_train)
    save_model(model,'nn')
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    sco = score(model, X_test, y_test)
    return sco


def svm(X_train, X_test, y_train, y_test):
    model = SVC(gamma='auto', probability=True)
    model.fit(X_train, y_train)
    save_model(model,'svm')
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    score(model, X_test, y_test)



def plot_confusion_matrix(conf_matrix, title):
    sn.heatmap(conf_matrix, annot=True)
    plt.title(title)
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.show()


def score(model, X_test, y_test):
    scores = {}
    predictions = model.predict(X_test)
    for i in range(len(X_test)):
        if y_test[i] not in scores.keys():
            scores[y_test[i]] = [0, 0]
        if predictions[i] == y_test[i]:
            scores[y_test[i]][0] += 1
        scores[y_test[i]][1] += 1
    for key, value in scores.items():
        output = "The score for {0} was {1}%".format(
            string.ascii_lowercase[key], round(value[0]/value[1]*100, 2))
        print(output)
    total_score = round(model.score(X_test, y_test)*100, 2)
    print("The total score was: {0}%".format(total_score))
    return total_score

def save_model(model, classifier):
    with open('./models/{}_model'.format(classifier), 'wb') as f:
        pickle.dump(model,f)

def load_model(classifier='svm'):
    with open('./models/{}_model'.format(classifier),'rb') as f:
        model = pickle.load(f)
    return model

def main():

    X_train, X_test, y_train, y_test = get_data(type_of_data='Default')
    score = svm(X_train, X_test, y_train, y_test)
    #classifier = load_model()
    #print(classifier.predict(X_test))

main()
