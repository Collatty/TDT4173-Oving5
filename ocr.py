
import string

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from preprocessing import get_data, pca_transform


def knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    sco = new_score(model, X_train, X_test, y_train, y_test)
    return sco


def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    prdeictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, prdeictions)
    score(model, X_test, y_test)


def nn(X_train, X_test, y_train, y_test):
    model = MLPClassifier(max_iter=500, hidden_layer_sizes=(200))
    model.fit(X_train, y_train)
    #sco = new_score(model, X_train, X_test, y_train, y_test)
    sco = score(model, X_test, y_test)
    return sco


def svm(X_train, X_test, y_train, y_test):
    model = SVC(gamma='auto')
    model.fit(X_train, y_train)
    prdeictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, prdeictions)
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
        # if predictions[i] != y_test[i]:
            # if y_test[i] == 25:
            #     print_photo(X_test[i])
            #     print(string.ascii_lowercase[predictions[i]])
        scores[y_test[i]][1] += 1
    for key, value in scores.items():
        output = "The score for {0} was {1}%".format(
            string.ascii_lowercase[key], round(value[0]/value[1]*100, 2))
        print(output)
    total_score = round(model.score(X_test, y_test)*100, 2)
    print("The total score was: {0}%".format(total_score))
    return total_score


def new_score(model, X_train, X_test, y_train, y_test):
    data_features = np.concatenate((X_train, X_test))
    data_labels = np.concatenate((y_train, y_test))

    score = cross_val_score(model, data_features, data_labels, cv=5)
    return score


def plot_score_PCA(accuracy_with_pca, accuracy_no_pca, title):
    #average_with_pca = sum(accuracy_with_pca)/len(accuracy_with_pca)
    #average_no_pca = sum(accuracy_no_pca)/len(accuracy_no_pca)
    plt.title(title)
    plt.ylabel('Accuracy percentage')
    plt.plot(accuracy_with_pca)
    plt.plot(accuracy_no_pca)
    # plt.plot(average_with_pca, linestyle='-.')
    # plt.plot(average_no_pca, linestyle='-.')
    plt.legend(['With PCA', 'No PCA'])
    plt.gca().set_yticklabels(['{:,.2f}%'.format(x)
                               for x in plt.gca().get_yticks()])
    plt.show()


def plot_score_hidden_layers_size(scores, number_in_hidden):
    plt.ylabel('Accuracy percentage')
    plt.plot(number_in_hidden, scores)
    plt.gca().set_yticklabels(['{:,.2f}%'.format(x)
                               for x in plt.gca().get_yticks()])
    plt.show()


def print_photo(array):
    array = np.reshape(array, (20, 20))
    plt.imshow(array, cmap='gray')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = get_data(
        type_of_data='Default', pca=False)
    score = nn(X_train, X_test, y_train, y_test)


main()
