import pickle
import itertools
import string
import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


# Get image on same format as training data
def preprocess(image, pca):
    # flatten array
    image_arr = image.flatten()
    # PCA
    image_arr = pca.transform([image_arr])
    return image_arr

# Remove images that are not centered around a letter


def check_for_white(image):
    white_top = True
    white_left = True
    white_bottom = True
    white_right = True
    for entry in image[0]:
        if entry != 1.0:
            white_top = False
    for entry in image[:, 0]:
        if entry != 1.0:
            white_left = False
    for entry in image[image.shape[0]-1]:
        if entry != 1.0:
            white_bottom = False
    for entry in image[:, image.shape[1]-1]:
        if entry != 1.0:
            white_right = False

    return white_top or white_left or white_bottom or white_right


# Threshold is to decide whether or not classification is good enough to be included
def classify(image, classifier, pca, threshold=0.999):
    if check_for_white(image):
        return -1
    image = preprocess(image, pca)
    predicted = classifier.predict_proba(image)
    pred_index = predicted.argmax()
    pred_val = predicted[0][pred_index]
    if pred_val >= threshold:
        return pred_index, pred_val
    else:
        return -1


def load_image(path):
    image = cv.imread(path, 0)
    # image = cv.imread('./detection-images/detection-1.jpg', 0)
    pixels = np.divide(image, 255)
    return pixels


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    windows = []
    for y in range(0, image.shape[0] - window_size[0], step_size):
        for x in range(0, image.shape[1] - window_size[1], step_size):
            # yield the current window
            windows.append(
                (y, x, image[y:(y + window_size[0]), x:(x + window_size[1])]))

    return windows


def classify_best_windows(image, classifier, pca,  step_size=5, window_size=(20, 20)):
    predicted = {}
    c = 0
    for (y, x, window) in sliding_window(image, step_size, window_size):
        pred = classify(window, classifier, pca)
        if pred != -1:
            predicted[c] = [[(y, x), pred, window]]
            c += 1
    return predicted


def separate_data_in_lists(predicted):
    coordinates = []
    pred = []
    prob = []
    for k, v in predicted.items():
        for l in v:
            coordinates.append(l[0])
            pred.append(l[1][0])
            prob.append(l[1][1])
    return coordinates, pred, prob


def make_neighbour_list(coordinates, pred, prob):
    euclidean = euclidean_distances(coordinates)
    neighbour_list = []
    for row in euclidean:
        neighbours = []
        for i in range(len(row)):
            # 20 is the distance threshold
            if row[i] < 20:
                neighbours.append([coordinates[i], pred[i], prob[i]])
        neighbour_list.append(neighbours)
    return neighbour_list


# help function return the third element of a list
def sort_third(val):
    return val[2]

# if two or more windows are close together (e.g. euclidean distance of 20), then chose the one with
# higher predicted probability


def remove_clusters(neighbour_list):
    final_windows = []
    for i in range(0, len(neighbour_list)):
        neighbour_list[i].sort(key=sort_third, reverse=True)
        final_windows.append(neighbour_list[i][0][0:2])
    return final_windows


def remove_duplicates(final_windows):
    final_windows.sort()
    final_windows = list(final_windows for final_windows,
                         _ in itertools.groupby(final_windows))
    return final_windows


def main():
    classifier = pickle.load(open('./models/nn_model', 'rb'))
    pca = pickle.load(open('./models/pca', 'rb'))
    image = load_image('./detection-images/detection-1.jpg')
    window_size = (20, 20)
    temp = cv.imread('./detection-images/detection-1.jpg', cv.IMREAD_COLOR)
    # temp = cv.imread('./detection-images/detection-1.jpg', cv.IMREAD_COLOR)
    predicted = classify_best_windows(
        image, classifier, pca, step_size=1, window_size=(20, 20))
    (coordinates, pred, prob) = separate_data_in_lists(predicted)
    neighbour_list = make_neighbour_list(coordinates, pred, prob)
    final_windows = remove_clusters(neighbour_list)
    final_windows = remove_duplicates(final_windows)

    for k in final_windows:
        y, x = k[0]
        # draw rectangle on image
        cv.rectangle(
            temp, (x, y), (x + window_size[1], y + window_size[0]), (0, 255, 0), 1)
        letter = string.ascii_lowercase[k[1]]
        print(letter)
        cv.putText(temp, letter, (x, y + 2 *
                                  window_size[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        plt.imshow(temp, cmap="gray", interpolation='nearest')

    plt.show()


main()
