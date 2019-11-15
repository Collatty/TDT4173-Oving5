import numpy as np
from preprocessing import load_pca
from ocr import load_model
import cv2 as cv
import itertools
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import string

classifier = load_model()


def preprocess(image):
    # flatten array
    image_arr = image.flatten()
    # PCA
    pca = load_pca()
    image_arr = pca.transform([image_arr])
    return image_arr


def classify(image, threshold=0.68):
    image = preprocess(image)
    predicted = classifier.predict_proba(image)
    pred_index = predicted.argmax()
    pred_val = predicted[0][pred_index]
    if pred_val >= threshold:
        return pred_index, pred_val
    else:
        return -1


def load_image():
    image = cv.imread('./detection-images/detection-2.jpg', 0)
    # image = cv.imread('./detection-images/detection-1.jpg', 0)
    pixels = np.divide(image, 255)
    return pixels


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0] - window_size[0], step_size):
        for x in range(0, image.shape[1] - window_size[1], step_size):
            # yield the current window
            yield (y, x, image[y:(y + window_size[0]), x:(x + window_size[1])])


def classify_best_windows(image, step_size=1, window_size=(20, 20)):
    predicted = {}
    c = 0
    for (y, x, window) in sliding_window(image, step_size, window_size):
        pred = classify(window)
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
    final_windows = list(final_windows for final_windows, _ in itertools.groupby(final_windows))
    return final_windows


def main():
    image = load_image()
    window_size = (20, 20)
    temp = cv.imread('./detection-images/detection-2.jpg', cv.IMREAD_COLOR)
    # temp = cv.imread('./detection-images/detection-1.jpg', cv.IMREAD_COLOR)
    predicted = classify_best_windows(image, step_size=1, window_size=(20, 20))
    (coordinates, pred, prob) = separate_data_in_lists(predicted)
    neighbour_list = make_neighbour_list(coordinates, pred, prob)
    final_windows = remove_clusters(neighbour_list)
    final_windows = remove_duplicates(final_windows)

    for k in final_windows:
        y, x = k[0]
        cv.rectangle(temp, (x, y), (x + window_size[1], y + window_size[0]), (0, 255, 0), 1)  # draw rectangle on image
        letter = string.ascii_lowercase[k[1]]
        print(letter)
        cv.putText(temp, letter, (x, y + 2 * window_size[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        plt.imshow(temp, cmap="gray", interpolation='nearest')

    plt.show()


main()

