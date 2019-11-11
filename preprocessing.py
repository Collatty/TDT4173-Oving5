

import glob
import random
import string
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2 as cv

import numpy as np
import skimage as sk
from skimage import transform
from skimage import util


# Allows you to load the data as 20x20 arrays
def load_data():
    data_features = []
    data_labels = []
    counter = 0  # map from charcs to numbers
    for character in string.ascii_lowercase:
        features_for_character = []
        labels_for_character = []
        for image in glob.iglob('./chars74k-lite/'+character+'/*.jpg'):
            image = cv.imread(image, 0)
            #thresh, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
            # image = cv.fastNlMeansDenoising(
            # image, templateWindowSize=7, searchWindowSize=3)
            pixels = np.array(image)
            pixels = np.divide(pixels, 255)
            pixels = np.array(pixels)
            features_for_character.append(pixels)
            labels_for_character.append(counter)
        data_features.append(features_for_character)
        data_labels.append(labels_for_character)
        counter += 1
    return data_features, data_labels


def get_data(type_of_data='Default'):
    data_features, data_labels = load_data()

    if type_of_data == 'Default':
        # data_features, data_labels = grow_dataset_through_augmentation(
            # 1000, data_features, data_labels)
        X_train, X_test, y_train, y_test, separated_X_train, separated_X_test, separated_y_train,  separated_y_test = split_data(
            data_features, data_labels)

        X_train = flatten_array_1D(X_train)
        X_test = flatten_array_1D(X_test)
        #X_train, X_test = pca_transform(X_train, X_test)
        return X_train, X_test, y_train, y_test

    elif type_of_data == "Untouched_test":
        X_train, X_test, y_train, y_test, separated_X_train, separated_X_test, separated_y_train,  separated_y_test = split_data(
            data_features, data_labels)
        X_train, y_train = grow_dataset_through_augmentation(
            200, separated_X_train, separated_y_train)
        X_train = flatten_array_2D(X_train)
        X_test = flatten_array_1D(X_test)
        X_train = join_lists(X_train)
        y_train = join_lists(y_train)
        #X_train, X_test = pca_transform(X_train, X_test)
        return X_train, X_test, y_train, y_test
    elif type_of_data == "Touched_test":
        # data_features, data_labels = grow_dataset_through_augmentation(
        #    1000, data_features, data_labels)
        X_train, X_test, y_train, y_test, separated_X_train, separated_X_test, separated_y_train,  separated_y_test = split_data(
            data_features, data_labels)
        X_train = flatten_array_1D(X_train)
        X_test = flatten_array_1D(X_test)
        #X_train, X_test = pca_transform(X_train, X_test)
        return X_train, X_test, y_train, y_test


# Reduce number of features to 100
def pca_transform(X_train, X_test):
    all_observations = X_train + X_test
    pca = PCA(40)
    all_observations = pca.fit_transform(all_observations)
    X_train = all_observations[0:len(X_train)]
    X_test = all_observations[-len(X_test):]
    return X_train, X_test


def split_data(data_features, data_labels):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    separated_X_train = []
    separated_y_train = []
    separated_X_test = []
    separated_y_test = []
    for i in range(len(data_features)):
        character_X_train, character_X_test, character_y_train, character_y_test = train_test_split(
            data_features[i], data_labels[i], test_size=0.2)
        X_train += character_X_train
        X_test += character_X_test
        y_train += character_y_train
        y_test += character_y_test
        separated_X_train.append(character_X_train)
        separated_y_train.append(character_y_train)
        separated_X_test.append(character_X_test)
        separated_y_test.append(character_y_test)
    return X_train, X_test, y_train, y_test, separated_X_train, separated_X_test, separated_y_train, separated_y_test

# Create more instances to train/test on


def grow_dataset_through_augmentation(number_required, data_features, data_labels):
    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }
    for i in range(len(string.ascii_lowercase)):
        while number_required > len(data_features[i]):
            random_number = random.randint(0, len(data_features[i])-1)
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](
                data_features[i][random_number])
            data_features[i].append(transformed_image)
            data_labels[i].append(i)
    return data_features, data_labels


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    # flip image
    return image_array[:, ::-1]


def flatten_array_2D(features):
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] = features[i][j].flatten()
    return features


def flatten_array_1D(features):
    for i in range(len(features)):
        features[i] = features[i].flatten()
    return features


def unflatten_array(array):
    if array.shape == (20, 20):
        return array
    return np.reshape(array, (20, 20))


def join_lists(list_of_lists):
    li = []
    for lis in list_of_lists:
        li += lis
    return li
