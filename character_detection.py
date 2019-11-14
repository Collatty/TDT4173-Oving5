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
       #flatten array
       image_arr = image.flatten()
       #PCA
       pca = load_pca()
       image_arr = pca.transform([image_arr])
       return image_arr

def classify(image, threshold=0.8):
       image = preprocess(image)
       predicted = classifier.predict_proba(image)
       pred_index = predicted.argmax()
       pred_val = predicted[0][pred_index]
       if(pred_val >= threshold):
              return (pred_index, pred_val)
       else:
              return -1

def load_image():
       image = cv.imread('./detection-images/detection-1.jpg', 0)
       #image = cv.imread('./detection-images/detection-2.jpg', 0)
       pixels = np.divide(image, 255)
       return pixels

def sliding_window(image, stepSize, windowSize):
       # slide a window across the image
       for y in range(0, image.shape[0] - windowSize[0], 2*stepSize):
              for x in range(0, image.shape[1] - windowSize[1], stepSize):
                     # yield the current window
                     yield (y, x, image[y:(y + windowSize[0]), x:(x + windowSize[1])])

image = load_image()
temp = image
windowSize = (20,20)
predicted = {}
for (y,x, window) in sliding_window(image, stepSize=2, windowSize=windowSize):
       pred = classify(window)
       if(pred!=-1):
              #print('(x,y)=({},{})'.format(x,y),pred)
              k = pred[0]
              if k in predicted:
                     predicted[k].append([(y,x),pred,window])
              else:
                     predicted[k] = [[(y,x),pred,window]]
              #plt.imshow(window,cmap="gray")
              #plt.show()

## if two or more windows are close together (e.g. euclidean distance of 5), then chose the one with
## higher predicted probability

windows = {}
for k,v in predicted.items():
       windows[k] = []
       if(len(v)==1):
              windows[k].append(v[0])
       else:
              #print(v)
              coordinates = []
              pred = []
              for l in v:
                     coordinates.append(l[0])
                     pred.append(l[1][1])

              euclidean = euclidean_distances(coordinates)
              neighbour_list = []
              for row in euclidean:
                     neighbours =[]
                     for i in range(len(row)):
                            # 5 is the distance threshold
                            if(row[i] == 0) or (row[i] < 10):
                                   neighbours.append(i)
                     neighbour_list.append(neighbours)

              neighbour_list.sort()
              neighbour_list = list(neighbour_list for neighbour_list, _ in itertools.groupby(neighbour_list))

              #indeces of windows with highest predicted value of nearby windows
              window_indices = []
              for group in neighbour_list:
                     max_i = 0
                     max_pred = 0
                     for i in group:
                            pred_prob = v[i][1][1]
                            if pred_prob > max_pred:
                                   max_pred = pred_prob
                                   max_i = i

                     window_indices.append(max_i)

              for i in window_indices:
                     windows[k].append(v[i])

       #print(windows)

for k,v in windows.items():
       for w in v:
              print(w[:-1])
              plt.imshow(w[-1],cmap="gray")
              plt.show()

# draw window on image

temp = cv.imread('./detection-images/detection-1.jpg',cv.IMREAD_COLOR)
for k,v in windows.items():
       for w in v:
              y,x = w[0]
              cv.rectangle(temp, (x, y), (x + windowSize[1], y + windowSize[0]), (0, 255, 0), 1) # draw rectangle on image
              letter = string.ascii_lowercase[w[1][0]]
              print(letter)
              cv.putText(temp, letter, (x,y + 2*windowSize[0]), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
              plt.imshow(temp,cmap="gray", interpolation='nearest')
              # show all windows
plt.show()

