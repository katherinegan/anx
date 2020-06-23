import numpy as np
import argparse
from cv2 import cv2
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from PIL import Image
# img = Image.open("../Images/testimg.jpg")


def get_patch(patch_center, image):
    '''select image patch in colored disc'''
    # define start and end values for image patch
    start_row = int(patch_center[1] - 4)
    end_row = int(patch_center[1] + 4)
    start_col = int(patch_center[0] - 4)
    end_col = int(patch_center[0] + 4)
    patch_image = image[start_row:end_row, start_col:end_col]
    return patch_image


def make_histogram(cluster):
    '''count the number of pixels in each cluster'''
    num_labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=num_labels)
    # normalize histogram
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    '''create image of given color
    returns tuple of bar and hsv values'''
    rgb_bar = np.zeros((height, width, 3), np.uint8)
    rgb_bar[:] = color
    blue, green, red = rgb_bar[0][0]
    return rgb_bar, (red, green, blue)


def sort_hsvs(hsv_list):
    '''sort list of HSV values
    returns list of indexes sorted by hue, then saturation, lastly value'''
    index_bars = []
    for index, hsv_val in enumerate(hsv_list):
        index_bars.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
    index_bars.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in index_bars]


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    largest = max(r, g, b)
    smallest = min(r, g, b)
    diff = largest - smallest
    if largest == smallest:
        h = 0
    elif largest == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif largest == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif largest == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    if largest == 0:
        s = 0
    else:
        s = (diff / largest)*100
    v = largest * 100
    return h, s, v


def get_h_value(patch):
    height, width, _ = np.shape(patch)
    # reshape array to be a list of RGB pixels
    image = patch.reshape((height * width), 3)

    # find the 3 most dominant colors in an image by clustering pixel intensities
    num_clusters = 1
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    histogram = make_histogram(clusters)
    # sort
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    bars = []
    rgb_values = []
    for index, rows in enumerate(combined):
        bar, rgb = make_bar(100, 100, rows[1])
        #print(f'Color {index + 1}')
        #print(f'  RGB values: {rgb}')
        #print('  HSV values:', rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
        rgb_values.append(rgb)
        bars.append(bar)

    # getting list of HSV values for 3 most dominant colors
    hsv_values = []
    for color in rgb_values:
        hue, sat, val = rgb_to_hsv(color[0], color[1], color[2])
        hsv_values.append((hue, sat, val))

    # get only h value for 3 most dominant colors
    h_values = []
    for i in range(1):
        h_values.append(hsv_values[i][0])

    return h_values


img = cv2.imread(
    "Images/calibration_pictures/occult_blood/140-3.png")
#hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
patch_centers = [[430, 50], [50, 430],
                 [50, 50], [430, 430]]  # counterclockwise staring at top right corner
all_h_values = []
for center in patch_centers:
    patch = get_patch(center, img)
    all_h_values.append(get_h_value(patch))

# find average h value for the color of each disc based on the 3 most dominant colors
avg_h_values = []
for disc in all_h_values:
    avg_h_values.append(np.average(disc))
average_h_value = np.average(avg_h_values)
print('Average H value by disc starting at the top right corner, moving counterclockwise:')
print(avg_h_values)


'''test and display the patch extracted from the image'''
# patch_image = get_patch([430, 430], img)
# h_value = []
# h_value.append(get_h_value(patch_image))
# height, width = patch_image.shape[0:2]
# print(height, width)

'''display image patch in a separate window'''
# cv2.imshow('patch_image', patch_image)
# cv2.waitKey()


'''sort bars by HSV values'''
# sorted_bar_indexes = sort_hsvs(rgb_values)
# sorted_bars = [bars[i] for i in sorted_bar_indexes]

'''displays 2 new windows, each with bars of the 3 most dominant colors in image'''
# cv2.imshow('Sorted by HSV values', np.hstack(sorted_bars))
# cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
# cv2.waitKey(0)
