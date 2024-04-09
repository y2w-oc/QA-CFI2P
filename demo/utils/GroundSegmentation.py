import random
import numpy as np
import math


def ground_segmentation(data):
    idx_segmented = []
    segmented_cloud = []
    iters = 100
    sigma = 0.4
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    pretotal = 0

    P = 0.99
    n = len(data)
    outline_ratio = 0.6
    for i in range(iters):
        ground_cloud = []
        idx_ground = []

        sample_index = random.sample(range(n), 3)
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]

        point1_2 = (point1 - point2)
        point1_3 = (point1 - point3)
        N = np.cross(point1_3, point1_2)

        a = N[0]
        b = N[1]
        c = N[2]
        d = -N.dot(point1)

        total_inlier = 0
        pointn_1 = (data - point1)
        distance = abs(pointn_1.dot(N)) / np.linalg.norm(N)

        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)

        if total_inlier > pretotal:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / n, 3))  # N = ------------
            pretotal = total_inlier
            best_a = a
            best_b = b
            best_c = c
            best_d = d

        if total_inlier > n * (1 - outline_ratio):
            break

    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]
    return ground_cloud.T, segmented_cloud.T