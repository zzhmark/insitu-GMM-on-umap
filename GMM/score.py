import numpy as np

from sklearn.metrics import normalized_mutual_info_score
import pandas as pd


def blob_score(pts1: np.ndarray, pts2: np.ndarray,
               mean1: pd.Series, mean2: pd.Series):
    """
    :param pts1: blob1 points coordinate list
    :param pts2: blob2 points coordinate list
    :param mean1: the mean intensity of blob1
    :param mean2: the mean intensity of blob2
    :return: a score
    Calculate 2 blobs' similarity score, determined by
    the relative difference of their intensity and their
    relative overlap area.
    """
    grad_term = 1 - np.abs(mean1 - mean2) / 256
    set1, set2 = set(tuple(i) for i in pts1), set(tuple(i) for i in pts2)
    overlap_term = len(set1.intersection(set2)) / len(set2.union(set2))
    return grad_term * overlap_term


def local_gmm_score(label1: np.ndarray, label2: np.ndarray,
                    means1: pd.Series, means2: pd.Series) -> float:
    """
    :param label1: labeled image, 2D numpy array
    :param label2: labeled image, 2D numpy array
    :param means1: means for points in different levels
    :param means2: means for points in different levels
    :return: the local score
    Calculate 2 images' similarity score, by working out similarity
    scores between any 2 blobs, and sum up the best matches.
    """
    n1, n2 = len(means1), len(means2)
    # List points for different levels.
    pts_list1 = [label1 == i + 1 for i in range(n1)]
    pts_list2 = [label2 == i + 1 for i in range(n2)]
    score_blobs = np.zeros((n1, n2))
    for i, pts1, mean1 in zip(range(n1), pts_list1, means1):
        for j, pts2, mean2 in zip(range(n2), pts_list2, means2):
            score_blobs[i, j] = blob_score(pts1, pts2, mean1, mean2)
    return np.max(score_blobs, axis=0).sum() + np.max(score_blobs, axis=1).sum()


def global_gmm_score(label1: np.ndarray, label2: np.ndarray):
    """
    :param label1: labeled image, 2D numpy array
    :param label2: labeled image, 2D numpy array
    :return: score
    """
    return normalized_mutual_info_score(label1, label2)
