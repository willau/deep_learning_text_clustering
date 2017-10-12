# -*- coding: utf-8 -*-

from collections import Counter
from operator import itemgetter
from sklearn import metrics
import numpy as np


def map_label(true_labels, pred_labels):
    label_pair = list(zip(pred_labels, true_labels))
    count = tuple(Counter(label_pair).items())
    mapping = dict()
    n_label = len(np.unique(true_labels))

    # map most likely labels from prediction to ground truth
    for label in range(n_label):
        tuples = [tup for tup in count if tup[0][0] == label]
        likely_tuple = max(tuples, key=itemgetter(1))[0]
        mapping[likely_tuple[0]] = likely_tuple[1]

    pred_labels_mapped = [mapping[x] for x in pred_labels]
    return pred_labels_mapped


def cluster_quality(true_labels, pred_labels):
    h, c, v = metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    rand = metrics.adjusted_rand_score(true_labels, pred_labels)
    print("Homogeneity: %0.3f" % h)
    print("Completeness: %0.3f" % c)
    print("V-measure: %0.3f" % v)
    print("NMI: %0.3f" % nmi)
    print("Rand: %0.3f" % rand)
    pred_labels_mapped = map_label(true_labels, pred_labels)
    acc = metrics.accuracy_score(true_labels, pred_labels_mapped)
    print("Acc: %0.3f" % acc)