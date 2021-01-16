from __future__ import division, unicode_literals
from collections import Counter
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import re

def distance(v, w):
   return math.sqrt(squared_distance(v, w))

def mean(x): 
    return sum(x) / len(x)

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels):
    """labels�� ���� ����� �����ͺ��� ���� �� ������ ������ ���ĵǾ� �ִٰ� ����"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                        for count in vote_counts.values()
                        if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

def knn_classify(k, labeled_points, new_point):
    """�� labeled_points�� (����������Ʈ, ���̺�) ������ �����Ǿ� ����"""

    #labeled_points�� ���� ����� �����ͺ��� ���� �� ������ ������ ����
    by_distance = sorted(labeled_points, key = lambda (point, _): distance(point, new_point))

    #���� ����� k ������ ����Ʈ�� ���̺��� ���캸��
    k_nearest_labels = [label for _, label in by_distance[:k]]

    #��ǥ�� ��
    return majority_vote(k_nearest_labels)

cities = [(-86.75,33.57,'Python'),(-88.25,30.63,'Python'),(-112.01,33.43,'Java'),(-110.93,32.116,'Java'),(-92.23,34.73,'R'),(-121.95,37.7,'R')]

#���α׷��� �� Ű, (�浵, ����) ���� ��
plots = { "Java" : ([], []), "Python" :([], []), "R" : ([], [])}

#�� ���α׷��� ���� ����� ��Ŀ�� �ٸ�
markers = {"Java" : "o", "Python" : "s", "R" : "^"}
colors = {"Java" : "r", "Python" : "b", "R" : "g"}

for (longitude, latitude), language in cities:
        plots[language][0].append(longitude)
        plots[language][1].append(latitude)

#�� ���α׷��� ���� ������ ����Ʈ�� �Ѹ�
for language, (x, y) in plots.iteritems():
    plt.scatter(x, y, color = colors[language], marker = markers[language], label = language, zorder = 10)

plt.legend(loc=0)
plt.axis([-130, -60, 20, 55])

plt.title("Favorite Programming Languages")
plt.show()

for k in [1, 3, 5, 7]:
    num_correct = 0

    for city in cities:
        location, actual_language = city
        other_cities = [other_city
                        for other_city in cities
                        if other_city != city]
        predicted_language = knn_classify(k, other_citits, location)

        if predicted_language == actual_language:
            num_correct += 1
    
    print k, "neighbor[s]:", num_correct, "correct out of", len(cities)

def random_point(dim):
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim))
            for _ in rnage(num_pairs)]

dimensions = range(1, 101)

avg_distances = []
min_distances = []

random.seed(0)

for dim in dimensions:
    distances = random_distances(dim, 10000)
    avg_distances.append(mean(distances))
    min_distances.append(min(distances))

min_avg_ratio = [min_dist / avg_dist
                 for min_dist, avg_dist in zip(min_distances, avg_distances)]
