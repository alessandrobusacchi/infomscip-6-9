import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas

Q = [0, 10]
T = np.array([[3, 3], [7, 3], [7, 7]])
k = 7
num_test_points = 10000
num_values = np.arange(100, 1100, 100)
num_repetitions = 20

#point in triangle?
def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def is_inside_triangle(point, vertices):
    p1 = sign(point, vertices[0], vertices[1])
    p2 = sign(point, vertices[1], vertices[2])
    p3 = sign(point, vertices[2], vertices[0])

    if (p1 < 0 and p2 < 0 and p3 < 0) or (p1 > 0 and p2 > 0 and p3 > 0):
        return True
    else:
        return False

rq1_results = []
#100,200,300,...,1000
for n in num_values:
    misclassifications_per_repeat = []
    #20 times
    for _ in range(num_repetitions):
        
        #generate n points randomly inside square based on value n and check if red or blue
        np.random.seed = 0
        points = np.random.uniform(Q[0], Q[1], (n, 2))
        labels = np.array([True if is_inside_triangle(p, T) else False for p in points])

        #do same thing with test points (10000)
        test_points = np.random.uniform(Q[0], Q[1], (num_test_points, 2))
        true_labels = np.array([True if is_inside_triangle(p, T) else False for p in test_points])

        #check ig points are classified correctly or not
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(points, labels)
        predictions = knn.predict(test_points)

        misclassified = np.sum(predictions != true_labels)
        misclassifications_per_repeat.append(misclassified)

    #stats to collect and to show )avg and std deviation)
    avg_misclassification = np.mean(misclassifications_per_repeat)
    std_dev_misclassification = np.std(misclassifications_per_repeat)
    rq1_results.append((n, avg_misclassification, std_dev_misclassification))


filename = "Exp1_Results.csv"
df = pandas.DataFrame(data=rq1_results)
df.to_csv(filename, index=False)