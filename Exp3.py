import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.neighbors import KNeighborsClassifier

Q = [0, 10]
k = 7
num_test_points = 10000
num_repetitions = 20
n = 800

def calculate_perimeter(vertices):
    return (
        np.linalg.norm(vertices[0] - vertices[1])
        + np.linalg.norm(vertices[1] - vertices[2])
        + np.linalg.norm(vertices[2] - vertices[0])
    )

def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def is_inside_triangle(point, vertices):
    p1 = sign(point, vertices[0], vertices[1])
    p2 = sign(point, vertices[1], vertices[2])
    p3 = sign(point, vertices[2], vertices[0])
    return (p1 < 0 and p2 < 0 and p3 < 0) or (p1 > 0 and p2 > 0 and p3 > 0)

#to generate triangles with varying perimeters but the same starting point for A
def generate_triangles_with_vertices():
    triangles = []
    for i in np.arange(0, 6.5, 0.5):
        A = np.array([3, 3])
        B = np.array([3.5 + i, 3])
        C = np.array([3.5 + i, 3.5 + i])
        triangle = np.array([A, B, C])
        triangles.append(triangle)
    return triangles

#triangle generated
triangle_shapes = generate_triangles_with_vertices()

rq3_results = []

for T in triangle_shapes:
    perimeter = calculate_perimeter(T)
    misclassifications_per_repeat = []
    
    for _ in range(num_repetitions):

        points = np.random.uniform(Q[0], Q[1], (n, 2))
        labels = np.array([is_inside_triangle(p, T) for p in points])

        test_points = np.random.uniform(Q[0], Q[1], (num_test_points, 2))
        true_labels = np.array([is_inside_triangle(p, T) for p in test_points])

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(points, labels)
        predictions = knn.predict(test_points)

        misclassified = np.sum(predictions != true_labels)
        misclassifications_per_repeat.append(misclassified)

    #data to collect
    avg_misclassification = np.mean(misclassifications_per_repeat)
    std_dev_misclassification = np.std(misclassifications_per_repeat)
    rq3_results.append((perimeter, avg_misclassification, std_dev_misclassification))

perimeters, avg_misclassifications, std_devs = zip(*rq3_results)
plt.errorbar(perimeters, avg_misclassifications, yerr=std_devs, fmt='-o', capsize=5)
plt.xlabel('Perimeter of Triangle')
plt.ylabel('Average Misclassifications')
plt.title('Misclassification Dependency on Triangle Perimeter')
plt.grid()
plt.show()
