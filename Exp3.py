import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import ConvexHull
import pandas

Q = [0, 10]
k = 7
num_test_points = 10000
num_repetitions = 20
n = 800

def calculate_area(vertices):
    return ConvexHull(np.array([vertices[0], vertices[1], vertices[2]])).volume

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

def generate_triangles():
    triangles = {}
    
    ratios = np.arange(1, 5.5, 0.5)
    for r in ratios:
        h = np.sqrt(16 / r)
        w = r * h
        
        A = np.array([5 - (w / 2), 3])
        B = np.array([5 + (w / 2), 3])
        C = np.array([5 + (w / 2), 3 + h])
        
        #shift
        shift_x = 9.5 - B[0]
        A_shifted = A + np.array([shift_x, 0])
        B_shifted = B + np.array([shift_x, 0])
        C_shifted = C + np.array([shift_x, 0])

        triangle = np.array([A_shifted, B_shifted, C_shifted])
        triangles[r] = triangle
        
    return triangles

def generate_triangles_old():
    triangles = {}
    
    ratios = np.arange(1, 5.5, 0.5)
    for r in ratios:
        h = np.sqrt(16 / r)
        w = r * h
        A = np.array([5 - (w/2), 3])
        B = np.array([5 + (w/2), 3])
        C = np.array([5 + (w/2), 3 + h])
        triangle = np.array([A, B, C])
        
        triangles[r] = triangle
    return triangles

triangle_shapes = generate_triangles()

rq3_results = []
for r, T in triangle_shapes.items():
    area = calculate_area(T)
    perimeter = calculate_perimeter(T)
    #print(area)
    #print(perimeter)
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

    avg_misclassification = np.mean(misclassifications_per_repeat)
    std_dev_misclassification = np.std(misclassifications_per_repeat)
    rq3_results.append((r, perimeter, avg_misclassification, std_dev_misclassification))

filename = "Exp3_Results.csv"
df = pandas.DataFrame(data=rq3_results)
df.to_csv(filename, index=False)

"""
#triangles
plt.figure(figsize=(8, 8))
for idx, T in enumerate(triangle_shapes.values()):
    polygon = Polygon(T, closed=True, fill=None, edgecolor=f"C{idx}", linewidth=2, label=f'Triangle {idx+1}')
    plt.gca().add_patch(polygon)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Triangles with varying Ratios")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
"""