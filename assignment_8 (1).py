# -*- coding: utf-8 -*-
"""assignment_8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CojrjID7yJUBRrBBqd_h8edw7J67C5RY
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

file_path = "/content/drive/MyDrive/Data (1).txt"
x = []
y = []
actual_labels = []

# Opening the file to read lines
with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        actual_labels.append(int(float(parts[2])))

# Converting to NumPy arrays for easier handling
x = np.array(x)
y = np.array(y)
actual_labels = np.array(actual_labels)

# Distance functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

# Assigning clusters based on chosen distance metric
def assign_clusters(points, centroids, distance_function):
    clusters = []
    for point in points:
        distances = [distance_function(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Updating centroids by computing the mean of the assigned points
def update_centroids(points, clusters, k):
    new_centroids = []
    for cluster in range(k):
        cluster_points = points[clusters == cluster]
        new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

# K-Means algorithm with customizable distance function and initial centroids
def kmeans(points, k, initial_centroids, distance_function=euclidean_distance, max_iterations=100, tolerance=1e-4):
    centroids = initial_centroids
    for _ in range(max_iterations):

        clusters = assign_clusters(points, centroids, distance_function)
        new_centroids = update_centroids(points, clusters, k)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids

    return centroids, clusters

# Accuracy calculation function
def calculate_accuracy(predicted_clusters, actual_clusters):
    # Since K-Means cluster assignments can be permuted, we need to match them correctly
    from sklearn.metrics import accuracy_score
    from scipy.optimize import linear_sum_assignment

    # Create a confusion matrix to match predicted clusters to true labels
    max_label = max(predicted_clusters.max(), actual_clusters.max()) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)
    for true_label, pred_label in zip(actual_clusters, predicted_clusters):
        confusion_matrix[true_label, pred_label] += 1

    # Use Hungarian algorithm to find the best cluster matching
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    matched_clusters = np.zeros_like(predicted_clusters)
    for i, j in zip(row_ind, col_ind):
        matched_clusters[predicted_clusters == j] = i

    return accuracy_score(actual_clusters, matched_clusters)

points = np.array(list(zip(x, y)))

# Chose the clusters as 2
k = 2

# Step 1: Define three different sets of initial centroids
initial_centroids_1 = np.array([[2, 2], [1,-1]])  # Set 1
initial_centroids_2 = np.array([[3, 4], [2,-4]])   # Set 2
initial_centroids_3 = np.array([[5, 5], [6, 6]])   # Set 3

distance_metrics = {
    'Euclidean': euclidean_distance,
    'Manhattan': manhattan_distance,
    'Chebyshev': chebyshev_distance
}
#plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

centroid_sets = [initial_centroids_1, initial_centroids_2, initial_centroids_3]
centroid_labels = ['Centroids Set 1', 'Centroids Set 2', 'Centroids Set 3']

plot_idx = 0
accuracy_results = {}

for distance_name, distance_function in distance_metrics.items():
    accuracies = []
    for i, initial_centroids in enumerate(centroid_sets):

        centroids, clusters = kmeans(points, k, initial_centroids, distance_function=distance_function)


        accuracy = calculate_accuracy(clusters, actual_labels)
        accuracies.append(accuracy)

        # Plot the result
        axes[plot_idx].scatter(points[:, 0], points[:, 1], c=clusters, s=100, cmap='viridis', label='Points')
        axes[plot_idx].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        axes[plot_idx].set_title(f'{distance_name} Distance with {centroid_labels[i]} (Acc: {accuracy:.2f})')
        axes[plot_idx].set_xlabel('X-axis')
        axes[plot_idx].set_ylabel('Y-axis')
        axes[plot_idx].legend()

        plot_idx += 1

    accuracy_results[distance_name] = accuracies

plt.tight_layout()
plt.show()

# Printing out accuracies for each distance metric and initial centroid set
for distance_name, accuracies in accuracy_results.items():
    print(f"{distance_name} Distance:")
    for i, accuracy in enumerate(accuracies):
        print(f"  {centroid_labels[i]}: {accuracy:.2f}")