import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_point_cloud(file_path):
    print("Loading CSV file...")
    df = pd.read_csv(file_path)
    points = df[['x', 'y', 'z']].to_numpy()
    return points

def detect_ellipses(points, min_axis_length=0.03):
    lower_percentile = 91
    upper_percentile = 97
    z_lower = np.percentile(points[:, 2], lower_percentile)
    z_upper = np.percentile(points[:, 2], upper_percentile)
    edge_points = points[(points[:, 2] >= z_lower) & (points[:, 2] <= z_upper)]

    print(f"Using {len(edge_points)} Z-range filtered edge points for clustering...")

    clustering = DBSCAN(eps=0.03, min_samples=3).fit(edge_points[:, :2])
    labels = clustering.labels_
    unique_labels = set(labels)
    ellipses = []

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = edge_points[labels == label, :2]
        cluster_points = np.unique(cluster_points, axis=0)

        if len(cluster_points) < 200:
            continue

        try:
            model = EllipseModel()
            if not model.estimate(cluster_points):
                print(f"Ellipse fitting failed on cluster {label}")
                continue
            xc, yc, a, b, theta = model.params

            # 최소 반축 길이 기준으로 필터링
            if min(a, b) < min_axis_length:
                print(f"Ellipse {label} rejected: axis too small (a={a:.3f}, b={b:.3f})")
                continue

            ellipses.append((xc, yc, a, b, theta))
        except Exception as e:
            print(f"Ellipse fitting failed on cluster {label}: {e}")
            continue

    return ellipses, edge_points

def compute_normal_vector(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[-1]
    return normal_vector

# def plot_point_cloud_3d(points, ellipses):
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(point_cloud)
#     vis.run()
#     vis.destroy_window()

def plot_point_cloud_3d(points):
    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='cividis', s=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud with Height-based Color')
    cbar = plt.colorbar(sc)
    cbar.set_label('Z value (height)')
    plt.axis('equal')
    plt.show()

def angle_with_z_axis(vector):
    vector = np.array(vector)
    z_axis = np.array([0, 0, 1])
    vector_norm = np.linalg.norm(vector)
    z_axis_norm = np.linalg.norm(z_axis)
    dot_product = np.dot(vector, z_axis)
    cos_theta = dot_product / (vector_norm * z_axis_norm)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def plot_detected_ellipses(points, edge_points, ellipses):
    plt.figure(figsize=(8,8))
    plt.scatter(points[:,0], points[:,1], s=1, c='blue', label="All Points")
    plt.scatter(edge_points[:,0], edge_points[:,1], s=1, c='orange', label="Filtered Z Points")
    ax = plt.gca()
    for (xc, yc, a, b, theta) in ellipses:
        ellipse_patch = patches.Ellipse((xc, yc), width=2*a, height=2*b,
                                        angle=np.degrees(theta), edgecolor='red', facecolor='none', lw=2, label="Detected Ellipse")
        ax.add_patch(ellipse_patch)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Detected Ellipses on XY Plane")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.axis("equal")
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select the CSV file", filetypes=[["CSV files", "*.csv"]])

    if not file_path:
        print("No file selected. Exiting.")
        return

    points = load_point_cloud(file_path)
    ellipses, edge_points = detect_ellipses(points, min_axis_length=0.6)

    print(f"Detected {len(ellipses)} ellipses.")
    for i, (xc, yc, a, b, theta) in enumerate(ellipses):
        print(f"Ellipse {i+1}: Center=({xc:.3f}, {yc:.3f}), a={a:.3f}, b={b:.3f}, angle={np.degrees(theta):.2f}°")

    # plot_point_cloud_3d(points)
    plot_detected_ellipses(points, edge_points, ellipses)

if __name__ == "__main__":
    main()
