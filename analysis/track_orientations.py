import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from pathlib import Path
from scipy.spatial.distance import cdist

# --- SETTINGS ---
image_folder = Path("rotation_test")      # subfolder with images
min_area = 50000                      # ignore small blobs
max_dist = 30                      # max jump for centroid match
max_objects = 10                   # number of objects to track

# --- LOAD IMAGE PATHS ---
image_paths = sorted(image_folder.glob("*.jpg"))
if not image_paths:
    raise FileNotFoundError(f"No .jpg files found in: {image_folder}")

# --- TRACKING STATE ---
object_tracks = {}  # {obj_id: [(x, y, angle)]}
prev_centroids = None

# --- PROCESS FRAMES ---
for t, path in enumerate(image_paths):
    image = io.imread(str(path))
    gray = color.rgb2gray(image)
    binary = gray > filters.threshold_otsu(gray)

    # Remove small objects
    labeled = measure.label(binary)
    cleaned = morphology.remove_small_objects(labeled, min_size=min_area)
    binary_clean = cleaned > 0

    # Get regions from cleaned binary image
    labeled_clean = measure.label(binary_clean)
    regions = measure.regionprops(labeled_clean)

    current_objects = []
    for region in regions:
        y, x = region.centroid
        angle = np.degrees(region.orientation)
        current_objects.append(((x, y), angle))

    matched_ids = {}

    # Initialize tracking
    if t == 0:
        for i, (centroid, angle) in enumerate(current_objects[:max_objects]):
            object_tracks[i] = [(centroid[0], centroid[1], angle)]
            matched_ids[i] = (centroid[0], centroid[1], angle)
        prev_centroids = [c for c, _ in current_objects]

    else:
        new_centroids = [c for c, _ in current_objects]
        new_angles = [a for _, a in current_objects]

        if prev_centroids and new_centroids:
            dist_matrix = cdist(prev_centroids, new_centroids)
            row_ind, col_ind = np.where(dist_matrix < max_dist)
            used_cols = set()

            for i, j in zip(row_ind, col_ind):
                if i in object_tracks and j not in used_cols:
                    x, y = new_centroids[j]
                    angle = new_angles[j]
                    object_tracks[i].append((x, y, angle))
                    matched_ids[i] = (x, y, angle)
                    used_cols.add(j)

        prev_centroids = new_centroids

    # --- PLOT BINARIZED IMAGE WITH OVERLAYS ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_clean, cmap='gray')
    ax.set_title(f"Binarized + Overlays (Frame {t+1})")

    for obj_id, data in matched_ids.items():
        x, y, angle = data

        # Centroid
        ax.plot(x, y, 'go', markersize=5)

        # Major axis
        rad = np.radians(angle)
        length = 40  # adjust axis length
        dx = np.cos(rad) * length
        dy = -np.sin(rad) * length
        ax.plot([x - dx, x + dx], [y - dy, y + dy], 'r-', linewidth=2)

        # Object ID label
        ax.text(x + 5, y + 5, f"ID {obj_id}", color='yellow', fontsize=9, weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- PLOT ORIENTATION TRACKS ---
plt.figure(figsize=(10, 6))
for obj_id, data in object_tracks.items():
    angles = [entry[2] for entry in data]
    plt.plot(angles, label=f'Object {obj_id}')
plt.title("Major Axis Orientation Over Time")
plt.xlabel("Frame")
plt.ylabel("Orientation (degrees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()