#Feature Extraction: Shape Features and HoG Texture Features
#Tools used in this file:
#Tool: GitHub CoPilot, for code completion
#for example function body, variable and function names

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.feature import hog
from skimage import exposure

#paths
BASE   = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(BASE, 'parachute', 'parachute', 'GT')
IMG_DIR = os.path.join(BASE, 'parachute', 'parachute', 'images')
OUT_DIR = os.path.join(BASE, 'output_features')
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 51
PARACHUTE_LABEL = 255   # verified: label 255 is the parachute in every GT frame



#shape feature extraction


def load_mask(frame_idx: int) -> np.ndarray:
    """Return binary mask (uint8, 0/255) of the parachute region."""
    path = os.path.join(GT_DIR, f'parachute_{frame_idx:05d}.png')
    gt   = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(path)
    return (gt == PARACHUTE_LABEL).astype(np.uint8) * 255


def compute_shape_features(mask: np.ndarray) -> dict:
    """
    Compute four shape descriptors for the largest connected component.

    Solidity          = A / A_convex
    Non-compactness   = P^2 / (4π·A)   [1 = perfect circle; >1 more complex]
    Circularity       = 4π·A / P^2     [inverse of non-compactness; 1 = circle]
    Eccentricity      = √(1 - (b/a)²)  derived from second-order central moments
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return dict(solidity=np.nan, non_compactness=np.nan,
                    circularity=np.nan, eccentricity=np.nan, area=0)

    #use largest contour (parachute canopy)
    c = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)

    #convex hull area for solidity
    hull         = cv2.convexHull(c)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else np.nan

    #compactness based measures
    if perimeter > 0 and area > 0:
        non_compactness = (perimeter ** 2) / (4 * np.pi * area)
        circularity     = (4 * np.pi * area) / (perimeter ** 2)
    else:
        non_compactness = np.nan
        circularity     = np.nan

    #eccentricity via image moments
    M = cv2.moments(c)
    if M['m00'] > 0:
        #second order central moments, covariance matrix eigenvalues
        mu20 = M['mu20'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        mu11 = M['mu11'] / M['m00']
        diff  = mu20 - mu02
        trace = mu20 + mu02
        lam1  = (trace + np.sqrt(diff**2 + 4*mu11**2)) / 2
        lam2  = (trace - np.sqrt(diff**2 + 4*mu11**2)) / 2
        lam2  = max(lam2, 0.0)   # numerical safety
        eccentricity = np.sqrt(1.0 - lam2/lam1) if lam1 > 0 else 0.0
    else:
        eccentricity = np.nan

    return dict(solidity=solidity, non_compactness=non_compactness,
                circularity=circularity, eccentricity=eccentricity,
                area=area)


def extract_all_shape_features() -> dict:
    results = {k: [] for k in ['frame', 'solidity', 'non_compactness',
                                'circularity', 'eccentricity', 'area']}
    for i in range(N_FRAMES):
        mask = load_mask(i)
        feats = compute_shape_features(mask)
        results['frame'].append(i)
        for k in ['solidity', 'non_compactness', 'circularity', 'eccentricity', 'area']:
            results[k].append(feats[k])
    return {k: np.array(v) for k, v in results.items()}


def plot_shape_features(sf: dict) -> None:
    frames = sf['frame']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Shape Features vs. Frame Index', fontsize=14, fontweight='bold')

    pairs = [
        ('solidity',        'Solidity',         'steelblue'),
        ('non_compactness', 'Non-Compactness',   'firebrick'),
        ('circularity',     'Circularity',       'seagreen'),
        ('eccentricity',    'Eccentricity',      'darkorange'),
    ]
    for ax, (key, label, colour) in zip(axes.flat, pairs):
        ax.plot(frames, sf[key], color=colour, linewidth=1.8, marker='o',
                markersize=3, label=label)
        ax.set_xlabel('Frame index', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, N_FRAMES-1])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'shape_features.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def print_shape_stats(sf: dict) -> None:
    print('\n=== Shape Feature Statistics ===')
    for key in ['solidity', 'non_compactness', 'circularity', 'eccentricity']:
        vals = sf[key]
        print(f'{key:20s}  mean={np.nanmean(vals):.4f}  '
              f'std={np.nanstd(vals):.4f}  '
              f'min={np.nanmin(vals):.4f}  '
              f'max={np.nanmax(vals):.4f}')



#HoG Texture Feature Extraction

def load_gray_image(frame_idx: int) -> np.ndarray:
    path = os.path.join(IMG_DIR, f'parachute_{frame_idx:05d}.png')
    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def compute_hog_at_angle(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Compute a 1-D HoG feature vector restricted to a single gradient orientation.

    Strategy
    --------
    1. Compute x/y gradients with a Sobel kernel.
    2. Derive per-pixel magnitude and orientation (unsigned: 0-180 deg).
    3. Build a soft-binned histogram limited to orientations within ±22.5 deg of
       the target angle (one 45 deg-wide bin centred on the target).  This yields
       the gradient energy aligned with each of the four requested directions.
    4. Additionally compute a full 9-bin unsigned HoG descriptor (scikit-image)
       for the complete feature vector; then extract the two bins nearest the
       target angle as a compact directional descriptor.

    Returns a 1-D array of per-cell HoG magnitudes for the requested angle,
    concatenated to a global scalar (mean energy in that direction).
    """
    #full HoG descriptor: 9 orientations, 0–180 deg, unsigned
    hog_vec, hog_img = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
        channel_axis=None,
    )

    #identify which of the 9 bins (each 20 deg wide) best matches the target angle
    bin_centres = np.arange(9) * 20 + 10   #10, 30, 50, … 170
    # Wrap 135 deg → works in unsigned domain
    diffs = np.abs(bin_centres - (angle_deg % 180))
    diffs = np.minimum(diffs, 180 - diffs)
    nearest_bin = int(np.argmin(diffs))

    #extract entries for that bin from the block-normalised descriptor
    #each block of 4 cells × 9 orientations contributes 36 values.
    #stride through descriptor picking index `nearest_bin` from every cell.
    n_orientations = 9
    n_cells_per_block_dim = 2
    cells_per_block = n_cells_per_block_dim ** 2   # 4
    block_size = cells_per_block * n_orientations  # 36

    n_blocks = len(hog_vec) // block_size
    directional_vals = []
    for b in range(n_blocks):
        block = hog_vec[b * block_size: (b+1) * block_size]
        # 4 cells, each contributing `nearest_bin`-th orientation
        for cell in range(cells_per_block):
            directional_vals.append(block[cell * n_orientations + nearest_bin])

    directional_arr = np.array(directional_vals, dtype=np.float32)
    return directional_arr, hog_vec, hog_img


HOG_ANGLES = [0, 45, 90, 135]

def extract_all_hog_features() -> dict:
    """
    Returns dict with keys '0', '45', '90', '135', each containing an
    (N_FRAMES x D) array of directional HoG vectors, plus 'full_hog'
    containing the full (N_FRAMES x D_full) HoG matrix.
    """
    angle_vecs   = {a: [] for a in HOG_ANGLES}
    full_hog_mat = []

    for i in range(N_FRAMES):
        gray = load_gray_image(i)
        for a in HOG_ANGLES:
            dv, full_hog, _ = compute_hog_at_angle(gray, float(a))
            angle_vecs[a].append(dv)
        # Store full HoG once (same for every angle call)
        full_hog_mat.append(full_hog)

    return {
        **{a: np.array(angle_vecs[a]) for a in HOG_ANGLES},
        'full_hog': np.array(full_hog_mat)
    }


def plot_hog_mean_energy(hog_feats: dict) -> None:
    """Plot mean HoG energy per frame for each angle."""
    fig, ax = plt.subplots(figsize=(11, 5))
    colours = ['steelblue', 'firebrick', 'seagreen', 'darkorange']
    frames  = np.arange(N_FRAMES)
    for colour, a in zip(colours, HOG_ANGLES):
        mean_energy = hog_feats[a].mean(axis=1)
        ax.plot(frames, mean_energy, color=colour, linewidth=1.8,
                marker='o', markersize=2.5, label=f'{a} deg')

    ax.set_xlabel('Frame index', fontsize=11)
    ax.set_ylabel('Mean directional HoG energy', fontsize=11)
    ax.set_title('HoG Directional Energy per Frame (0, 45, 90, 135 deg)',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Gradient angle', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, N_FRAMES-1])

    out = os.path.join(OUT_DIR, 'hog_energy_per_frame.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_hog_visualisation(frame_indices=(0, 10, 25, 40, 50)) -> None:
    """Show HoG glyph visualisations for a selection of frames."""
    fig, axes = plt.subplots(len(frame_indices), 2,
                             figsize=(10, 3.5 * len(frame_indices)))
    fig.suptitle('HoG Visualisations (selected frames)', fontsize=12,
                 fontweight='bold')

    for row, idx in enumerate(frame_indices):
        gray = load_gray_image(idx)
        _, _, hog_img = compute_hog_at_angle(gray, 0.0)   # full HoG image
        hog_rescaled  = exposure.rescale_intensity(hog_img, in_range=(0, 0.2))

        axes[row, 0].imshow(gray, cmap='gray')
        axes[row, 0].set_title(f'Frame {idx} – grayscale', fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(hog_rescaled, cmap='inferno')
        axes[row, 1].set_title(f'Frame {idx} – HoG', fontsize=9)
        axes[row, 1].axis('off')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'hog_visualisations.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_hog_heatmap(hog_feats: dict) -> None:
    """Heatmap of mean directional energy across frames for each angle."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle('HoG Directional Energy Heatmap (frames × feature components)',
                 fontsize=11, fontweight='bold')

    for ax, a in zip(axes, HOG_ANGLES):
        mat = hog_feats[a]      # (N_FRAMES, D)
        im  = ax.imshow(mat, aspect='auto', cmap='viridis',
                        interpolation='nearest')
        ax.set_title(f'{a} deg', fontsize=11)
        ax.set_xlabel('Feature index', fontsize=9)
        if a == 0:
            ax.set_ylabel('Frame index', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'hog_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_shape_sample_masks(frame_indices=(0, 10, 25, 40, 50)) -> None:
    """Show GT masks overlaid on images for a sample of frames."""
    fig, axes = plt.subplots(1, len(frame_indices),
                             figsize=(3 * len(frame_indices), 3.5))
    fig.suptitle('Parachute GT Masks (selected frames)', fontsize=11,
                 fontweight='bold')

    for ax, idx in zip(axes, frame_indices):
        img  = cv2.cvtColor(
            cv2.imread(os.path.join(IMG_DIR, f'parachute_{idx:05d}.png')),
            cv2.COLOR_BGR2RGB)
        mask = load_mask(idx)
        overlay = img.copy()
        overlay[mask == 255] = [255, 80, 0]
        ax.imshow(overlay)
        ax.set_title(f'Frame {idx}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'sample_masks.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def print_hog_stats(hog_feats: dict) -> None:
    print('\n=== HoG Feature Statistics (mean directional energy) ===')
    for a in HOG_ANGLES:
        mean_e = hog_feats[a].mean(axis=1)
        print(f'  {a:3d} deg  mean={mean_e.mean():.5f}  '
              f'std={mean_e.std():.5f}  '
              f'min={mean_e.min():.5f}  '
              f'max={mean_e.max():.5f}')


def _mean_orientation_histogram(full_hog_vec: np.ndarray) -> np.ndarray:
    """
    Compute the mean energy per orientation bin across all blocks/cells.
    Returns a 9-element array (bins centred at 10, 30, ..., 170 degrees).
    """
    n_orientations  = 9
    cells_per_block = 4   # 2x2 block
    block_size      = cells_per_block * n_orientations   # 36
    n_blocks        = len(full_hog_vec) // block_size
    reshaped        = full_hog_vec[:n_blocks * block_size].reshape(
                          n_blocks, cells_per_block, n_orientations)
    return reshaped.mean(axis=(0, 1))   # (9,)


def plot_hog_orientation_histograms(full_hog_mat: np.ndarray,
                                    frame_indices=(0, 10, 25, 40, 50)) -> None:
    """
    Plot the full 9-bin mean orientation histogram for selected frames.
    Shows the complete per-image feature vector structure.
    """
    bin_centres = np.arange(9) * 20 + 10   # 10, 30, ..., 170 deg
    colours     = plt.cm.viridis(np.linspace(0.1, 0.85, len(frame_indices)))

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(14, 4), sharey=True)
    fig.suptitle(
        'Full HoG Orientation Distribution per Frame\n'
        '(mean energy per 20-degree bin, averaged over all normalised blocks)',
        fontsize=11, fontweight='bold')

    for ax, idx, col in zip(axes, frame_indices, colours):
        hist = _mean_orientation_histogram(full_hog_mat[idx])
        ax.bar(bin_centres, hist, width=16, color=col,
               edgecolor='white', linewidth=0.5)
        ax.set_title(f'Frame {idx}', fontsize=9)
        ax.set_xlabel('Bin centre (deg)', fontsize=8)
        ax.set_xticks(bin_centres)
        ax.set_xticklabels([f'{b}' for b in bin_centres], fontsize=7, rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        if idx == frame_indices[0]:
            ax.set_ylabel('Mean bin energy', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'hog_orientation_histograms.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_hog_vector_similarity(full_hog_mat: np.ndarray) -> None:
    """
    Plot (1) cosine similarity of each frame's full HoG vector to frame 0,
    and (2) cosine similarity between consecutive frames.
    Quantifies how the full feature vector drifts across the sequence.
    """
    norms = np.linalg.norm(full_hog_mat, axis=1, keepdims=True)
    normed = full_hog_mat / np.maximum(norms, 1e-12)

    sim_to_0   = normed @ normed[0]                        # (N,)
    sim_consec = (normed[1:] * normed[:-1]).sum(axis=1)    # (N-1,)

    print('\n=== HoG Vector Similarity ===')
    print(f'  Similarity to frame 0:  mean={sim_to_0.mean():.4f}  '
          f'min={sim_to_0.min():.4f}  final={sim_to_0[-1]:.4f}')
    print(f'  Consecutive similarity: mean={sim_consec.mean():.4f}  '
          f'min={sim_consec.min():.4f}')

    frames = np.arange(N_FRAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('HoG Feature Vector Similarity Across Frames',
                 fontsize=11, fontweight='bold')

    axes[0].plot(frames, sim_to_0, color='steelblue', linewidth=1.8,
                 marker='o', markersize=3)
    axes[0].set_xlabel('Frame index', fontsize=10)
    axes[0].set_ylabel('Cosine similarity to frame 0', fontsize=10)
    axes[0].set_title('Cumulative drift from initial appearance', fontsize=10)
    axes[0].set_ylim([min(sim_to_0.min() - 0.01, 0.89), 1.01])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, N_FRAMES), sim_consec, color='seagreen',
                 linewidth=1.8, marker='o', markersize=3)
    axes[1].set_xlabel('Frame index', fontsize=10)
    axes[1].set_ylabel('Cosine similarity to previous frame', fontsize=10)
    axes[1].set_title('Frame-to-frame HoG consistency', fontsize=10)
    axes[1].set_ylim([min(sim_consec.min() - 0.01, 0.97), 1.005])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'hog_vector_similarity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


#main
if __name__ == '__main__':
    print('--- Feature Extraction -------------------------------------------')

    #shape features
    print('\n[1/4] Computing shape features ...')
    sf = extract_all_shape_features()
    print_shape_stats(sf)
    plot_shape_features(sf)
    plot_shape_sample_masks()

    # Save numerical results
    csv_path = os.path.join(OUT_DIR, 'shape_features.csv')
    header   = 'frame,solidity,non_compactness,circularity,eccentricity,area'
    data     = np.column_stack([sf['frame'], sf['solidity'], sf['non_compactness'],
                                sf['circularity'], sf['eccentricity'], sf['area']])
    np.savetxt(csv_path, data, delimiter=',', header=header,
               comments='', fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.1f'])
    print(f'Saved: {csv_path}')

    #HoG features
    print('\n[2/4] Computing HoG features ...')
    hog_feats = extract_all_hog_features()
    print_hog_stats(hog_feats)

    print('\n[3/4] Plotting HoG features ...')
    plot_hog_mean_energy(hog_feats)
    plot_hog_visualisation()
    plot_hog_heatmap(hog_feats)
    plot_hog_orientation_histograms(hog_feats['full_hog'])
    plot_hog_vector_similarity(hog_feats['full_hog'])

    print('\n[4/4] All feature extraction complete.')
    print(f'Output files are in: {OUT_DIR}')
