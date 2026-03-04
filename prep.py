import os
import glob
import re
import numpy as np
import ot
import pickle
from scipy.spatial.distance import cdist
from scipy.signal import convolve
from scipy.stats import zscore
from nilearn.maskers import NiftiLabelsMasker
import sys
import warnings

# Import BrainIAK
try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("[Fatal] brainiak is not installed. Please run pip install brainiak")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# ================= Configuration Class (Embedded directly for easy modification) =================
class Config:
    # --- Path Configuration ---
    DATA_ROOT = "/data/spacetop"
    ATLAS_PATH = "/home/zhuanghaojun/spacetop/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
    OUTPUT_ROOT = "/home/zhuanghaojun/spacetop"
    MODEL_ROOT = os.path.join(OUTPUT_ROOT, "models")
    IMG_ROOT = os.path.join(OUTPUT_ROOT, "images")
    MAPPING_FILE = os.path.join(OUTPUT_ROOT, "gw_mapping.npy")

    # --- Dataset Splitting ---
    # Training set: 0002 - 0016 (15 subjects in total)
    TRAIN_SUBJECTS = [f"sub-{i:04d}" for i in range(2, 17)]
    # Test set: 0017 - 0021 (5 subjects in total)
    TEST_SUBJECTS = [f"sub-{i:04d}" for i in range(17, 22)]

    ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

    # --- Task Configuration ---
    TASKS = {'social': 'social', 'video': 'alignvideo'}
    FILENAME_REGEX = re.compile(r"(sub-\d+)_(ses-\d+)_task-([a-zA-Z0-9]+)_.*?(run-\d+)_bold\.nii\.gz")

    # --- Parameters ---
    N_ROIS = 400
    IMG_SIZE = 20
    SMOOTH_WINDOW = 5


# ================= 1. Basic Tools =================

def scan_dataset():
    """Scan all files and return the index"""
    print(f"Scanning {Config.DATA_ROOT} ...")
    dataset_index = []

    # Scan everyone in the list
    for sub in Config.ALL_SUBJECTS:
        sub_dir = os.path.join(Config.DATA_ROOT, sub)
        if not os.path.exists(sub_dir): continue

        for root, _, files in os.walk(sub_dir):
            for file in files:
                if not file.endswith("bold.nii.gz"): continue
                match = Config.FILENAME_REGEX.search(file)
                if match:
                    f_sub, f_ses, f_task_label, f_run = match.groups()
                    if f_sub != sub: continue

                    found_key = None
                    for key, val in Config.TASKS.items():
                        if val in f_task_label or f_task_label in val:
                            found_key = key
                            break

                    if found_key:
                        dataset_index.append({
                            'sub': f_sub, 'ses': f_ses, 'task_key': found_key,
                            'task_label': f_task_label, 'run': f_run, 'path': os.path.join(root, file)
                        })

    # [Key] Sort: sub -> ses -> task -> run
    dataset_index.sort(key=lambda x: (x['sub'], x['ses'], x['task_key'], x['run']))
    print(f"Scan complete: Found {len(dataset_index)} files in total.")
    return dataset_index


def extract_signals_safe(masker, nii_path):
    """
    [Robust version] Extract signals and align to 400 dimensions.
    Combines quantity validation and precise Label ID placement to prevent misalignment.
    """
    try:
        # 1. Extract signals (Time, N_extracted)
        ts_extracted = masker.fit_transform(nii_path)
        T, N_curr = ts_extracted.shape

        # 2. Prepare standard container (400, Time)
        X_full = np.zeros((Config.N_ROIS, T))

        # 3. Process labels
        # masker.labels_ contains all labels in the atlas (usually including background 0)
        # We first filter out background 0 to get the "theoretically expected list of extracted labels"
        raw_labels = masker.labels_
        clean_labels = []

        for l in raw_labels:
            try:
                val = int(l)
                if val != 0: clean_labels.append(val)
            except ValueError:
                continue

        # --- [Your suggestion] Add quantity consistency check ---
        # Theoretically, the length of clean_labels should strictly equal the number of extracted data columns N_curr
        if len(clean_labels) != N_curr:
            print(f"  [Warning] Severe dimension mismatch: Atlas label count ({len(clean_labels)}) != Extracted data columns ({N_curr})")
            print(f"            File: {os.path.basename(nii_path)}")
            # This situation is very dangerous, forced assignment will cause complete misalignment, it is recommended to skip this file directly
            return None

        # 4. Precise placement (seat by number)
        # The index of enumerate(clean_labels) exactly corresponds to the column index of ts_extracted
        # label_val corresponds to the real brain region ID
        for col_idx, label_val in enumerate(clean_labels):

            # Schaefer atlas IDs are usually 1-400 -> array index 0-399
            target_row_idx = label_val - 1

            # Safety check: Ensure Label ID is within our defined 400-dimensional range
            if 0 <= target_row_idx < Config.N_ROIS:
                X_full[target_row_idx, :] = ts_extracted[:, col_idx]
            else:
                # This situation is rare unless the wrong atlas is used (e.g., 1000 dimensions)
                # print(f"  [Info] Ignoring out-of-range label: {label_val}")
                pass

        return X_full

    except Exception as e:
        print(f"  [Extract Error] {os.path.basename(nii_path)}: {e}")
        return None

def temporal_denoising(X):
    """Temporal denoising"""
    X = np.nan_to_num(X)
    X_denoised = np.zeros_like(X)
    for i in range(X.shape[0]):
        sig = X[i, :]
        if np.var(sig) == 0: continue
        sig_c = sig - sig.mean()
        acf = np.correlate(sig_c, sig_c, mode='full')[sig.size - 1:]
        if acf[0] != 0:
            acf /= acf[0]
            w = Config.SMOOTH_WINDOW
            acf_s = convolve(acf, np.ones(w) / w, mode='same')
            neg = np.where(acf_s < 0)[0]
            cut = neg[0] if len(neg) > 0 else 10
            weights = acf_s[:cut] / (acf_s[:cut].sum() + 1e-9)
            X_denoised[i, :] = convolve(sig, weights, mode='same')
    return X_denoised


# ================= 2. SRM Module (Find intersection + Align everyone) =================
def find_common_video_run(all_files):
    """
    [Core logic] Find a common (ses, run) combination among all subjects.
    It must be present in everyone to be used for SRM alignment.
    """
    video_files = [f for f in all_files if f['task_key'] == 'video']

    # Build the video set for each person: Set('ses-01_run-01', 'ses-01_run-02', ...)
    sub_runs = {}
    for f in video_files:
        key = f"{f['ses']}_{f['run']}"  # Combination key
        if f['sub'] not in sub_runs: sub_runs[f['sub']] = set()
        sub_runs[f['sub']].add(key)

    # Find intersection
    # We only care about the people defined in Config.ALL_SUBJECTS
    common_keys = None
    valid_subs_count = 0

    for sub in Config.ALL_SUBJECTS:
        if sub in sub_runs:
            if common_keys is None:
                common_keys = sub_runs[sub]
            else:
                common_keys = common_keys.intersection(sub_runs[sub])
            valid_subs_count += 1
        else:
            print(f"  [Warning] Subject {sub} has no Video data and cannot participate in SRM alignment.")

    if not common_keys:
        raise ValueError("No common Video (ses+run) found among all subjects! Cannot perform strict alignment.")

    # Sort and take the first one as the anchor (e.g., 'ses-01_run-01')
    best_key = sorted(list(common_keys))[0]
    target_ses, target_run = best_key.split('_')

    print(f"\n>>> [SRM Alignment Anchor] Found common Video: {target_ses} {target_run}")
    print(f"    Coverage of subjects: {valid_subs_count}/{len(Config.ALL_SUBJECTS)}")

    return target_ses, target_run


def get_or_train_srm(masker, all_files):
    """
    [Multi-core accelerated version] Train SRM
    Uses joblib to read data in parallel, significantly reducing waiting time.
    """
    srm_path = os.path.join(Config.MODEL_ROOT, "srm_model.pkl")
    sub_list_path = os.path.join(Config.MODEL_ROOT, "srm_subjects.pkl")
    os.makedirs(Config.MODEL_ROOT, exist_ok=True)

    if os.path.exists(srm_path) and os.path.exists(sub_list_path):
        print(">>> [SRM] Loading existing model...")
        with open(srm_path, 'rb') as f: srm = pickle.load(f)
        with open(sub_list_path, 'rb') as f: subs = pickle.load(f)
        return srm, subs

    # 1. Find common Video
    target_ses, target_run = find_common_video_run(all_files)
    print(f">>> [SRM] Starting parallel reading of training data (Anchor: {target_ses} {target_run})...")

    # --- Helper function: Processing logic for a single subject ---
    def process_one_subject(sub):
        # Find the specific run for this subject in all_files
        target_file = next((f for f in all_files if
                            f['sub'] == sub and
                            f['task_key'] == 'video' and
                            f['ses'] == target_ses and
                            f['run'] == target_run), None)

        if target_file is None:
            return None  # Missing this file

        # Extract signals
        X = extract_signals_safe(masker, target_file['path'])
        if X is None: return None

        # Z-score (Required for SRM)
        X_z = np.nan_to_num(zscore(X, axis=1))
        return (sub, X_z)

    # --- Start parallel processing ---
    # n_jobs=-1 means using all CPU cores (careful with memory overflow)
    # If server memory is limited, it is recommended to set n_jobs=4 or 8
    print(f"    Starting multi-core accelerated processing for {len(Config.ALL_SUBJECTS)} subjects...")

    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_subject)(sub) for sub in Config.ALL_SUBJECTS
    )

    # --- Organize results ---
    train_data = []
    training_subs = []
    min_time = np.inf

    for res in results:
        if res is None: continue
        sub_id, X_z = res
        train_data.append(X_z)
        training_subs.append(sub_id)
        min_time = min(min_time, X_z.shape[1])

    print(f"\n    Reading complete. Valid subjects: {len(training_subs)} (Truncated T={min_time})")

    if len(train_data) < 2:
        raise ValueError("Not enough valid training data (less than 2 subjects)!")

    # Truncate & Train
    data_aligned = [d[:, :min_time] for d in train_data]

    print("    Starting to train SRM model...")
    srm = SRM(n_iter=10, features=Config.N_ROIS)
    srm.fit(data_aligned)

    with open(srm_path, 'wb') as f:
        pickle.dump(srm, f)
    with open(sub_list_path, 'wb') as f:
        pickle.dump(training_subs, f)

    return srm, training_subs


# ================= 3. GW Module (Strictly limited to training set) =================
from joblib import Parallel, delayed


def compute_gw_mapping(srm_model, srm_subs, masker, all_files):
    """
    [Ultra-fast version] Calculate GW mapping
    1. Use multi-core parallel processing (Parallel).
    2. Only use the "common anchor video" from SRM alignment, no longer reading all videos, massively speeding up.
    """
    map_path = Config.MAPPING_FILE
    if os.path.exists(map_path):
        print(f">>> [GW] Loading existing mapping: {map_path}")
        return np.load(map_path)

    print(f"\n>>> [GW] Calculating global mapping (Strictly limited to training set 0002-0016)...")

    # 1. Determine training list (both in SRM and in Config.TRAIN_SUBJECTS)
    valid_train_subs = [s for s in Config.TRAIN_SUBJECTS if s in srm_subs]

    if not valid_train_subs:
        raise ValueError("No valid training set subjects to calculate GW mapping!")

    # 2. Retrieve the "common anchor" again (ses, run)
    # This way we only read this one file, which is the fastest and most stable
    target_ses, target_run = find_common_video_run(all_files)
    print(f"    Locking anchor video: {target_ses} {target_run} (Only using this file to calculate topology)")

    # --- Define processing function for a single subject (for parallelization) ---
    def process_one_sub_gw(sub):
        # Accurately find the anchor file for this subject
        target_file = next((f for f in all_files if
                            f['sub'] == sub and
                            f['task_key'] == 'video' and
                            f['ses'] == target_ses and
                            f['run'] == target_run), None)

        if target_file is None: return None

        # 1. Extract
        X = extract_signals_safe(masker, target_file['path'])
        if X is None: return None

        # 2. Project to shared space (SRM Transform)
        # S = W.T @ Zscore(X)
        try:
            sub_idx = srm_subs.index(sub)
            W = srm_model.w_[sub_idx]
            X_z = np.nan_to_num(zscore(X, axis=1))
            X_shared = W.T @ X_z
        except ValueError:
            return None

        # 3. Denoising
        X_den = temporal_denoising(X_shared)

        # 4. Calculate correlation matrix (400x400)
        # This is the core input for GW
        corr = np.corrcoef(X_den)
        return np.nan_to_num(corr)

    # --- Start parallel calculation ---
    print(f"    Starting multi-core acceleration (n_jobs=8)...")
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_sub_gw)(sub) for sub in valid_train_subs
    )

    # --- Aggregate results ---
    sum_corr = np.zeros((Config.N_ROIS, Config.N_ROIS))
    count = 0

    for res in results:
        if res is not None:
            sum_corr += res
            count += 1

    if count == 0:
        raise ValueError("GW calculation failed: No valid correlation matrices extracted.")

    mean_corr = sum_corr / count
    print(f"\n    GW correlation matrix calculation complete (Based on {count} subjects).")

    # --- Solve Optimal Transport (This step is fast, usually a few seconds) ---
    print("    Solving Optimal Transport (Gromov-Wasserstein)...")

    # Source domain structure (1 - Corr)
    C1 = 1.0 - mean_corr
    C1 /= (C1.max() + 1e-9)

    # Target domain structure (2D Grid)
    sz = Config.IMG_SIZE
    coords = np.array([(i, j) for i in range(sz) for j in range(sz)])
    C2 = cdist(coords, coords, metric='euclidean')
    C2 /= (C2.max() + 1e-9)

    p = ot.unif(Config.N_ROIS)
    q = ot.unif(sz * sz)

    # Calculate GW mapping
    gw_map = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-4)

    # Normalize
    row_sums = gw_map.sum(axis=1, keepdims=True)
    gw_map_norm = gw_map / (row_sums + 1e-9)

    np.save(map_path, gw_map_norm)
    print(f"    GW mapping saved to {map_path}")
    return gw_map_norm


# ================= 4. Main Pipeline =================

def run_prep():
    if not os.path.exists(Config.ATLAS_PATH):
        print(f"[Fatal] Atlas not found: {Config.ATLAS_PATH}")
        sys.exit(1)
    masker = NiftiLabelsMasker(labels_img=Config.ATLAS_PATH, standardize=True, verbose=0)

    # 1. Scan
    all_files = scan_dataset()
    if not all_files: sys.exit(1)

    # 2. SRM Alignment (All members participate to ensure everyone has a W matrix)
    srm_model, srm_subs = get_or_train_srm(masker, all_files)

    # 3. GW Mapping (Only training set participates)
    S_global = compute_gw_mapping(srm_model, srm_subs, masker, all_files)

    # -------------------------------------------------
    # 4. Generate Images (Multi-core parallel ultra-fast version)
    # -------------------------------------------------
    print(f"\n>>> [Pipeline] Starting multi-core acceleration (n_jobs=8) to process all data (Train + Test)...")

    # Define processing logic for a single file
    def process_one_file_final(item):
        # 1. Path preparation
        sub_out_dir = os.path.join(Config.IMG_ROOT, item['sub'])
        os.makedirs(sub_out_dir, exist_ok=True)
        fname = f"{item['sub']}_{item['ses']}_{item['task_key']}_{item['run']}_bcne.npy"
        save_path = os.path.join(sub_out_dir, fname)

        # 2. Check if already exists
        if os.path.exists(save_path):
            return None  # Skip

        # 3. Check if subject is in the SRM list
        if item['sub'] not in srm_subs:
            return None  # Skip

        # 4. Extract signals (Most time-consuming step)
        # Note: This is in a child process, print might not show immediately, so rely on joblib's progress bar
        X = extract_signals_safe(masker, item['path'])
        if X is None: return None

        try:
            # 5. Project (SRM Transform)
            sub_idx = srm_subs.index(item['sub'])
            W = srm_model.w_[sub_idx]

            # Z-score + Project
            X_z = np.nan_to_num(zscore(X, axis=1))
            X_shared = W.T @ X_z

            # 6. Denoising
            X_den = temporal_denoising(X_shared)

            # 7. Mapping (GW Project)
            # S_global.T (Pixels, Feats) @ X (Feats, Time) -> (Pixels, Time)
            mapped = S_global.T @ X_den

            # 8. Save
            T = mapped.shape[1]
            # Reshape -> (Time, Channel=1, H, W)
            img = mapped.T.reshape(T, 1, Config.IMG_SIZE, Config.IMG_SIZE)
            np.save(save_path, img)
            return item['sub']  # Return successfully processed subject name for statistics

        except Exception as e:
            return None

    # --- Start parallel task ---
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_file_final)(item) for item in all_files
    )

    print("\n\n>>> All done! All images have been generated.")


if __name__ == "__main__":
    run_prep()