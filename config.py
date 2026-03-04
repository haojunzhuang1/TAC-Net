import os
import re


class Config:
    # Input and Output paths
    DATA_ROOT = "/data/spacetop"

    ATLAS_PATH = "/home/zhuanghaojun/spacetop/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"

    OUTPUT_ROOT = "/home/zhuanghaojun/spacetop"

    MODEL_ROOT = os.path.join(OUTPUT_ROOT, "models")

    # Other paths
    IMG_ROOT = os.path.join(OUTPUT_ROOT, "images")
    RESULT_ROOT = os.path.join(OUTPUT_ROOT, "manifolds")
    MAPPING_FILE = os.path.join(OUTPUT_ROOT, "gw_mapping.npy")

    # ================= Experimental Parameters =================
    # Subject list (sub-0002 to sub-0021)
    # Training set: 0002 - 0016 (15 subjects in total)
    TRAIN_SUBJECTS = [f"sub-{i:04d}" for i in range(2, 17)]
    # Test set: 0017 - 0021 (5 subjects in total)
    TEST_SUBJECTS = [f"sub-{i:04d}" for i in range(17, 22)]

    ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

    # Key: Abbreviation used in the code (must contain 'video' for training SRM)
    # Value: The actual string appearing in the filename
    TASKS = {
        'social': 'social',
        'video': 'alignvideo'  # Spacetop's video task is usually called alignvideo
    }

    # ================= Regular Expression Configuration =================
    # Used to extract metadata from filenames
    # Adapts to filenames like: sub-0002_ses-01_task-alignvideo_acq-mb8_run-01_bold.nii.gz
    FILENAME_REGEX = re.compile(r"(sub-\d+)_(ses-\d+)_task-([a-zA-Z0-9]+)_.*?(run-\d+)_bold\.nii\.gz")

    # ================= Spacetop Parameters =================
    N_ROIS = 400
    IMG_SIZE = 20
    SMOOTH_WINDOW = 5
    TR = 0.46

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    RECURSION_DEPTH = 3

    LATENT_DIM = 3