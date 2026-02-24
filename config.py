import os
import re


class Config:
    # ================= 路径配置 =================
    # 请确保这里是存放原始 .nii.gz 文件的根目录
    DATA_ROOT = "/data/spacetop"

    ATLAS_PATH = "/home/zhuanghaojun/spacetop/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"

    OUTPUT_ROOT = "/home/zhuanghaojun/spacetop"

    MODEL_ROOT = os.path.join(OUTPUT_ROOT, "models")

    # 其他输出路径
    IMG_ROOT = os.path.join(OUTPUT_ROOT, "bcne_images")
    RESULT_ROOT = os.path.join(OUTPUT_ROOT, "manifolds")
    MAPPING_FILE = os.path.join(OUTPUT_ROOT, "gw_mapping.npy")

    # ================= 实验参数 =================
    # 被试列表 (sub-0002 到 sub-0021)
    # 训练集: 0002 - 0016 (共15人) -> 用于计算 GW 映射
    TRAIN_SUBJECTS = [f"sub-{i:04d}" for i in range(2, 17)]
    # 测试集: 0017 - 0021 (共5人) -> 只做变换，不参与映射计算
    TEST_SUBJECTS = [f"sub-{i:04d}" for i in range(17, 22)]

    # 全部被试 (SRM 需要对齐所有人)
    ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

    # 任务名映射
    # Key: 代码中使用的简写 (必须包含 'video' 用于训练 SRM)
    # Value: 文件名中实际出现的字符串
    TASKS = {
        'social': 'social',
        'video': 'alignvideo'  # Spacetop 的 video 任务通常叫 alignvideo
    }

    # ================= 正则表达式配置 =================
    # 用于从文件名提取元数据
    # 适配文件名如: sub-0002_ses-01_task-alignvideo_acq-mb8_run-01_bold.nii.gz
    FILENAME_REGEX = re.compile(r"(sub-\d+)_(ses-\d+)_task-([a-zA-Z0-9]+)_.*?(run-\d+)_bold\.nii\.gz")

    # ================= BCNE 参数 =================
    N_ROIS = 400
    IMG_SIZE = 20
    SMOOTH_WINDOW = 5
    TR = 0.46

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    RECURSION_DEPTH = 3

    LATENT_DIM = 3