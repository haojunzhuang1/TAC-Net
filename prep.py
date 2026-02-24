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
from joblib import Parallel, delayed

# 引入 BrainIAK
try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("[Fatal] 未安装 brainiak。请运行 pip install brainiak")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# ================= 配置类 (直接内嵌，方便你修改) =================
class Config:
    # --- 路径配置 ---
    DATA_ROOT = "/data/spacetop"
    ATLAS_PATH = "/home/zhuanghaojun/spacetop/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
    OUTPUT_ROOT = "/home/zhuanghaojun/spacetop"
    MODEL_ROOT = os.path.join(OUTPUT_ROOT, "models")
    IMG_ROOT = os.path.join(OUTPUT_ROOT, "bcne_images")
    MAPPING_FILE = os.path.join(OUTPUT_ROOT, "gw_mapping.npy")

    # --- 数据集划分 ---
    # 训练集: 0002 - 0016 (共15人) -> 用于计算 GW 映射
    TRAIN_SUBJECTS = [f"sub-{i:04d}" for i in range(2, 17)]
    # 测试集: 0017 - 0021 (共5人) -> 只做变换，不参与映射计算
    TEST_SUBJECTS = [f"sub-{i:04d}" for i in range(17, 22)]

    # 全部被试 (SRM 需要对齐所有人)
    ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

    # --- 任务配置 ---
    TASKS = {'social': 'social', 'video': 'alignvideo'}
    FILENAME_REGEX = re.compile(r"(sub-\d+)_(ses-\d+)_task-([a-zA-Z0-9]+)_.*?(run-\d+)_bold\.nii\.gz")

    # --- 参数 ---
    N_ROIS = 400
    IMG_SIZE = 20
    SMOOTH_WINDOW = 5


# ================= 1. 基础工具 =================

def scan_dataset():
    """扫描所有文件，返回索引"""
    print(f"正在扫描 {Config.DATA_ROOT} ...")
    dataset_index = []

    # 扫描列表里的所有人
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

    # [关键] 排序: sub -> ses -> task -> run
    dataset_index.sort(key=lambda x: (x['sub'], x['ses'], x['task_key'], x['run']))
    print(f"扫描完成: 共找到 {len(dataset_index)} 个文件。")
    return dataset_index


def extract_signals_safe(masker, nii_path):
    """
    [健壮版] 提取信号并对齐至 400 维。
    结合了数量校验和 Label ID 精准归位，防止错位。
    """
    try:
        # 1. 提取信号 (Time, N_extracted)
        ts_extracted = masker.fit_transform(nii_path)
        T, N_curr = ts_extracted.shape

        # 2. 准备标准容器 (400, Time)
        X_full = np.zeros((Config.N_ROIS, T))

        # 3. 处理标签
        # masker.labels_ 包含了图谱里的所有标签（通常包括背景 0）
        # 我们先过滤掉背景 0，得到“理论上应该提取出的标签列表”
        raw_labels = masker.labels_
        clean_labels = []

        for l in raw_labels:
            try:
                val = int(l)
                if val != 0: clean_labels.append(val)
            except ValueError:
                continue

        # --- [你的建议] 增加数量一致性校验 ---
        # 理论上，clean_labels 的长度应该严格等于提取出的数据列数 N_curr
        if len(clean_labels) != N_curr:
            print(f"  [Warning] 维度严重不匹配: 图谱标签数({len(clean_labels)}) != 提取数据列数({N_curr})")
            print(f"            文件: {os.path.basename(nii_path)}")
            # 这种情况非常危险，强行赋值会导致全部错位，建议直接跳过该文件
            return None

        # 4. 精准归位 (对号入座)
        # enumerate(clean_labels) 的 index 正好对应 ts_extracted 的列索引
        # label_val 对应真实的脑区 ID
        for col_idx, label_val in enumerate(clean_labels):

            # Schaefer 图谱 ID 通常是 1-400 -> 数组索引 0-399
            target_row_idx = label_val - 1

            # 安全检查：确保 Label ID 在我们定义的 400 维范围内
            if 0 <= target_row_idx < Config.N_ROIS:
                X_full[target_row_idx, :] = ts_extracted[:, col_idx]
            else:
                # 这种情况很少见，除非用了错误的图谱（比如 1000 维的）
                # print(f"  [Info] 忽略超出范围的标签: {label_val}")
                pass

        return X_full

    except Exception as e:
        print(f"  [Extract Error] {os.path.basename(nii_path)}: {e}")
        return None

def temporal_denoising(X):
    """时序降噪"""
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


# ================= 2. SRM 模块 (寻找交集 + 全员对齐) =================

def find_common_video_run(all_files):
    """
    [核心逻辑] 在所有被试中寻找一个公共的 (ses, run) 组合。
    必须是每个人都有的，才能用来做 SRM 对齐。
    """
    video_files = [f for f in all_files if f['task_key'] == 'video']

    # 构建每个人的 video 集合: Set('ses-01_run-01', 'ses-01_run-02', ...)
    sub_runs = {}
    for f in video_files:
        key = f"{f['ses']}_{f['run']}"  # 组合键
        if f['sub'] not in sub_runs: sub_runs[f['sub']] = set()
        sub_runs[f['sub']].add(key)

    # 找交集
    # 我们只关心 Config.ALL_SUBJECTS 里定义的人
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
            print(f"  [Warning] 被试 {sub} 没有 Video 数据，无法参与 SRM 对齐。")

    if not common_keys:
        raise ValueError("所有被试之间没有找到共同的 Video (ses+run)！无法进行严格对齐。")

    # 排序取第一个作为锚点 (e.g., 'ses-01_run-01')
    best_key = sorted(list(common_keys))[0]
    target_ses, target_run = best_key.split('_')

    print(f"\n>>> [SRM 对齐锚点] 找到公共 Video: {target_ses} {target_run}")
    print(f"    覆盖被试数: {valid_subs_count}/{len(Config.ALL_SUBJECTS)}")

    return target_ses, target_run


def get_or_train_srm(masker, all_files):
    """
    [多核加速版] 训练 SRM
    使用 joblib 并行读取数据，显著减少等待时间。
    """
    srm_path = os.path.join(Config.MODEL_ROOT, "srm_model.pkl")
    sub_list_path = os.path.join(Config.MODEL_ROOT, "srm_subjects.pkl")
    os.makedirs(Config.MODEL_ROOT, exist_ok=True)

    if os.path.exists(srm_path) and os.path.exists(sub_list_path):
        print(">>> [SRM] 加载已有模型...")
        with open(srm_path, 'rb') as f: srm = pickle.load(f)
        with open(sub_list_path, 'rb') as f: subs = pickle.load(f)
        return srm, subs

    # 1. 寻找共有的 Video
    target_ses, target_run = find_common_video_run(all_files)
    print(f">>> [SRM] 开始并行读取训练数据 (锚点: {target_ses} {target_run})...")

    # --- 辅助函数：单个被试的处理逻辑 ---
    def process_one_subject(sub):
        # 在 all_files 里找该被试的特定 run
        target_file = next((f for f in all_files if
                            f['sub'] == sub and
                            f['task_key'] == 'video' and
                            f['ses'] == target_ses and
                            f['run'] == target_run), None)

        if target_file is None:
            return None  # 缺失该文件

        # 提取信号
        X = extract_signals_safe(masker, target_file['path'])
        if X is None: return None

        # Z-score (SRM 必须)
        X_z = np.nan_to_num(zscore(X, axis=1))
        return (sub, X_z)

    # --- 开始并行处理 ---
    # n_jobs=-1 表示使用所有 CPU 核心 (小心内存溢出)
    # 如果服务器内存有限，建议设为 n_jobs=4 或 8
    print(f"    正在启动多核加速处理 {len(Config.ALL_SUBJECTS)} 个被试...")

    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_subject)(sub) for sub in Config.ALL_SUBJECTS
    )

    # --- 整理结果 ---
    train_data = []
    training_subs = []
    min_time = np.inf

    for res in results:
        if res is None: continue
        sub_id, X_z = res
        train_data.append(X_z)
        training_subs.append(sub_id)
        min_time = min(min_time, X_z.shape[1])

    print(f"\n    读取完成。有效被试: {len(training_subs)} (截断 T={min_time})")

    if len(train_data) < 2:
        raise ValueError("有效训练数据不足 2 人！")

    # 截断 & 训练
    data_aligned = [d[:, :min_time] for d in train_data]

    print("    开始训练 SRM 模型...")
    srm = SRM(n_iter=10, features=Config.N_ROIS)
    srm.fit(data_aligned)

    with open(srm_path, 'wb') as f:
        pickle.dump(srm, f)
    with open(sub_list_path, 'wb') as f:
        pickle.dump(training_subs, f)

    return srm, training_subs


# ================= 3. GW 模块 (严格仅限训练集) =================

# 确保文件头部引入了 joblib
from joblib import Parallel, delayed


def compute_gw_mapping(srm_model, srm_subs, masker, all_files):
    """
    [极速版] 计算 GW 映射
    1. 使用多核并行 (Parallel)。
    2. 仅使用 SRM 对齐时的那个“公共锚点视频”，不再读取所有视频，大幅提速。
    """
    map_path = Config.MAPPING_FILE
    if os.path.exists(map_path):
        print(f">>> [GW] 加载已有映射: {map_path}")
        return np.load(map_path)

    print(f"\n>>> [GW] 计算全局映射 (严格仅限训练集 0002-0016)...")

    # 1. 确定训练名单 (既在 SRM 里，又在 Config.TRAIN_SUBJECTS 里)
    valid_train_subs = [s for s in Config.TRAIN_SUBJECTS if s in srm_subs]

    if not valid_train_subs:
        raise ValueError("没有有效的训练集被试用于计算 GW 映射！")

    # 2. 重新找回那个“公共锚点” (ses, run)
    # 这样我们只读这一个文件，速度最快且最稳
    target_ses, target_run = find_common_video_run(all_files)
    print(f"    锁定锚点视频: {target_ses} {target_run} (仅使用此文件计算拓扑)")

    # --- 定义单个被试的处理函数 (用于并行) ---
    def process_one_sub_gw(sub):
        # 精确查找该被试的锚点文件
        target_file = next((f for f in all_files if
                            f['sub'] == sub and
                            f['task_key'] == 'video' and
                            f['ses'] == target_ses and
                            f['run'] == target_run), None)

        if target_file is None: return None

        # 1. 提取
        X = extract_signals_safe(masker, target_file['path'])
        if X is None: return None

        # 2. 投影到共享空间 (SRM Transform)
        # S = W.T @ Zscore(X)
        try:
            sub_idx = srm_subs.index(sub)
            W = srm_model.w_[sub_idx]
            X_z = np.nan_to_num(zscore(X, axis=1))
            X_shared = W.T @ X_z
        except ValueError:
            return None

        # 3. 降噪
        X_den = temporal_denoising(X_shared)

        # 4. 计算相关矩阵 (400x400)
        # 这是 GW 的核心输入
        corr = np.corrcoef(X_den)
        return np.nan_to_num(corr)

    # --- 开始并行计算 ---
    print(f"    启动多核加速 (n_jobs=8)...")
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_sub_gw)(sub) for sub in valid_train_subs
    )

    # --- 汇总结果 ---
    sum_corr = np.zeros((Config.N_ROIS, Config.N_ROIS))
    count = 0

    for res in results:
        if res is not None:
            sum_corr += res
            count += 1

    if count == 0:
        raise ValueError("GW 计算失败：没有提取到任何有效的相关矩阵。")

    mean_corr = sum_corr / count
    print(f"\n    GW 相关矩阵计算完成 (基于 {count} 个被试)。")

    # --- 解最优传输 (这一步很快，通常几秒钟) ---
    print("    正在解最优传输 (Gromov-Wasserstein)...")

    # 源域结构 (1 - Corr)
    C1 = 1.0 - mean_corr
    C1 /= (C1.max() + 1e-9)

    # 目标域结构 (2D Grid)
    sz = Config.IMG_SIZE
    coords = np.array([(i, j) for i in range(sz) for j in range(sz)])
    C2 = cdist(coords, coords, metric='euclidean')
    C2 /= (C2.max() + 1e-9)

    p = ot.unif(Config.N_ROIS)
    q = ot.unif(sz * sz)

    # 计算 GW 映射
    gw_map = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-4)

    # 归一化
    row_sums = gw_map.sum(axis=1, keepdims=True)
    gw_map_norm = gw_map / (row_sums + 1e-9)

    np.save(map_path, gw_map_norm)
    print(f"    GW 映射已保存至 {map_path}")
    return gw_map_norm


# ================= 4. 主流程 =================

def run_prep():
    if not os.path.exists(Config.ATLAS_PATH):
        print(f"[Fatal] 图谱未找到: {Config.ATLAS_PATH}")
        sys.exit(1)
    masker = NiftiLabelsMasker(labels_img=Config.ATLAS_PATH, standardize=True, verbose=0)

    # 1. 扫描
    all_files = scan_dataset()
    if not all_files: sys.exit(1)

    # 2. SRM 对齐 (全员参与，确保都有 W 矩阵)
    srm_model, srm_subs = get_or_train_srm(masker, all_files)

    # 3. GW 映射 (仅训练集参与)
    S_global = compute_gw_mapping(srm_model, srm_subs, masker, all_files)

    # -------------------------------------------------
    # 4. 生成图像 (多核并行极速版)
    # -------------------------------------------------
    print(f"\n>>> [Pipeline] 启动多核加速 (n_jobs=8) 处理所有数据 (Train + Test)...")

    # 定义单个文件的处理逻辑
    def process_one_file_final(item):
        # 1. 路径准备
        sub_out_dir = os.path.join(Config.IMG_ROOT, item['sub'])
        os.makedirs(sub_out_dir, exist_ok=True)
        fname = f"{item['sub']}_{item['ses']}_{item['task_key']}_{item['run']}_bcne.npy"
        save_path = os.path.join(sub_out_dir, fname)

        # 2. 检查是否已存在
        if os.path.exists(save_path):
            return None  # 跳过

        # 3. 检查被试是否在 SRM 名单里
        if item['sub'] not in srm_subs:
            return None  # 跳过

        # 4. 提取信号 (最耗时的一步)
        # 注意: 这里是在子进程中，print 不一定会立刻显示，所以依赖 joblib 的进度条
        X = extract_signals_safe(masker, item['path'])
        if X is None: return None

        try:
            # 5. 投影 (SRM Transform)
            sub_idx = srm_subs.index(item['sub'])
            W = srm_model.w_[sub_idx]

            # Z-score + Project
            X_z = np.nan_to_num(zscore(X, axis=1))
            X_shared = W.T @ X_z

            # 6. 降噪 (BCNE)
            X_den = temporal_denoising(X_shared)

            # 7. 映射 (GW Project)
            # S_global.T (Pixels, Feats) @ X (Feats, Time) -> (Pixels, Time)
            mapped = S_global.T @ X_den

            # 8. 保存
            T = mapped.shape[1]
            # Reshape -> (Time, Channel=1, H, W)
            img = mapped.T.reshape(T, 1, Config.IMG_SIZE, Config.IMG_SIZE)
            np.save(save_path, img)
            return item['sub']  # 返回成功处理的被试名用于统计

        except Exception as e:
            # 并行时报错不太好打印，可以 pass 或者记录日志
            return None

    # --- 启动并行任务 ---
    # n_jobs=8 (根据你之前的测试，8核非常稳)
    # verbose=5 会显示进度条和预估剩余时间
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one_file_final)(item) for item in all_files
    )

    print("\n\n>>> 全部完成！所有图像已生成。")


if __name__ == "__main__":
    run_prep()