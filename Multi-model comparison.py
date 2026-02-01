import os
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score, \
    matthews_corrcoef, roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import re
from tqdm import tqdm
import random
from scipy import ndimage
import shutil
from math import sqrt, log2
from functools import partial
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.exposure import rescale_intensity
from scipy.stats import entropy as sci_entropy
from scipy.stats import skew, kurtosis
from skimage.feature import canny
import cv2
import timm
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from itertools import cycle

# --- Global Settings ---
os.environ['TORCH_HOME'] = r'----'
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Matplotlib Display Settings ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

EXCLUDED_CLASSES = [0]
CLASSIFICATION_CLASSES = []
NUM_CLASSIFICATION_CLASSES = 0
CSV_LABEL_TO_OUTPUT_CLASS_ID = {}

# --- Define Class Name Mapping ---
CLASS_NAMES_MAP = {
    1: "Leymus chinensis",
    2: "Scirpus triqueter",
    3: "Bolboschoenus maritimus",
    4: "Calamagrostis epigejos",
    5: "Carex heterostachya",
    6: "Stipa capillata",
    7: "Lappula myosotis",
    8: "Suaeda salsa",
    9: "Artemisia frigida",
    10: "Phragmites australis",
    11: "Medicago sativa",
    12: "Potentilla chinensis",
    13: "Sonchus brachyotus",
    14: "Stellera chamaejasme",
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

RED_IDX = 0
GREEN_IDX = 1
BLUE_IDX = 2
NIR_IDX = 3
RED_EDGE_IDX = 4

ALL_IMAGE_FEATURES = [
    'Red', 'Green', 'Blue', 'NIR', 'Red_Edge',
    'NDVI', 'SAVI', 'NDWI',
    'NDRE', 'PRI', 'GNDVI', 'EVI', 'OSAVI',
    'ARVI', 'GCI', 'NDCI', 'RVI', 'VARI',
    'LBP_Spatial',
    'LBP_Mean', 'LBP_StdDev', 'LBP_Entropy',
    'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy',
    'GLCM_ASM', 'GLCM_Correlation',
    'NIR_Skewness', 'NIR_Kurtosis',
    'Edge_Density'
]

TOP_FEATURES_TO_USE = [
    'VARI', 'RVI', 'LBP_Spatial', 'GLCM_Correlation', 'NDWI',
    'LBP_Entropy', 'GCI', 'NDRE', 'Edge_Density',
    'PRI', 'ARVI', 'NIR_Kurtosis', 'NIR_Skewness', 'LBP_StdDev',
    'GLCM_Homogeneity', 'NDCI', 'OSAVI', 'GNDVI', 'SAVI',
]

ACTIVE_FEATURES = TOP_FEATURES_TO_USE
DILATION_RATES = [1, 2, 3, 4]


def load_tif_mappings(tif_directory):
    ms_tif_map = {}
    if os.path.exists(tif_directory):
        for f in os.listdir(tif_directory):
            if f.lower().endswith('.tif'):
                base_name = os.path.splitext(f)[0]
                base_name = re.sub(r'[_-]?result$', '', base_name, flags=re.IGNORECASE)
                ms_tif_map[base_name] = os.path.join(tif_directory, f)
    return ms_tif_map


def calculate_image_channels(raw_patch_bands, target_size, epsilon=1e-8, feature_list=None):
    if feature_list is None:
        feature_list = ALL_IMAGE_FEATURES

    num_expected_features = len(feature_list)

    if raw_patch_bands is None or raw_patch_bands.shape[0] < 5 or target_size <= 0:
        return np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)

    max_ms_val = 65535.0

    def _process_channel(channel_data, min_val=-1.0, max_val=1.0):
        processed = np.nan_to_num(channel_data, nan=(min_val + max_val) / 2.0, posinf=max_val, neginf=min_val)
        processed = np.clip(processed, min_val, max_val)
        range_diff = max_val - min_val
        return ((processed - min_val) / range_diff * 2 - 1).astype(
            np.float32) if range_diff > epsilon else np.zeros_like(processed, dtype=np.float32)

    try:
        red = (raw_patch_bands[RED_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        green = (raw_patch_bands[GREEN_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        blue = (raw_patch_bands[BLUE_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        nir = (raw_patch_bands[NIR_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        red_edge = (raw_patch_bands[RED_EDGE_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
    except IndexError:
        return np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)

    gndvi = (nir - green) / (nir + green + epsilon)
    gndvi_for_mask = np.nan_to_num(gndvi, nan=0.0).clip(-1, 1)
    gndvi_8bit = ((gndvi_for_mask + 1) * 127.5).astype(np.uint8)

    if gndvi_8bit.size == 0 or np.all(gndvi_8bit == gndvi_8bit.flat[0]):
        vegetation_mask = np.zeros_like(gndvi_8bit, dtype=bool)
    else:
        try:
            _, otsu_mask = cv2.threshold(gndvi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            vegetation_mask = otsu_mask.astype(bool)
        except cv2.error:
            vegetation_mask = gndvi_for_mask > np.mean(gndvi_for_mask)

    calculated_channels = {}
    calculated_channels['Red'] = _process_channel(red, 0.0, 1.0)
    calculated_channels['Green'] = _process_channel(green, 0.0, 1.0)
    calculated_channels['Blue'] = _process_channel(blue, 0.0, 1.0)
    calculated_channels['NIR'] = _process_channel(nir, 0.0, 1.0)
    calculated_channels['Red_Edge'] = _process_channel(red_edge, 0.0, 1.0)

    mask_fill_value = -1.0

    ndvi = (nir - red) / (nir + red + epsilon)
    calculated_channels['NDVI'] = _process_channel(np.where(vegetation_mask, ndvi, mask_fill_value))
    savi = ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5
    calculated_channels['SAVI'] = _process_channel(np.where(vegetation_mask, savi, mask_fill_value))
    ndwi = (green - nir) / (green + nir + epsilon)
    calculated_channels['NDWI'] = _process_channel(np.where(vegetation_mask, ndwi, mask_fill_value))
    ndre = (nir - red_edge) / (nir + red_edge + epsilon)
    calculated_channels['NDRE'] = _process_channel(np.where(vegetation_mask, ndre, mask_fill_value))
    pri = (green - blue) / (green + blue + epsilon)
    calculated_channels['PRI'] = _process_channel(np.where(vegetation_mask, pri, mask_fill_value))
    gndvi_masked = np.where(vegetation_mask, gndvi, mask_fill_value)
    calculated_channels['GNDVI'] = _process_channel(gndvi_masked)
    L = 1.0;
    C1 = 6.0;
    C2 = 7.5
    evi = 2.5 * (nir - red) / (nir + C1 * red - C2 * blue + L + epsilon)
    calculated_channels['EVI'] = _process_channel(np.where(vegetation_mask, evi, mask_fill_value), min_val=-1.0,
                                                  max_val=1.0)
    L_osavi = 0.16
    osavi = (1 + L_osavi) * (nir - red) / (nir + red + L_osavi + epsilon)
    calculated_channels['OSAVI'] = _process_channel(np.where(vegetation_mask, osavi, mask_fill_value), min_val=-1.0,
                                                    max_val=1.0)
    arvi_numerator = nir - (2 * red - blue)
    arvi_denominator = nir + (2 * red - blue) + epsilon
    arvi = arvi_numerator / arvi_denominator
    calculated_channels['ARVI'] = _process_channel(np.where(vegetation_mask, arvi, mask_fill_value))
    gci = (nir / (green + epsilon)) - 1
    calculated_channels['GCI'] = _process_channel(np.where(vegetation_mask, gci, mask_fill_value), min_val=-10.0,
                                                  max_val=10.0)
    ndci = (red_edge - blue) / (red_edge + blue + epsilon)
    calculated_channels['NDCI'] = _process_channel(np.where(vegetation_mask, ndci, mask_fill_value))
    rvi = nir / (red + epsilon)
    calculated_channels['RVI'] = _process_channel(np.where(vegetation_mask, rvi, mask_fill_value), min_val=0.0,
                                                  max_val=10.0)
    vari = (green - red) / (green + red - blue + epsilon)
    calculated_channels['VARI'] = _process_channel(np.where(vegetation_mask, vari, mask_fill_value))

    lbp_spatial_channel = np.full((target_size, target_size), -1.0, dtype=np.float32)
    lbp_mean_norm, lbp_std_norm, lbp_entropy_norm = -1.0, -1.0, -1.0

    if any(f in feature_list for f in ['LBP_Spatial', 'LBP_Mean', 'LBP_StdDev', 'LBP_Entropy']):
        try:
            if gndvi_8bit.size == 0 or np.all(gndvi_8bit == gndvi_8bit.flat[0]):
                lbp_image = np.zeros_like(gndvi_8bit, dtype=np.float32)
            else:
                P_LBP = 8;
                R_LBP = 1
                lbp_image = local_binary_pattern(gndvi_8bit, P_LBP, R_LBP, 'uniform')

            max_lbp_val_for_norm = P_LBP + 2
            lbp_spatial_channel = _process_channel(lbp_image, 0, max_lbp_val_for_norm)

            lbp_mean = np.mean(lbp_image)
            lbp_std = np.std(lbp_image)
            n_bins_lbp = int(lbp_image.max() + 1) if lbp_image.size > 0 and lbp_image.max() >= 0 else 1
            hist_lbp, _ = np.histogram(lbp_image.ravel(), bins=n_bins_lbp, range=(0, n_bins_lbp), density=True)
            hist_lbp = hist_lbp[hist_lbp > 0]
            lbp_entropy = sci_entropy(hist_lbp, base=2) if hist_lbp.size > 0 else 0.0

            lbp_mean_norm = _process_channel(np.array([lbp_mean]), 0, max_lbp_val_for_norm)[0]
            lbp_std_norm = _process_channel(np.array([lbp_std]), 0, max_lbp_val_for_norm / 2)[0]
            max_entropy_for_P = np.log2(P_LBP + 1) if P_LBP > 0 else 1.0
            lbp_entropy_norm = _process_channel(np.array([lbp_entropy]), 0, max_entropy_for_P)[0]
        except Exception:
            pass

    calculated_channels['LBP_Spatial'] = lbp_spatial_channel
    calculated_channels['LBP_Mean'] = np.full((target_size, target_size), lbp_mean_norm, dtype=np.float32)
    calculated_channels['LBP_StdDev'] = np.full((target_size, target_size), lbp_std_norm, dtype=np.float32)
    calculated_channels['LBP_Entropy'] = np.full((target_size, target_size), lbp_entropy_norm, dtype=np.float32)

    glcm_contrast_norm, glcm_dissimilarity_norm, glcm_homogeneity_norm, glcm_energy_norm, glcm_asm_norm, glcm_correlation_norm = \
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

    if any(f in feature_list for f in
           ['GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy', 'GLCM_ASM', 'GLCM_Correlation']):
        try:
            if gndvi_8bit.size == 0 or np.all(gndvi_8bit == gndvi_8bit.flat[0]):
                glcm = np.zeros((256, 256, 1, 1), dtype=np.float64)
            else:
                distances_glcm = [1];
                angles_glcm = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4];
                levels_glcm = 256
                glcm = graycomatrix(gndvi_8bit, distances=distances_glcm, angles=angles_glcm, levels=levels_glcm,
                                    symmetric=True, normed=True)

            def get_glcm_prop(glcm_matrix, prop_name):
                if glcm_matrix.size == 0 or glcm_matrix.ndim < 4 or glcm_matrix.shape[0] == 0:
                    return 0.0
                try:
                    prop_val = np.mean(graycoprops(glcm_matrix, prop_name))
                    return np.nan_to_num(prop_val, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    return 0.0

            glcm_contrast = get_glcm_prop(glcm, 'contrast')
            glcm_dissimilarity = get_glcm_prop(glcm, 'dissimilarity')
            glcm_homogeneity = get_glcm_prop(glcm, 'homogeneity')
            glcm_energy = get_glcm_prop(glcm, 'energy')
            glcm_asm = get_glcm_prop(glcm, 'ASM')
            glcm_correlation = get_glcm_prop(glcm, 'correlation')

            glcm_contrast_norm = _process_channel(np.array([glcm_contrast]), 0, (levels_glcm - 1) ** 2)[0]
            glcm_dissimilarity_norm = _process_channel(np.array([glcm_dissimilarity]), 0, (levels_glcm - 1))[0]
            glcm_homogeneity_norm = _process_channel(np.array([glcm_homogeneity]), 0, 1)[0]
            glcm_energy_norm = _process_channel(np.array([glcm_energy]), 0, 1)[0]
            glcm_asm_norm = _process_channel(np.array([glcm_asm]), 0, 1)[0]
            glcm_correlation_norm = _process_channel(np.array([glcm_correlation]), -1, 1)[0]
        except Exception:
            pass

    calculated_channels['GLCM_Contrast'] = np.full((target_size, target_size), glcm_contrast_norm, dtype=np.float32)
    calculated_channels['GLCM_Dissimilarity'] = np.full((target_size, target_size), glcm_dissimilarity_norm,
                                                        dtype=np.float32)
    calculated_channels['GLCM_Homogeneity'] = np.full((target_size, target_size), glcm_homogeneity_norm,
                                                      dtype=np.float32)
    calculated_channels['GLCM_Energy'] = np.full((target_size, target_size), glcm_energy_norm, dtype=np.float32)
    calculated_channels['GLCM_ASM'] = np.full((target_size, target_size), glcm_asm_norm, dtype=np.float32)
    calculated_channels['GLCM_Correlation'] = np.full((target_size, target_size), glcm_correlation_norm,
                                                      dtype=np.float32)

    patch_skew_norm = -1.0;
    patch_kurtosis_norm = -1.0
    if any(f in feature_list for f in ['NIR_Skewness', 'NIR_Kurtosis']):
        try:
            nir_flat = nir.ravel()
            if nir_flat.size > 1 and np.std(nir_flat) > epsilon and np.isfinite(nir_flat).all():
                patch_skew = skew(nir_flat)
                patch_kurtosis = kurtosis(nir_flat, fisher=True)
                patch_skew_norm = _process_channel(np.array([patch_skew]), -5.0, 5.0)[0]
                patch_kurtosis_norm = _process_channel(np.array([patch_kurtosis]), -3.0, 20.0)[0]
        except Exception:
            pass

    calculated_channels['NIR_Skewness'] = np.full((target_size, target_size), patch_skew_norm, dtype=np.float32)
    calculated_channels['NIR_Kurtosis'] = np.full((target_size, target_size), patch_kurtosis_norm, dtype=np.float32)

    edge_density_norm = -1.0
    if 'Edge_Density' in feature_list:
        try:
            if not np.isfinite(nir).all() or nir.size == 0 or np.all(nir == nir.flat[0]):
                nir_canny_input = np.zeros_like(nir, dtype=np.uint8)
            else:
                nir_canny_input = (rescale_intensity(nir, out_range=(0, 255))).astype(np.uint8)

            if np.max(nir_canny_input) == np.min(nir_canny_input):
                edges = np.zeros_like(nir_canny_input, dtype=bool)
            else:
                edges = canny(nir_canny_input, sigma=1.0)
            edge_density = np.mean(edges.astype(np.float32))
            edge_density_norm = _process_channel(np.array([edge_density]), 0, 1)[0]
        except Exception:
            pass

    calculated_channels['Edge_Density'] = np.full((target_size, target_size), edge_density_norm, dtype=np.float32)

    feature_stack = []
    for name in feature_list:
        feature = calculated_channels.get(name)
        if feature is None:
            feature = np.full((target_size, target_size), -1.0, dtype=np.float32)
        feature_stack.append(feature)

    if not feature_stack:
        return np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)

    final_stack = np.stack(feature_stack, axis=0)
    if final_stack.shape[0] != num_expected_features:
        correct_shape_array = np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)
        copy_channels = min(final_stack.shape[0], num_expected_features)
        correct_shape_array[:copy_channels, :, :] = final_stack[:copy_channels, :, :]
        final_stack = correct_shape_array

    return final_stack


class ClassificationAugmentation:
    def __init__(self):
        self.rot_p = 0.7
        self.flip_p = 0.7
        self.noise_p = 0.4
        self.noise_std = 0.05
        self.cut_p = 0.5
        self.cut_ratio = 0.15
        self.cut_fill = -1.0

    def __call__(self, img, target):
        img = img.astype(np.float32)
        if random.random() < self.rot_p:
            img = ndimage.rotate(img, random.choice([90, 180, 270]), axes=(1, 2),
                                 reshape=False, order=1, mode='reflect')
        if random.random() < self.flip_p:
            img = np.flip(img, axis=2).copy()
        if random.random() < self.flip_p:
            img = np.flip(img, axis=1).copy()
        if random.random() < self.noise_p:
            img += np.random.normal(0, self.noise_std, img.shape)
        if random.random() < self.cut_p:
            h, w = img.shape[1:]
            s = int(self.cut_ratio * min(h, w))
            if s > 0:
                x, y = random.randint(0, w - s), random.randint(0, h - s)
                img[:, y:y + s, x:x + s] = self.cut_fill
        return np.clip(img, -1.0, 1.0).astype(np.float32), target


class ClassificationDataset(Dataset):
    def __init__(self, df, map, size, stats, n_cls, cls_map, tf=None, feats=None):
        self.df = df.reset_index(drop=True)
        self.map, self.size, self.tf = map, size, tf
        self.bands = [1, 2, 3, 4, 5]
        self.mean, self.std = stats if stats else (None, None)
        self.feats = feats if feats else ALL_IMAGE_FEATURES
        self.idx_map = {n: i for i, n in enumerate(ALL_IMAGE_FEATURES)}
        self.cls_map = cls_map
        self.def_cls = min(cls_map.values()) if cls_map else 0
        self.num_image_features = len(self.feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.map.get(row['image_name'])
        tid = self.cls_map.get(row['label'], self.def_cls)

        if not path or not os.path.exists(path): return self._dummy()

        try:
            with rasterio.open(path) as src:
                if not (src.bounds.left <= row['x'] <= src.bounds.right and src.bounds.bottom <= row[
                    'y'] <= src.bounds.top):
                    return self._dummy()
                r, c = src.index(row['x'], row['y'])
                win = rasterio.windows.Window(c - self.size // 2, r - self.size // 2, self.size, self.size)
                raw = src.read(self.bands, window=win, boundless=True, fill_value=0,
                               out_shape=(5, self.size, self.size))
                if raw.shape[0] < 5:
                    tmp = np.zeros((5, self.size, self.size), dtype=raw.dtype)
                    tmp[:raw.shape[0]] = raw
                    raw = tmp

            all_feats = calculate_image_channels(raw, self.size, feature_list=ALL_IMAGE_FEATURES)
            img = all_feats[[self.idx_map[n] for n in self.feats]]

            if self.tf: img, tid = self.tf(img, tid)

            if self.mean is not None:
                img = (img - self.mean.reshape(-1, 1, 1)) / (self.std.reshape(-1, 1, 1) + 1e-8)

            return torch.from_numpy(np.clip(img, -1.0, 1.0)).float(), torch.tensor(tid, dtype=torch.long)
        except:
            return self._dummy()

    def _dummy(self):
        return torch.full((len(self.feats), self.size, self.size), -1.0, dtype=torch.float32), torch.tensor(
            self.def_cls, dtype=torch.long)


def collate_fn_classification(batch):
    batch = list(filter(lambda x: x and x[0] is not None, batch))
    if not batch: return None, None
    imgs, lbls = zip(*batch)
    return torch.stack(imgs), torch.stack(lbls)


def cutmix_data(imgs, lbls, alpha, num_classes):
    if alpha <= 0:
        return imgs, lbls, lbls, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0)).to(imgs.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
    imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[idx, :, bby1:bby2, bbx1:bbx2]
    return imgs, lbls, lbls[idx], lam


def rand_bbox(size, lam):
    W, H = size[3], size[2]
    cut = np.sqrt(1. - lam)
    cw, ch = int(W * cut), int(H * cut)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H), np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2,
                                                                                                          0, H)


def plot_classification_results(images, true_labels, pred_labels, class_names, mean, std, save_dir, filename_prefix="",
                                image_features=None, font_size=26):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = images.shape[0]

    if image_features is None:
        image_features = ACTIVE_FEATURES

    vis_channel_indices = {}
    original_bands = ['Red', 'Green', 'Blue', 'NIR', 'Red_Edge']
    for band_name in original_bands:
        if band_name in image_features:
            vis_channel_indices[band_name] = image_features.index(band_name)

    vis_mode = 'none'
    if all(b in vis_channel_indices for b in ['Red', 'Green', 'Blue']):
        vis_mode = 'rgb'
    elif 'NIR' in vis_channel_indices and 'Red' in vis_channel_indices and 'Green' in vis_channel_indices:
        vis_mode = 'false_color_nir'
    elif 'NIR' in vis_channel_indices:
        vis_mode = 'grayscale_nir'
    elif 'Red_Edge' in vis_channel_indices:
        vis_mode = 'grayscale_red_edge'
    elif 'Red' in vis_channel_indices:
        vis_mode = 'grayscale_red'
    elif image_features:
        vis_mode = 'grayscale_first_feature'

    for i in range(min(num_samples, 10)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        img_np = images[i].cpu().numpy()

        if isinstance(true_labels, torch.Tensor):
            true_labels_np = true_labels.cpu().numpy()
        else:
            true_labels_np = true_labels

        if isinstance(pred_labels, torch.Tensor):
            pred_labels_np = pred_labels.cpu().numpy()
        else:
            pred_labels_np = pred_labels

        true_label_idx = int(round(true_labels_np[i].item())) if isinstance(true_labels_np[i].item(), float) else \
            true_labels_np[i].item()
        pred_label_idx = pred_labels_np[i].item()

        if true_label_idx < len(class_names):
            true_label = class_names[true_label_idx]
        else:
            true_label = f"Unknown ({true_label_idx})"

        if pred_label_idx < len(class_names):
            pred_label = class_names[pred_label_idx]
        else:
            pred_label = f"Unknown ({pred_label_idx})"

        denormalized_img_global = img_np
        if mean is not None and std is not None and len(mean) == img_np.shape[0]:
            denormalized_img_global = (img_np * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1))

        denormalized_img_global = (denormalized_img_global + 1) / 2
        denormalized_img_global = np.clip(denormalized_img_global, 0, 1)

        display_img = None
        cmap = 'gray'

        if vis_mode == 'rgb':
            r_idx, g_idx, b_idx = vis_channel_indices['Red'], vis_channel_indices['Green'], vis_channel_indices['Blue']
            display_img = np.stack([
                denormalized_img_global[r_idx, :, :],
                denormalized_img_global[g_idx, :, :],
                denormalized_img_global[b_idx, :, :]
            ], axis=-1)
        elif vis_mode == 'false_color_nir':
            nir_idx, r_idx, g_idx = vis_channel_indices['NIR'], vis_channel_indices['Red'], vis_channel_indices['Green']
            display_img = np.stack([
                denormalized_img_global[nir_idx, :, :],
                denormalized_img_global[r_idx, :, :],
                denormalized_img_global[g_idx, :, :]
            ], axis=-1)
        elif vis_mode == 'grayscale_nir':
            display_img = denormalized_img_global[vis_channel_indices['NIR'], :, :]
        elif vis_mode == 'grayscale_red_edge':
            display_img = denormalized_img_global[vis_channel_indices['Red_Edge'], :, :]
        elif vis_mode == 'grayscale_red':
            display_img = denormalized_img_global[vis_channel_indices['Red'], :, :]
        elif vis_mode == 'grayscale_first_feature':
            display_img = denormalized_img_global[0, :, :]
        else:
            display_img = np.zeros((images.shape[2], images.shape[3]))

        if display_img is not None:
            ax.imshow(display_img, cmap=cmap if len(display_img.shape) == 2 else None)
        else:
            ax.imshow(np.zeros((images.shape[2], images.shape[3])), cmap='gray')

        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=font_size)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_sample_{i + 1}.png"))
        plt.close(fig)


def analyze_classification_results(y_true, y_pred, class_names, num_classes, save_dir, filename_suffix="",
                                   font_size=26, print_report=True):
    if y_true.size == 0 or y_pred.size == 0:
        print(f"Skipping analysis {filename_suffix}: y_true or y_pred is empty.")
        return None

    all_possible_labels = np.arange(num_classes)

    overall_accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=all_possible_labels, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=all_possible_labels, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=all_possible_labels)
    mcc = matthews_corrcoef(y_true, y_pred)

    if print_report:
        print(f"\n--- Evaluation Metrics ({filename_suffix}) ---")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")
        print(f"Macro Average F1 Score: {macro_f1:.4f}")
        print(f"Cohen Kappa Score: {kappa:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")

        f1_per_class = f1_score(y_true, y_pred, average=None, labels=all_possible_labels, zero_division=0)
        print("\nF1 Score per Class:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  {class_names[i] if i < len(class_names) else f'Unknown Class {i}'}: {f1:.4f}")

        report = classification_report(y_true, y_pred, labels=all_possible_labels, target_names=class_names,
                                       zero_division=0)
        print("\nClassification Report:\n", report)

    return {
        'accuracy': overall_accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'kappa': kappa,
        'mcc': mcc,
        'f1_per_class': f1_score(y_true, y_pred, average=None, labels=all_possible_labels, zero_division=0)
    }


def plot_roc_curve(y_true_one_hot, y_score, num_classes, class_names, save_dir, model_name="", font_size=26):
    plt.figure(figsize=(12, 10))
    lw = 2

    y_score = np.clip(y_score, 0, 1)

    if not y_true_one_hot.size or not y_score.size:
        plt.close()
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    valid_fprs_for_macro = []
    valid_tprs_for_macro = []

    for i in range(num_classes):
        if len(np.unique(y_true_one_hot[:, i])) > 1:
            try:
                fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=lw,
                         label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
                valid_fprs_for_macro.append(fpr[i])
                valid_tprs_for_macro.append(tpr[i])
            except ValueError as e:
                fpr[i], tpr[i], roc_auc[i] = [0.0], [0.0], 0.0
        else:
            fpr[i], tpr[i], roc_auc[i] = [0.0], [0.0], 0.0

    y_true_micro = y_true_one_hot.ravel()
    y_score_micro = y_score.ravel()

    if len(np.unique(y_true_micro)) > 1:
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_micro, y_score_micro)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})',
                     color='deeppink', linestyle=':', lw=lw + 1)
        except ValueError as e:
            print(f"Skipping micro-average ROC calculation due to error: {e}")
    else:
        print(f"Skipping micro-average ROC calculation: only one class in labels.")

    if len(valid_fprs_for_macro) > 0:
        all_fpr = np.unique(np.concatenate(valid_fprs_for_macro))
        mean_tpr = np.zeros_like(all_fpr)
        for i_tpr, i_fpr in zip(valid_tprs_for_macro, valid_fprs_for_macro):
            mean_tpr += np.interp(all_fpr, i_fpr, i_tpr)
        mean_tpr /= len(valid_fprs_for_macro)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-avg (AUC = {roc_auc["macro"]:.4f})',
                 color='navy', linestyle=':', lw=lw + 1)
    else:
        print(f"Skipping macro-average ROC calculation: insufficient classes or all failed.")

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title(f'ROC Curve - {model_name}', fontsize=font_size)
    plt.legend(loc="lower right", fontsize=font_size * 0.7)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_curve_{model_name}.png"))
    plt.close()


def plot_pr_curve(y_true_one_hot, y_score, num_classes, class_names, save_dir, model_name="", font_size=26):
    plt.figure(figsize=(12, 10))
    lw = 2

    y_score = np.clip(y_score, 0, 1)

    if not y_true_one_hot.size or not y_score.size:
        plt.close()
        return

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        if len(np.unique(y_true_one_hot[:, i])) > 1:
            try:
                precision[i], recall[i], _ = precision_recall_curve(y_true_one_hot[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(y_true_one_hot[:, i], y_score[:, i])
                plt.plot(recall[i], precision[i], lw=lw,
                         label=f'{class_names[i]} (AP = {average_precision[i]:.4f})')
            except ValueError as e:
                precision[i], recall[i], average_precision[i] = [0.0], [1.0], 0.0
        else:
            precision[i], recall[i], average_precision[i] = [0.0], [1.0], 0.0

    y_true_micro = y_true_one_hot.ravel()
    y_score_micro = y_score.ravel()

    if len(np.unique(y_true_micro)) > 1:
        try:
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_micro, y_score_micro)
            average_precision["micro"] = average_precision_score(y_true_one_hot, y_score, average="micro")
            plt.plot(recall["micro"], precision["micro"],
                     label=f'Micro-avg (AP = {average_precision["micro"]:.4f})',
                     color='deeppink', linestyle=':', lw=lw + 1)
        except ValueError as e:
            print(f"Skipping micro-average PR calculation due to error: {e}")
    else:
        print(f"Skipping micro-average PR calculation: only one class in labels.")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=font_size)
    plt.legend(loc="lower left", fontsize=font_size * 0.7)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_curve_{model_name}.png"))
    plt.close()


def plot_low_f1_samples(vis_images, vis_true_labels, vis_pred_labels, low_f1_class_ids,
                        class_names, normalization_stats, save_dir, patch_size,
                        num_samples_per_class=5, filename_prefix="", image_features=None, font_size=26):
    os.makedirs(save_dir, exist_ok=True)
    mean, std = normalization_stats

    if isinstance(vis_true_labels, torch.Tensor):
        vis_true_labels = vis_true_labels.cpu().numpy()
    if isinstance(vis_pred_labels, torch.Tensor):
        vis_pred_labels = vis_pred_labels.cpu().numpy()

    if image_features is None:
        image_features = ACTIVE_FEATURES

    vis_channel_indices = {}
    original_bands = ['Red', 'Green', 'Blue', 'NIR', 'Red_Edge']
    for band_name in original_bands:
        if band_name in image_features:
            vis_channel_indices[band_name] = image_features.index(band_name)

    vis_mode = 'none'
    if all(b in vis_channel_indices for b in ['Red', 'Green', 'Blue']):
        vis_mode = 'rgb'
    elif 'NIR' in vis_channel_indices and 'Red' in vis_channel_indices and 'Green' in vis_channel_indices:
        vis_mode = 'false_color_nir'
    elif 'NIR' in vis_channel_indices:
        vis_mode = 'grayscale_nir'
    elif 'Red_Edge' in vis_channel_indices:
        vis_mode = 'grayscale_red_edge'
    elif 'Red' in vis_channel_indices:
        vis_mode = 'grayscale_red'
    elif image_features:
        vis_mode = 'grayscale_first_feature'

    sample_count = 0
    for class_id in low_f1_class_ids:
        indices_for_class = np.where(vis_true_labels == class_id)[0]
        if len(indices_for_class) == 0:
            continue

        sampled_indices = random.sample(list(indices_for_class), min(len(indices_for_class), num_samples_per_class))

        for i, idx in enumerate(sampled_indices):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            img_np = vis_images[idx].cpu().numpy()
            true_label_idx = vis_true_labels[idx].item()
            pred_label_idx = vis_pred_labels[idx].item()

            true_label = class_names[true_label_idx] if true_label_idx < len(
                class_names) else f"Unknown ({true_label_idx})"
            pred_label = class_names[pred_label_idx] if pred_label_idx < len(
                class_names) else f"Unknown ({pred_label_idx})"

            denormalized_img_global = img_np
            if mean is not None and std is not None and len(mean) == img_np.shape[0]:
                denormalized_img_global = (img_np * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1))

            denormalized_img_global = (denormalized_img_global + 1) / 2
            denormalized_img_global = np.clip(denormalized_img_global, 0, 1)

            display_img = None
            cmap = 'gray'

            if vis_mode == 'rgb':
                r_idx, g_idx, b_idx = vis_channel_indices['Red'], vis_channel_indices['Green'], vis_channel_indices[
                    'Blue']
                display_img = np.stack(
                    [denormalized_img_global[r_idx], denormalized_img_global[g_idx], denormalized_img_global[b_idx]],
                    axis=-1)
            elif vis_mode == 'false_color_nir':
                nir_idx, r_idx, g_idx = vis_channel_indices['NIR'], vis_channel_indices['Red'], vis_channel_indices[
                    'Green']
                display_img = np.stack(
                    [denormalized_img_global[nir_idx], denormalized_img_global[r_idx], denormalized_img_global[g_idx]],
                    axis=-1)
            elif vis_mode == 'grayscale_nir':
                display_img = denormalized_img_global[vis_channel_indices['NIR']]
            elif vis_mode == 'grayscale_red_edge':
                display_img = denormalized_img_global[vis_channel_indices['Red_Edge']]
            elif vis_mode == 'grayscale_red':
                display_img = denormalized_img_global[vis_channel_indices['Red']]
            elif vis_mode == 'grayscale_first_feature':
                display_img = denormalized_img_global[0]
            else:
                display_img = np.zeros((patch_size, patch_size))

            ax.imshow(display_img, cmap=cmap if len(display_img.shape) == 2 else None)

            ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=font_size)
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_class_{class_id}_sample_{i + 1}.png"))
            plt.close(fig)
            sample_count += 1


def calculate_dataset_stats(dataset, max_samples=1000):
    print("Calculating Dataset Statistics...")
    ldr = DataLoader(dataset, batch_size=1, shuffle=False)
    feats = []
    for i, (img, _) in enumerate(tqdm(ldr, total=max_samples)):
        if i >= max_samples: break
        if img is not None and img.shape[1] == dataset.num_image_features:
            f = img.permute(0, 2, 3, 1).reshape(-1, img.shape[1]).cpu().numpy()
            if f.shape[0] > 0:
                idx = np.random.choice(f.shape[0], min(f.shape[0], 1000), replace=False)
                feats.append(f[idx])

    if not feats: return None, None
    arr = np.concatenate(feats, 0)
    mean, std = arr.mean(0), arr.std(0)
    std[std < 1e-8] = 1.0
    return mean, std


# --- Model Architecture ---
class ECALayer(nn.Module):
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        k = int(abs(log2(ch) + b) / gamma)
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(self.avg(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y).expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, 1, 1), nn.BatchNorm2d(outc)
        )
        self.skip = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()
        self.eca = ECALayer(outc)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.eca(self.conv(x) + self.skip(x)))


class MultiScaleConvBlock(nn.Module):
    def __init__(self, inc, outc, dilations=[1, 2, 3, 4]):
        super().__init__()
        branches = []
        bc = max(1, outc // len(dilations))
        for d in dilations:
            branches.append(nn.Sequential(
                nn.Conv2d(inc, bc, 3, padding=d, dilation=d),
                nn.BatchNorm2d(bc), nn.ReLU(True)
            ))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(nn.Conv2d(bc * len(dilations), outc, 1), nn.BatchNorm2d(outc), nn.ReLU(True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], 1))


class SelfAttentionLayer(nn.Module):
    def __init__(self, ch, heads=8, drop=0.1):
        super().__init__()
        self.heads, self.hd = heads, ch // heads
        self.q = nn.Linear(ch, ch)
        self.k = nn.Linear(ch, ch)
        self.v = nn.Linear(ch, ch)
        self.out = nn.Linear(ch, ch)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = x.flatten(2).transpose(1, 2)
        nxf = self.norm(xf)
        q = self.q(nxf).view(B, -1, self.heads, self.hd).transpose(1, 2)
        k = self.k(nxf).view(B, -1, self.heads, self.hd).transpose(1, 2)
        v = self.v(nxf).view(B, -1, self.heads, self.hd).transpose(1, 2)

        att = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.hd ** 0.5), -1)
        res = torch.matmul(self.drop(att), v).transpose(1, 2).reshape(B, -1, C)
        return (xf + self.drop(self.out(res))).transpose(1, 2).reshape(B, C, H, W)


class TransFCNClassifier(nn.Module):
    def __init__(self, inc, n_cls, heads=8):
        super().__init__()
        self.ms = MultiScaleConvBlock(max(1, inc), 64, dilations=DILATION_RATES)
        self.pool = nn.MaxPool2d(2)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.bot = EncoderBlock(512, 1024)
        self.att = SelfAttentionLayer(1024, heads)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.c4 = nn.Conv2d(512, 256, 1)
        self.c3 = nn.Conv2d(256, 256, 1)
        self.cb = nn.Conv2d(1024, 512, 1)

        self.fuse = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.ReLU(True))
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, n_cls)
        )

    def forward(self, x):
        p1 = self.pool(self.ms(x))
        e2 = self.e2(p1);
        p2 = self.pool(e2)
        e3 = self.e3(p2);
        p3 = self.pool(e3)
        e4 = self.e4(p3);
        p4 = self.pool(e4)

        b = self.att(self.bot(p4))
        feat = self.fuse(torch.cat([self.cb(self.up3(b)), self.c4(self.up4(e4)), self.c3(e3)], 1))
        return self.cls(feat)


def train_model_with_onecyclelr(model, train_loader, val_loader, device, model_dir,
                                num_classification_classes,
                                total_epochs, model_name, class_weights, class_names, normalization_stats,
                                max_lr=1e-3, accumulation_steps=4,
                                image_features_list=None,
                                use_cutmix=False, cutmix_alpha=1.0, focal_loss_gamma=2.0,
                                label_smoothing=0.1):
    # Loss: CrossEntropy with weights and label smoothing as described in manuscript
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=total_epochs,
        steps_per_epoch=len(train_loader)
    )

    best_model_path = os.path.join(model_dir, f"{model_name}_best_model.pth")
    current_best_val_f1 = -1.0

    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [{model_name} Training]", leave=False)
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_pbar):
            if images is None or labels is None:
                continue

            images = images.to(device)
            labels = labels.to(device)

            if use_cutmix and images.size(0) > 1:
                images, labels_a, labels_b, lambda_val = cutmix_data(images, labels, cutmix_alpha,
                                                                     num_classification_classes)
                outputs = model(images)
                loss = lambda_val * criterion(outputs, labels_a) + (1 - lambda_val) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * accumulation_steps
            train_pbar.set_postfix(loss=running_loss / (i + 1),
                                   lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_true_labels, val_pred_labels = [], []
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if images is None or labels is None:
                    continue

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        if val_true_labels:
            all_true_labels = np.array(val_true_labels)
            all_pred_labels = np.array(val_pred_labels)

            avg_loss = running_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0

            val_f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted',
                                       labels=np.arange(num_classification_classes), zero_division=0)
            val_f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro',
                                    labels=np.arange(num_classification_classes), zero_division=0)

            print(
                f"Epoch {epoch + 1} [{model_name}]: Train Loss: {avg_loss:.4f}, Val Weighted F1: {val_f1_weighted:.4f}, "
                f"Val Macro F1: {val_f1_macro:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_f1_weighted > current_best_val_f1:
                current_best_val_f1 = val_f1_weighted
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Saved best model for {model_name} with Weighted F1: {current_best_val_f1:.4f} to {best_model_path}")

    return best_model_path


def evaluate_model(model, loader, device, num_classes, max_vis_samples=100):
    model.eval()

    all_pred_labels = []
    all_true_labels = []
    all_probs = []
    all_images_for_vis = []
    collected_vis_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating Model"):
            if images is None or labels is None:
                continue
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_pred_labels.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if collected_vis_samples < max_vis_samples:
                num_to_add = min(images.size(0), max_vis_samples - collected_vis_samples)
                all_images_for_vis.append(images[:num_to_add].cpu())
                collected_vis_samples += num_to_add

    final_preds = np.array(all_pred_labels)
    final_true_labels = np.array(all_true_labels)
    final_probs = np.array(all_probs)

    if all_images_for_vis:
        all_images_for_vis = torch.cat(all_images_for_vis, dim=0)
    else:
        in_channels = loader.dataset.num_image_features if hasattr(loader.dataset, 'num_image_features') else 1
        patch_s = loader.dataset.patch_size if hasattr(loader.dataset, 'patch_size') else 64
        all_images_for_vis = torch.empty((0, in_channels, patch_s, patch_s))

    return final_preds, final_true_labels, final_probs, all_images_for_vis


def calculate_feature_contribution(model, data_loader, device, image_features,
                                   num_samples_to_analyze=200):
    model.eval()
    feature_contributions = np.zeros(len(image_features))
    samples_processed = 0

    if len(data_loader.dataset) == 0:
        print("Dataset empty, skipping feature contribution calculation.")
        return {name: 0.0 for name in image_features}

    pbar = tqdm(data_loader, desc="Calculating Feature Contributions",
                total=min(len(data_loader) * data_loader.batch_size, num_samples_to_analyze), unit="sample")

    for i, (images, labels) in enumerate(pbar):
        if samples_processed >= num_samples_to_analyze:
            break
        if images is None or labels is None:
            continue

        images = images.to(device)
        images.requires_grad_(True)

        outputs = model(images)
        outputs.sum().backward()

        if images.grad is not None:
            batch_contributions = images.grad.abs().sum(dim=(0, 2, 3)).cpu().numpy()
            feature_contributions += batch_contributions
            samples_processed += images.size(0)
            pbar.update(images.size(0))

    pbar.close()

    if samples_processed == 0:
        print("No samples processed for feature contribution calculation.")
        return {name: 0.0 for name in image_features}

    avg_feature_contributions = feature_contributions / samples_processed
    total_contribution = avg_feature_contributions.sum()
    if total_contribution > 1e-8:
        normalized_contributions = avg_feature_contributions / total_contribution
    else:
        normalized_contributions = avg_feature_contributions

    contribution_dict = {
        name: normalized_contributions[i] for i, name in enumerate(image_features)
    }
    return contribution_dict


if __name__ == '__main__':
    # --- Configuration Parameters ---
    csv_path = r"------.csv"
    tif_directory = r"------"
    model_export_dir = r"------"
    os.makedirs(model_export_dir, exist_ok=True)

    batch_size = 32
    patch_size = 64
    use_oversampling = True
    use_augmentation = True
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0

    MODEL_NAME_MAIN = "TransFCN"

    ATTENTION_HEADS = 8

    TOTAL_EPOCHS = 120
    MAX_LR = 1e-2
    FOCAL_LOSS_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1

    dynamic_num_workers = 0

    print("Loading data...")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Error: CSV file not found at {csv_path}")

    # Load and clean CSV
    df = pd.read_csv(csv_path, encoding='gbk').dropna(subset=['x', 'y', 'label', 'image_name'])

    if df.empty:
        raise SystemExit("Error: CSV loaded but no valid data found.")

    df['label'] = df['label'].astype(int)

    # Clean file names in CSV to match TIF directory
    df['image_name_cleaned'] = df['image_name'].astype(str).str.strip().str.replace(r'\.tif$', '', regex=True,
                                                                                    case=False).str.replace(
        r'[_-]?result$', '', regex=True, case=False)

    ms_tif_map = load_tif_mappings(tif_directory)
    if not ms_tif_map:
        print(f"Warning: No TIF files found in {tif_directory}. Data loading will fail.")

    df['image_name'] = df['image_name_cleaned']
    df_filtered = df[df['image_name'].isin(ms_tif_map.keys())].copy()

    if EXCLUDED_CLASSES:
        df_filtered = df_filtered[~df_filtered['label'].isin(EXCLUDED_CLASSES)].copy()

    if len(df_filtered) == 0:
        raise SystemExit("Error: No valid samples available for training. Check filenames and paths.")

    df = df_filtered
    print(f"Total samples after preliminary filtering: {len(df)}")

    unique_csv_labels = sorted(df['label'].unique().tolist())
    if not unique_csv_labels:
        raise SystemExit("Error: No unique labels found in filtered dataset.")

    CLASSIFICATION_CLASSES = []
    CSV_LABEL_TO_OUTPUT_CLASS_ID = {}
    current_output_id = 0
    for csv_label in unique_csv_labels:
        class_name = CLASS_NAMES_MAP.get(csv_label, f'Class_{csv_label}')
        CLASSIFICATION_CLASSES.append(class_name)
        CSV_LABEL_TO_OUTPUT_CLASS_ID[csv_label] = current_output_id
        current_output_id += 1

    NUM_CLASSIFICATION_CLASSES = len(CLASSIFICATION_CLASSES)
    if NUM_CLASSIFICATION_CLASSES == 0:
        raise SystemExit("Error: No classification classes defined.")

    print(f"Classification Classes: {CLASSIFICATION_CLASSES}")
    print(f"Number of Classes: {NUM_CLASSIFICATION_CLASSES}")

    df['stratify_label'] = df['label'].apply(lambda x: CSV_LABEL_TO_OUTPUT_CLASS_ID.get(x, -1))
    df = df[df['stratify_label'] != -1].copy()

    label_counts = df['stratify_label'].value_counts()
    labels_to_keep = label_counts[label_counts > 1].index
    if len(labels_to_keep) < NUM_CLASSIFICATION_CLASSES:
        print(
            f"Warning: Some classes have only one sample and will be excluded from stratified split. Kept classes: {labels_to_keep.tolist()}")
        df = df[df['stratify_label'].isin(labels_to_keep)].copy()
        if not df.empty:
            unique_strat_labels = sorted(df['stratify_label'].unique().tolist())
            if unique_strat_labels != list(range(len(unique_strat_labels))):
                print("Remapping stratify_labels to continuous IDs after filtering.")
                old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(unique_strat_labels)}
                df['stratify_label'] = df['stratify_label'].map(old_to_new_id)
                NUM_CLASSIFICATION_CLASSES = len(unique_strat_labels)
                temp_class_names = [None] * NUM_CLASSIFICATION_CLASSES
                for original_csv_label, original_output_id in list(CSV_LABEL_TO_OUTPUT_CLASS_ID.items()):
                    if original_output_id in old_to_new_id:
                        temp_class_names[old_to_new_id[original_output_id]] = CLASS_NAMES_MAP.get(
                            original_csv_label, f'Class_{original_csv_label}')
                CLASSIFICATION_CLASSES = [name for name in temp_class_names if name is not None]
                CSV_LABEL_TO_OUTPUT_CLASS_ID = {k: old_to_new_id[v] for k, v in CSV_LABEL_TO_OUTPUT_CLASS_ID.items() if
                                                v in unique_strat_labels}
                print(f"New Class Count: {NUM_CLASSIFICATION_CLASSES}")
                print(f"New Classification Classes: {CLASSIFICATION_CLASSES}")

    if len(df) < NUM_CLASSIFICATION_CLASSES * 2:
        raise SystemExit(
            f"Error: Insufficient samples ({len(df)}) to perform stratified split for {NUM_CLASSIFICATION_CLASSES} classes.")
    if len(np.unique(df['stratify_label'])) < 2:
        raise SystemExit("Error: Only one class remains after filtering, classification not possible.")

    # Split: Train (70%), Val (10%), Test (20%)
    tr, te = train_test_split(df, test_size=0.2, stratify=df['stratify_label'], random_state=42)
    tr, val = train_test_split(tr, test_size=0.125, stratify=tr['stratify_label'], random_state=42)

    print("\nCalculated Class Weights (Method: Effective Number of Samples, Beta=0.999):")
    beta = 0.999
    counts = tr['stratify_label'].value_counts().sort_index().to_dict()
    samples_per_cls = [counts.get(i, 0) for i in range(NUM_CLASSIFICATION_CLASSES)]

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    effective_num = np.where(effective_num == 0, 1.0, effective_num)

    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * NUM_CLASSIFICATION_CLASSES

    wts_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(wts_tensor)

    # Pre-calculate stats for normalization
    temp_ds_stats = ClassificationDataset(tr.iloc[:2000], ms_tif_map,
                                          patch_size, None, NUM_CLASSIFICATION_CLASSES, CSV_LABEL_TO_OUTPUT_CLASS_ID,
                                          None, feats=ACTIVE_FEATURES)
    try:
        mean, std = calculate_dataset_stats(temp_ds_stats)
        normalization_stats = (mean, std)
    except ValueError as e:
        print(f"Error: Could not calculate dataset stats: {e}. Normalization skipped.")
        normalization_stats = (None, None)

    train_transform = ClassificationAugmentation() if use_augmentation else None

    # Handle SMOTE high-dim synthesis
    print("Executing SMOTE High-Dimensional Feature Space Synthesis...")
    try:
        temp_list_x = []
        temp_list_y = []

        raw_train_ds = ClassificationDataset(tr, ms_tif_map, patch_size, normalization_stats,
                                             NUM_CLASSIFICATION_CLASSES, CSV_LABEL_TO_OUTPUT_CLASS_ID,
                                             None, feats=ACTIVE_FEATURES)

        for i in tqdm(range(len(raw_train_ds)), desc="Building Feature Space"):
            img_tensor, lbl_tensor = raw_train_ds[i]
            temp_list_x.append(img_tensor.numpy().flatten())
            temp_list_y.append(lbl_tensor.item())

        X_train_flat = np.array(temp_list_x)
        y_train_raw = np.array(temp_list_y)

        min_class_size = np.min(np.bincount(y_train_raw))
        k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1

        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled_flat, y_train_res = sm.fit_resample(X_train_flat, y_train_raw)

        X_train_reshaped = X_resampled_flat.reshape(-1, len(ACTIVE_FEATURES), patch_size, patch_size)

        print(f"SMOTE Synthesis Complete. Original: {len(y_train_raw)}, Balanced: {len(y_train_res)}")


        class BalancedDataset(Dataset):
            def __init__(self, x_array, y_array, transform=None):
                self.x = x_array
                self.y = y_array
                self.transform = transform

            def __len__(self): return len(self.y)

            def __getitem__(self, idx):
                img, lbl = self.x[idx], self.y[idx]
                if self.transform:
                    img, lbl = self.transform(img, lbl)
                return torch.from_numpy(img).float(), torch.tensor(lbl, dtype=torch.long)


        train_ds = BalancedDataset(X_train_reshaped, y_train_res, train_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=dynamic_num_workers, pin_memory=True, drop_last=True)

    except Exception as e:
        print(f"SMOTE Synthesis Failed: {e}. Skipping enhancement.")
        train_ds = ClassificationDataset(tr, ms_tif_map, patch_size, normalization_stats,
                                         NUM_CLASSIFICATION_CLASSES, CSV_LABEL_TO_OUTPUT_CLASS_ID, train_transform,
                                         feats=ACTIVE_FEATURES)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=dynamic_num_workers,
                                  pin_memory=torch.cuda.is_available(), drop_last=True,
                                  collate_fn=collate_fn_classification)

    # Setup Val/Test Data
    val_ds = ClassificationDataset(val, ms_tif_map, patch_size, normalization_stats,
                                   NUM_CLASSIFICATION_CLASSES, CSV_LABEL_TO_OUTPUT_CLASS_ID, None,
                                   feats=ACTIVE_FEATURES)
    test_ds = ClassificationDataset(te, ms_tif_map, patch_size, normalization_stats,
                                    NUM_CLASSIFICATION_CLASSES, CSV_LABEL_TO_OUTPUT_CLASS_ID, None,
                                    feats=ACTIVE_FEATURES)

    val_loader = DataLoader(val_ds, batch_size * 2, shuffle=False, num_workers=dynamic_num_workers,
                            collate_fn=collate_fn_classification, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size * 2, shuffle=False, num_workers=dynamic_num_workers,
                             collate_fn=collate_fn_classification, pin_memory=True)

    # Model Definitions
    num_image_channels = len(ACTIVE_FEATURES)
    models_to_train = {
        MODEL_NAME_MAIN: lambda: TransFCNClassifier(inc=num_image_channels,
                                                    n_cls=NUM_CLASSIFICATION_CLASSES,
                                                    heads=ATTENTION_HEADS),
        "EfficientNetV2": lambda: timm.create_model('tf_efficientnetv2_s', pretrained=False,
                                                    num_classes=NUM_CLASSIFICATION_CLASSES,
                                                    in_chans=num_image_channels),
        "ViT": lambda: timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                         num_classes=NUM_CLASSIFICATION_CLASSES,
                                         in_chans=num_image_channels,
                                         img_size=patch_size),
        "SwinTransformer": lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=False,
                                                     num_classes=NUM_CLASSIFICATION_CLASSES,
                                                     in_chans=num_image_channels,
                                                     img_size=patch_size,
                                                     window_size=4),
        "ConvNeXt": lambda: timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSIFICATION_CLASSES,
                                              in_chans=num_image_channels),
        "MobileViT": lambda: timm.create_model('mobilevit_s', pretrained=False, num_classes=NUM_CLASSIFICATION_CLASSES,
                                               in_chans=num_image_channels),
        "NFNet": lambda: timm.create_model('nfnet_l0', pretrained=False, num_classes=NUM_CLASSIFICATION_CLASSES,
                                           in_chans=num_image_channels),
        "RegNet": lambda: timm.create_model('regnetx_002', pretrained=False, num_classes=NUM_CLASSIFICATION_CLASSES,
                                            in_chans=num_image_channels),
        "DeiT": lambda: timm.create_model('deit_tiny_patch16_224', pretrained=False,
                                          num_classes=NUM_CLASSIFICATION_CLASSES,
                                          in_chans=num_image_channels,
                                          img_size=patch_size),
    }

    final_model_paths = {}

    print(f"\n{'=' * 80}\n{' ' * 20}--- Starting Training and Evaluation ---{' ' * 20}\n{'=' * 80}")

    for model_name, model_fn in models_to_train.items():
        print(f"\n" + "=" * 60)
        print(f"--- Training Model: {model_name} ---")
        print("=" * 60)

        current_model = model_fn().to(device)

        model_state_path = train_model_with_onecyclelr(
            current_model, train_loader, val_loader, device, model_export_dir,
            NUM_CLASSIFICATION_CLASSES, TOTAL_EPOCHS, model_name, wts_tensor,
            CLASSIFICATION_CLASSES, normalization_stats,
            max_lr=MAX_LR, image_features_list=ACTIVE_FEATURES,
            use_cutmix=USE_CUTMIX, cutmix_alpha=CUTMIX_ALPHA,
            focal_loss_gamma=FOCAL_LOSS_GAMMA, label_smoothing=LABEL_SMOOTHING
        )
        final_model_paths[model_name] = model_state_path

    print("\n" + "=" * 60)
    print("--- Final Evaluation and Plotting for All Models ---")
    print("=" * 60)

    all_models_results_final = {}

    for model_name, model_fn in models_to_train.items():
        print(f"\n--- Completing Final Evaluation for {model_name} ---")
        eval_model = model_fn().to(device)

        if model_name in final_model_paths and final_model_paths[model_name] and os.path.exists(
                final_model_paths[model_name]):
            eval_model.load_state_dict(torch.load(final_model_paths[model_name], map_location=device))
        else:
            print(f"Warning: Best state for {model_name} not found. Skipping.")
            continue

        test_preds, test_true_labels, test_probs, test_images_for_vis = evaluate_model(
            eval_model, test_loader, device, NUM_CLASSIFICATION_CLASSES, max_vis_samples=10
        )

        if test_true_labels.size > 0:
            results_dict = analyze_classification_results(
                test_true_labels, test_preds, CLASSIFICATION_CLASSES,
                NUM_CLASSIFICATION_CLASSES, model_export_dir,
                filename_suffix=f"{model_name}_final_test_set", print_report=True
            )
            all_models_results_final[model_name] = results_dict

            # Visualizations
            y_true_one_hot = F.one_hot(torch.from_numpy(test_true_labels),
                                       num_classes=NUM_CLASSIFICATION_CLASSES).numpy()
            plot_roc_curve(y_true_one_hot, test_probs, NUM_CLASSIFICATION_CLASSES, CLASSIFICATION_CLASSES,
                           model_export_dir,
                           model_name=model_name, font_size=26)
            plot_pr_curve(y_true_one_hot, test_probs, NUM_CLASSIFICATION_CLASSES, CLASSIFICATION_CLASSES,
                          model_export_dir,
                          model_name=model_name, font_size=26)

            if test_images_for_vis.numel() > 0 and normalization_stats[0] is not None:
                num_vis_images = test_images_for_vis.shape[0]
                vis_true_labels = test_true_labels[:num_vis_images]
                vis_pred_labels = test_preds[:num_vis_images]

                plot_classification_results(
                    test_images_for_vis, vis_true_labels, vis_pred_labels,
                    CLASSIFICATION_CLASSES, normalization_stats[0], normalization_stats[1], model_export_dir,
                    filename_prefix=f"{model_name}_final_test_predictions",
                    image_features=ACTIVE_FEATURES, font_size=26
                )

                if results_dict['f1_per_class'] is not None:
                    f1_threshold = 0.80
                    low_f1_class_ids = [i for i, f1 in enumerate(results_dict['f1_per_class']) if f1 < f1_threshold]
                    if low_f1_class_ids:
                        plot_low_f1_samples(
                            test_images_for_vis, vis_true_labels, vis_pred_labels, low_f1_class_ids,
                            CLASSIFICATION_CLASSES, normalization_stats, model_export_dir, patch_size,
                            num_samples_per_class=5, filename_prefix=f"{model_name}_final_low_f1_samples",
                            image_features=ACTIVE_FEATURES, font_size=26
                        )

            # Feature contribution for main model only
            if model_name == MODEL_NAME_MAIN:
                print(f"\n--- Calculating Feature Contributions for {model_name} ---")
                feature_contributions = calculate_feature_contribution(eval_model, test_loader, device,
                                                                       ACTIVE_FEATURES,
                                                                       num_samples_to_analyze=200)
                print("Normalized Feature Contributions:")
                for feature, contribution in sorted(feature_contributions.items(), key=lambda item: item[1],
                                                    reverse=True):
                    print(f"  {feature}: {contribution:.4f}")
        else:
            print(f"Skipping visualization for {model_name} as test set is empty.")

    print("\n" + "=" * 60)
    print("--- Final Comparison of All Models ---")
    print("=" * 60)
    results_df_final = pd.DataFrame.from_dict(all_models_results_final, orient='index')
    if 'f1_per_class' in results_df_final.columns:
        results_df_final = results_df_final.drop(columns=['f1_per_class'])
    print(results_df_final.round(4))

    print("\nAll models trained and evaluated. Check the output directory for plots and results.")