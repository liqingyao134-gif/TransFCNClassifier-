import os
import sys
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import ast
import argparse
import random
import json
from tqdm import tqdm
from scipy import ndimage
from math import log2
from scipy.stats import entropy as sci_entropy, skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, canny
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
import cv2

# --- Visualization ---
import matplotlib

matplotlib.use('Agg')  # Backend for saving files without display
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# --- Global Settings ---
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [USER CONFIG] Global Configuration
CONFIG = {
    'SEED': 42,
    'CACHE_DIR': r'./cache_folder',  # Changed to local relative path
    'FONT_FAMILY': 'DejaVu Sans',  # Standard font for Latin names
    'DILATION_RATES': [1, 2, 3, 4],
    'BAND_INDICES': {
        'RED': 0, 'GREEN': 1, 'BLUE': 2, 'NIR': 3, 'RED_EDGE': 4
    }
}

os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
os.environ['TORCH_HOME'] = CONFIG['CACHE_DIR']

# --- Latin Class Mapping ---
# Mapped based on your provided list and standard botanical translations for the gaps.
LATIN_CLASS_NAMES_MAP = {
    0: "Others",
    1: "Leymus chinensis",  # 羊草
    2: "Scirpus triqueter",  # 藨草
    3: "Bolboschoenus maritimus",  # 三棱草
    4: "Calamagrostis epigejos",  # 拂子茅 (Standard fill)
    5: "Carex heterostachya",  # 异穗薹草
    6: "Stipa capillata",  # 针茅
    7: "Lappula myosotis",  # 鹤虱 (Standard fill)
    8: "Suaeda salsa",  # 碱蓬
    9: "Artemisia frigida",  # 冷蒿
    10: "Phragmites australis",  # 芦苇 (Standard fill)
    11: "Medicago sativa",  # 苜蓿
    12: "Potentilla chinensis",  # 委陵菜 (Standard fill)
    13: "Sonchus brachyotus",  # 苦菜
    14: "Stellera chamaejasme",  # 狼毒
}

# Default Feature List (Overwritten by model_config.txt if present)
DEFAULT_FEATURES = [
    'Red', 'Green', 'Blue', 'NIR', 'Red_Edge', 'NDVI', 'SAVI', 'NDWI',
    'NDRE', 'PRI', 'GNDVI', 'EVI', 'OSAVI', 'ARVI', 'GCI', 'NDCI', 'RVI', 'VARI',
    'LBP_Spatial', 'LBP_Mean', 'LBP_StdDev', 'LBP_Entropy',
    'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy',
    'GLCM_ASM', 'GLCM_Correlation',
    'NIR_Skewness', 'NIR_Kurtosis', 'Edge_Density'
]


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


set_seed(CONFIG['SEED'])


# --- Feature Extraction Logic ---

def extract_features(raw_patch_bands, target_size, epsilon=1e-8, feature_list=None):
    """
    Extracts multispectral, texture, and statistical features from a raw image patch.
    """
    if feature_list is None: feature_list = DEFAULT_FEATURES
    num_feats = len(feature_list)

    if raw_patch_bands is None or raw_patch_bands.shape[0] < 5 or target_size <= 0:
        return np.full((num_feats, target_size, target_size), -1.0, dtype=np.float32)

    max_val = 65535.0  # Assuming 16-bit uint

    # Helper for normalization
    def _norm(ch, min_v=-1.0, max_v=1.0):
        p = np.nan_to_num(ch, nan=(min_v + max_v) / 2.0, posinf=max_v, neginf=min_v).clip(min_v, max_v)
        rng = max_v - min_v
        return ((p - min_v) / rng * 2 - 1).astype(np.float32) if rng > epsilon else np.zeros_like(p, dtype=np.float32)

    # 1. Basic Bands
    b = CONFIG['BAND_INDICES']
    try:
        red = (raw_patch_bands[b['RED']].astype(np.float32) / max_val).clip(0, 1)
        green = (raw_patch_bands[b['GREEN']].astype(np.float32) / max_val).clip(0, 1)
        blue = (raw_patch_bands[b['BLUE']].astype(np.float32) / max_val).clip(0, 1)
        nir = (raw_patch_bands[b['NIR']].astype(np.float32) / max_val).clip(0, 1)
        red_edge = (raw_patch_bands[b['RED_EDGE']].astype(np.float32) / max_val).clip(0, 1)
    except:
        return np.full((num_feats, target_size, target_size), -1.0, dtype=np.float32)

    # Pre-calculate GNDVI for Texture features
    gndvi_raw = (nir - green) / (nir + green + epsilon)
    gndvi_8bit = ((np.nan_to_num(gndvi_raw, nan=0.0).clip(-1, 1) + 1) * 127.5).astype(np.uint8)

    calc = {}

    # Spectral Features
    calc['Red'], calc['Green'], calc['Blue'] = _norm(red), _norm(green), _norm(blue)
    calc['NIR'], calc['Red_Edge'] = _norm(nir), _norm(red_edge)

    # Vegetation Indices
    calc['NDVI'] = _norm((nir - red) / (nir + red + epsilon))
    calc['SAVI'] = _norm(((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5)
    calc['NDWI'] = _norm((green - nir) / (green + nir + epsilon))
    calc['NDRE'] = _norm((nir - red_edge) / (nir + red_edge + epsilon))
    calc['PRI'] = _norm((green - blue) / (green + blue + epsilon))
    calc['GNDVI'] = _norm(gndvi_raw)
    calc['EVI'] = _norm(2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + epsilon))
    calc['OSAVI'] = _norm(1.16 * (nir - red) / (nir + red + 0.16 + epsilon))
    calc['ARVI'] = _norm((nir - (2 * red - blue)) / (nir + (2 * red - blue) + epsilon))
    calc['GCI'] = _norm((nir / (green + epsilon)) - 1, -10, 10)
    calc['NDCI'] = _norm((red_edge - blue) / (red_edge + blue + epsilon))
    calc['RVI'] = _norm(nir / (red + epsilon), 0, 10)
    calc['VARI'] = _norm((green - red) / (green + red - blue + epsilon))

    # LBP Features
    lbp_sp, lbp_m, lbp_s, lbp_e = -1.0, -1.0, -1.0, -1.0
    if any('LBP' in f for f in feature_list):
        try:
            lbp = local_binary_pattern(gndvi_8bit, 8, 1, 'uniform') if gndvi_8bit.size > 0 else np.zeros_like(
                gndvi_8bit)
            hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max() + 1), density=True)
            lbp_sp = _norm(lbp, 0, 10)
            lbp_m = _norm(np.array([np.mean(lbp)]), 0, 10)[0]
            lbp_s = _norm(np.array([np.std(lbp)]), 0, 5)[0]
            lbp_e = _norm(np.array([sci_entropy(hist[hist > 0], base=2)]), 0, log2(9))[0]
        except:
            pass
    calc['LBP_Spatial'] = lbp_sp if isinstance(lbp_sp, np.ndarray) else np.full((target_size, target_size), -1.0,
                                                                                dtype=np.float32)
    calc['LBP_Mean'] = np.full((target_size, target_size), lbp_m, dtype=np.float32)
    calc['LBP_StdDev'] = np.full((target_size, target_size), lbp_s, dtype=np.float32)
    calc['LBP_Entropy'] = np.full((target_size, target_size), lbp_e, dtype=np.float32)

    # GLCM Features
    gl_c, gl_d, gl_h, gl_e, gl_a, gl_cor = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    if any('GLCM' in f for f in feature_list):
        try:
            glcm = graycomatrix(gndvi_8bit, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True,
                                normed=True)

            def gp(p):
                return np.nan_to_num(np.mean(graycoprops(glcm, p)), nan=0.0)

            gl_c = _norm(np.array([gp('contrast')]), 0, 65025)[0]
            gl_d = _norm(np.array([gp('dissimilarity')]), 0, 255)[0]
            gl_h = _norm(np.array([gp('homogeneity')]), 0, 1)[0]
            gl_e = _norm(np.array([gp('energy')]), 0, 1)[0]
            gl_a = _norm(np.array([gp('ASM')]), 0, 1)[0]
            gl_cor = _norm(np.array([gp('correlation')]), -1, 1)[0]
        except:
            pass

    for k, v in zip(['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'ASM', 'Correlation'],
                    [gl_c, gl_d, gl_h, gl_e, gl_a, gl_cor]):
        calc[f'GLCM_{k}'] = np.full((target_size, target_size), v, dtype=np.float32)

    # Stats & Edge
    n_sk, n_ku, ed = -1.0, -1.0, -1.0
    try:
        nf = nir.ravel()
        if nf.size > 1:
            n_sk = _norm(np.array([skew(nf)]), -5, 5)[0]
            n_ku = _norm(np.array([kurtosis(nf, fisher=True)]), -3, 20)[0]
    except:
        pass

    try:
        edges = canny(rescale_intensity(nir, out_range=(0, 255)).astype(np.uint8), sigma=1.0)
        ed = _norm(np.array([np.mean(edges)]), 0, 1)[0]
    except:
        pass

    calc['NIR_Skewness'] = np.full((target_size, target_size), n_sk, dtype=np.float32)
    calc['NIR_Kurtosis'] = np.full((target_size, target_size), n_ku, dtype=np.float32)
    calc['Edge_Density'] = np.full((target_size, target_size), ed, dtype=np.float32)

    # Stack requested
    stack = [calc.get(n, np.full((target_size, target_size), -1.0, dtype=np.float32)) for n in feature_list]
    return np.stack(stack, axis=0)


# --- Model Architecture (TransFCN) ---

class ECALayer(nn.Module):
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        k = int(abs(log2(max(1, ch)) + b) / gamma)
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(self.avg(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y).expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, inc, outc, use_att=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, 1, 1), nn.BatchNorm2d(outc)
        )
        self.skip = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()
        self.eca = ECALayer(outc) if use_att else nn.Identity()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.eca(self.conv(x)) + self.skip(x))


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
        self.fuse = nn.Sequential(nn.Conv2d(bc * len(dilations), outc, 1), nn.BatchNorm2d(outc), nn.ReLU(True))
        self.branches = nn.ModuleList(branches)

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
        self.ms = MultiScaleConvBlock(max(1, inc), 64, dilations=CONFIG['DILATION_RATES'])
        self.pool = nn.MaxPool2d(2)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.bot = EncoderBlock(512, 1024)
        self.att = SelfAttentionLayer(1024, heads)

        self.up = lambda x, s: F.interpolate(x, size=s, mode='bilinear', align_corners=False)
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
        e1 = self.ms(x)
        p1 = self.pool(e1)
        e2 = self.e2(p1)
        p2 = self.pool(e2)
        e3 = self.e3(p2)
        p3 = self.pool(e3)
        e4 = self.e4(p3)
        p4 = self.pool(e4)

        b = self.att(self.bot(p4))

        # FPN-like Fusion
        size = e3.shape[2:]
        feat = self.fuse(torch.cat([self.cb(self.up(b, size)), self.c4(self.up(e4, size)), self.c3(e3)], 1))
        return self.cls(feat)


# --- Configuration Loader ---

class ModelConfig:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_config()
        self.load_stats()
        self.load_classes()

    def load_config(self):
        try:
            with open(os.path.join(self.model_dir, "model_config.txt"), 'r') as f:
                cfg = dict(line.strip().split('=', 1) for line in f if '=' in line)

            self.patch_size = int(cfg['patch_size'])
            self.num_channels = int(cfg['num_image_channels'])
            self.features = [x.strip() for x in cfg['image_features'].split(',') if x.strip()]
            self.raw_class_map = ast.literal_eval(cfg.get('csv_label_to_output_class_id', '{}'))
        except Exception as e:
            raise ValueError(f"Failed to load model_config.txt: {e}")

    def load_stats(self):
        self.mean = np.load(os.path.join(self.model_dir, "normalization_mean.npy"))
        self.std = np.load(os.path.join(self.model_dir, "normalization_std.npy"))

    def load_classes(self):
        # Determine inference classes based on training mapping and Latin dictionary
        raw_names = np.load(os.path.join(self.model_dir, "classification_class_names.npy"), allow_pickle=True)
        self.num_model_classes = len(raw_names)

        # Build display map
        # Output ID 0 from model usually maps to some class.
        # We need to map Model_Output_ID -> CSV_ID -> Latin_Name

        # Reverse map: Model_Output_ID -> CSV_Label
        id_to_csv = {v: k for k, v in self.raw_class_map.items()}

        self.inference_classes = ["Unclassified"] * (self.num_model_classes + 1)
        self.inference_classes[0] = LATIN_CLASS_NAMES_MAP.get(0, "Others")

        for out_id in range(self.num_model_classes):
            csv_id = id_to_csv.get(out_id)
            if csv_id is not None:
                self.inference_classes[out_id + 1] = LATIN_CLASS_NAMES_MAP.get(csv_id, f"Unknown_ID_{csv_id}")
            else:
                self.inference_classes[out_id + 1] = str(raw_names[out_id])


# --- Inference Engine ---

def predict_tif(model, tif_path, out_dir, cfg, conf_thresh=0.4, block_factor=4, overlap=2, smooth=3):
    print(f"Processing: {os.path.basename(tif_path)}")
    model.eval()

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    out_png = os.path.join(out_dir, f"{base_name}_Latin_Vis.png")
    out_tif = os.path.join(out_dir, f"{base_name}_pred.tif")

    # Setup Visualization Palette
    n_cls = len(cfg.inference_classes)
    colors = ['#FFFFFF'] + sns.color_palette("hls", n_cls - 1).as_hex()
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(n_cls + 1), cmap.N)

    # Gaussian Weighting for Smooth Overlap
    def get_gaussian(size):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        return np.exp(-(xx ** 2 + yy ** 2) / (2. * (size / 6) ** 2)).astype(np.float32)

    gaussian = get_gaussian(cfg.patch_size)

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        prof = src.profile
        prof.update(dtype=rasterio.uint8, count=1, compress='lzw', blockxsize=256, blockysize=256)
        prof.pop('nodata', None)

        # Block Calculation
        inf_bs = max(cfg.patch_size, (cfg.patch_size * block_factor // cfg.patch_size + 1) * cfg.patch_size)
        step = max(1, cfg.patch_size // overlap)

        with rasterio.open(out_tif, 'w', **prof) as dst:
            for y0 in tqdm(range(0, h, inf_bs), desc="Inference Blocks"):
                for x0 in range(0, w, inf_bs):
                    # Define Windows
                    win_w, win_h = min(inf_bs, w - x0), min(inf_bs, h - y0)
                    out_win = rasterio.windows.Window(x0, y0, win_w, win_h)

                    # Read with padding for context
                    pad = cfg.patch_size
                    read_win = rasterio.windows.Window(
                        max(0, x0 - pad), max(0, y0 - pad),
                        min(w, x0 + win_w + pad) - max(0, x0 - pad),
                        min(h, y0 + win_h + pad) - max(0, y0 - pad)
                    )

                    # Read Bands
                    bands = [1, 2, 3, 4, 5]  # Assuming standard bands present
                    raw = src.read(bands, window=read_win, boundless=True, fill_value=0)
                    if raw.shape[0] < 5:
                        raw = np.pad(raw, ((0, 5 - raw.shape[0]), (0, 0), (0, 0)))

                    # Accumulators
                    acc_prob = np.zeros((win_h, win_w, n_cls), dtype=np.float32)
                    acc_weight = np.zeros((win_h, win_w), dtype=np.float32)

                    # Sliding Window inside Block
                    rh, rw = raw.shape[1], raw.shape[2]
                    off_x, off_y = x0 - read_win.col_off, y0 - read_win.row_off

                    for py in range(0, win_h, step):
                        for px in range(0, win_w, step):
                            # Extract Patch from Raw Buffer
                            sy, sx = off_y + py, off_x + px
                            if sy < 0 or sx < 0 or sy + cfg.patch_size > rh or sx + cfg.patch_size > rw: continue

                            patch = raw[:, sy:sy + cfg.patch_size, sx:sx + cfg.patch_size]
                            if np.all(patch == 0): continue

                            # Feature Extraction & Inference
                            feats = extract_features(patch, cfg.patch_size, feature_list=cfg.features)
                            feats = (feats - cfg.mean.reshape(-1, 1, 1)) / (cfg.std.reshape(-1, 1, 1) + 1e-8)

                            inp = torch.from_numpy(np.clip(feats, -1, 1)).float().unsqueeze(0).to(device)
                            with torch.no_grad():
                                prob = F.softmax(model(inp), 1).cpu().numpy().flatten()

                            # Add to Accumulator (weighted)
                            full_prob = np.zeros(n_cls);
                            full_prob[1:] = prob

                            dy, dx = min(win_h, py + cfg.patch_size), min(win_w, px + cfg.patch_size)
                            h_slice, w_slice = dy - py, dx - px

                            w_mask = gaussian[:h_slice, :w_slice]
                            acc_prob[py:dy, px:dx] += w_mask[..., None] * full_prob
                            acc_weight[py:dy, px:dx] += w_mask

                    # Normalize & Argmax
                    mask = acc_weight > 0
                    final_block = np.zeros((win_h, win_w), dtype=np.uint8)

                    if np.any(mask):
                        avg_prob = np.zeros_like(acc_prob)
                        avg_prob[mask] = acc_prob[mask] / acc_weight[mask, None]
                        preds = np.argmax(avg_prob, axis=2)
                        confs = np.max(avg_prob, axis=2)

                        # Thresholding
                        preds[(confs < conf_thresh) | ~mask] = 0
                        final_block = preds.astype(np.uint8)

                    # Smoothing
                    if smooth > 1:
                        final_block = ndimage.median_filter(final_block, size=smooth)

                    # Write
                    dst.write(final_block, 1, window=out_win)

    # Visualization Generation
    print("Generating Plot...")
    with rasterio.open(out_tif) as src:
        res_map = src.read(1)

    # Calculate Stats
    counts = np.bincount(res_map.flatten(), minlength=n_cls)
    total = res_map.size
    print("\n--- Statistics (Latin) ---")
    for i, name in enumerate(cfg.inference_classes):
        if counts[i] > 0:
            print(f"  {name}: {counts[i]} ({counts[i] / total * 100:.2f}%)")

    # Plot
    plt.figure(figsize=(12, 12))
    plt.rcParams['font.family'] = CONFIG['FONT_FAMILY']
    im = plt.imshow(res_map, cmap=cmap, norm=norm)

    # Legend
    cbar = plt.colorbar(im, ticks=np.arange(n_cls) + 0.5, shrink=0.7)
    cbar.ax.set_yticklabels([cfg.inference_classes[i] if i < n_cls else '' for i in range(n_cls)])
    cbar.ax.tick_params(labelsize=10)

    plt.title(f"{base_name} Classification", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")


# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multispectral Inference (TransFCN) with Latin Nomenclature")
    parser.add_argument('--tif_dir', type=str, required=True, help="Directory containing .tif images")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing model checkpoint and config")
    parser.add_argument('--model_name', type=str, default="TransFCN_Optimized_FocalLoss_FPN",
                        help="Base name of the model file")
    args = parser.parse_args()

    # Load Config
    try:
        cfg = ModelConfig(args.model_dir)
        print(f"Loaded Config. Patch: {cfg.patch_size}, Channels: {cfg.num_channels}")
    except Exception as e:
        print(f"Config Error: {e}")
        sys.exit(1)

    # Load Model
    model = TransFCNClassifier(cfg.num_channels, cfg.num_model_classes).to(device)
    ckpt_path = os.path.join(args.model_dir, f"{args.model_name}_best_model.pth")

    if not os.path.exists(ckpt_path):
        print(f"Error: Model not found at {ckpt_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Model Weights Loaded.")

    # Run Inference
    out_dir = os.path.join(args.model_dir, "inference_results_latin")
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(args.tif_dir):
        if f.lower().endswith('.tif'):
            predict_tif(model, os.path.join(args.tif_dir, f), out_dir, cfg)

    print("\nProcessing Complete.")