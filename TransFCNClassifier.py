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
    matthews_corrcoef
from imblearn.over_sampling import SMOTE
import re
from tqdm import tqdm
import random
from scipy import ndimage
from math import log2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.exposure import rescale_intensity
from scipy.stats import entropy as sci_entropy
from scipy.stats import skew, kurtosis
from skimage.feature import canny
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Settings ---
os.environ['TORCH_HOME'] = r'D:\big\Cache_Folder'
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# --- Constants ---
RED_IDX, GREEN_IDX, BLUE_IDX, NIR_IDX, RED_EDGE_IDX = 0, 1, 2, 3, 4
ALL_IMAGE_FEATURES = [
    'Red', 'Green', 'Blue', 'NIR', 'Red_Edge',
    'NDVI', 'SAVI', 'NDWI', 'NDRE', 'PRI', 'GNDVI', 'EVI', 'OSAVI',
    'ARVI', 'GCI', 'NDCI', 'RVI', 'VARI',
    'LBP_Spatial', 'LBP_Mean', 'LBP_StdDev', 'LBP_Entropy',
    'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy',
    'GLCM_ASM', 'GLCM_Correlation',
    'NIR_Skewness', 'NIR_Kurtosis', 'Edge_Density'
]
TOP_FEATURES_TO_USE = [
    'VARI', 'RVI', 'LBP_Spatial', 'GLCM_Correlation', 'NDWI',
    'LBP_Entropy', 'GCI', 'LBP_Mean', 'NDRE', 'Edge_Density',
    'PRI', 'ARVI', 'NIR_Kurtosis', 'NIR_Skewness', 'LBP_StdDev',
    'GLCM_Homogeneity', 'NDCI', 'OSAVI', 'GNDVI', 'SAVI'
]
ACTIVE_FEATURES = TOP_FEATURES_TO_USE
DILATION_RATES = [1, 2, 3, 4]


def load_tif_mappings(tif_directory):
    print("Scanning TIF files...")
    ms_tif_map = {}
    if os.path.exists(tif_directory):
        for f in os.listdir(tif_directory):
            if f.lower().endswith('.tif'):
                base_name = re.sub(r'[_-]?result$', '', os.path.splitext(f)[0], flags=re.IGNORECASE)
                ms_tif_map[base_name] = os.path.join(tif_directory, f)
    print(f"Found {len(ms_tif_map)} multispectral files.")
    return ms_tif_map


def calculate_image_channels(raw_patch_bands, target_size, epsilon=1e-8, feature_list=None):
    if feature_list is None: feature_list = ALL_IMAGE_FEATURES
    num_expected_features = len(feature_list)

    if raw_patch_bands is None or raw_patch_bands.shape[0] < 5:
        return np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)

    max_ms_val = 65535.0

    def _process(ch, min_v=-1.0, max_v=1.0):
        p = np.nan_to_num(ch, nan=(min_v + max_v) / 2.0, posinf=max_v, neginf=min_v).clip(min_v, max_v)
        rng = max_v - min_v
        return ((p - min_v) / rng * 2 - 1).astype(np.float32) if rng > epsilon else np.zeros_like(p, dtype=np.float32)

    try:
        red = (raw_patch_bands[RED_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        green = (raw_patch_bands[GREEN_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        blue = (raw_patch_bands[BLUE_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        nir = (raw_patch_bands[NIR_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
        red_edge = (raw_patch_bands[RED_EDGE_IDX].astype(np.float32) / max_ms_val).clip(0, 1)
    except:
        return np.full((num_expected_features, target_size, target_size), -1.0, dtype=np.float32)

    gndvi = (nir - green) / (nir + green + epsilon)
    gndvi_8bit = ((np.nan_to_num(gndvi, nan=0.0).clip(-1, 1) + 1) * 127.5).astype(np.uint8)

    if gndvi_8bit.size == 0 or np.all(gndvi_8bit == gndvi_8bit.flat[0]):
        vegetation_mask = np.ones_like(gndvi_8bit, dtype=bool)
    else:
        try:
            _, otsu = cv2.threshold(gndvi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            vegetation_mask = otsu.astype(bool)
        except:
            vegetation_mask = gndvi > np.mean(gndvi)

    calc = {}
    fill = -1.0

    # Basic Bands
    calc['Red'], calc['Green'], calc['Blue'] = _process(red), _process(green), _process(blue)
    calc['NIR'], calc['Red_Edge'] = _process(nir), _process(red_edge)

    # Indices
    calc['NDVI'] = _process(np.where(vegetation_mask, (nir - red) / (nir + red + epsilon), fill))
    calc['SAVI'] = _process(np.where(vegetation_mask, ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5, fill))
    calc['NDWI'] = _process(np.where(vegetation_mask, (green - nir) / (green + nir + epsilon), fill))
    calc['NDRE'] = _process(np.where(vegetation_mask, (nir - red_edge) / (nir + red_edge + epsilon), fill))
    calc['PRI'] = _process(np.where(vegetation_mask, (green - blue) / (green + blue + epsilon), fill))
    calc['GNDVI'] = _process(np.where(vegetation_mask, gndvi, fill))
    calc['EVI'] = _process(
        np.where(vegetation_mask, 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + epsilon), fill))
    calc['OSAVI'] = _process(np.where(vegetation_mask, (1.16) * (nir - red) / (nir + red + 0.16 + epsilon), fill))
    calc['ARVI'] = _process(
        np.where(vegetation_mask, (nir - (2 * red - blue)) / (nir + (2 * red - blue) + epsilon), fill))
    calc['GCI'] = _process(np.where(vegetation_mask, (nir / (green + epsilon)) - 1, fill), -10, 10)
    calc['NDCI'] = _process(np.where(vegetation_mask, (red_edge - blue) / (red_edge + blue + epsilon), fill))
    calc['RVI'] = _process(np.where(vegetation_mask, nir / (red + epsilon), fill), 0, 10)
    calc['VARI'] = _process(np.where(vegetation_mask, (green - red) / (green + red - blue + epsilon), fill))

    # LBP
    lbp_sp, lbp_m, lbp_s, lbp_e = fill, fill, fill, fill
    if any('LBP' in f for f in feature_list):
        try:
            lbp = local_binary_pattern(gndvi_8bit, 8, 1, 'uniform') if gndvi_8bit.size > 0 else np.zeros_like(
                gndvi_8bit)
            lbp_sp = _process(lbp, 0, 10)
            lbp_m = _process(np.array([np.mean(lbp)]), 0, 10)[0]
            lbp_s = _process(np.array([np.std(lbp)]), 0, 5)[0]
            hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max() + 1), density=True)
            lbp_e = _process(np.array([sci_entropy(hist[hist > 0], base=2)]), 0, log2(9))[0]
        except:
            pass
    calc['LBP_Spatial'] = lbp_sp if isinstance(lbp_sp, np.ndarray) else np.full((target_size, target_size), fill,
                                                                                dtype=np.float32)
    calc['LBP_Mean'] = np.full((target_size, target_size), lbp_m, dtype=np.float32)
    calc['LBP_StdDev'] = np.full((target_size, target_size), lbp_s, dtype=np.float32)
    calc['LBP_Entropy'] = np.full((target_size, target_size), lbp_e, dtype=np.float32)

    # GLCM
    gl_c, gl_d, gl_h, gl_e, gl_a, gl_cor = fill, fill, fill, fill, fill, fill
    if any('GLCM' in f for f in feature_list):
        try:
            glcm = graycomatrix(gndvi_8bit, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True,
                                normed=True)

            def gp(p):
                return np.nan_to_num(np.mean(graycoprops(glcm, p)), nan=0.0)

            gl_c = _process(np.array([gp('contrast')]), 0, 65025)[0]
            gl_d = _process(np.array([gp('dissimilarity')]), 0, 255)[0]
            gl_h = _process(np.array([gp('homogeneity')]), 0, 1)[0]
            gl_e = _process(np.array([gp('energy')]), 0, 1)[0]
            gl_a = _process(np.array([gp('ASM')]), 0, 1)[0]
            gl_cor = _process(np.array([gp('correlation')]), -1, 1)[0]
        except:
            pass

    for k, v in zip(['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'ASM', 'Correlation'],
                    [gl_c, gl_d, gl_h, gl_e, gl_a, gl_cor]):
        calc[f'GLCM_{k}'] = np.full((target_size, target_size), v, dtype=np.float32)

    # Stats
    n_sk, n_ku = fill, fill
    if any('NIR' in f for f in feature_list):
        try:
            nf = nir.ravel()
            if nf.size > 1:
                n_sk = _process(np.array([skew(nf)]), -5, 5)[0]
                n_ku = _process(np.array([kurtosis(nf, fisher=True)]), -3, 20)[0]
        except:
            pass
    calc['NIR_Skewness'] = np.full((target_size, target_size), n_sk, dtype=np.float32)
    calc['NIR_Kurtosis'] = np.full((target_size, target_size), n_ku, dtype=np.float32)

    # Edge
    ed = fill
    if 'Edge_Density' in feature_list:
        try:
            edges = canny(rescale_intensity(nir, out_range=(0, 255)).astype(np.uint8), sigma=1.0)
            ed = _process(np.array([np.mean(edges)]), 0, 1)[0]
        except:
            pass
    calc['Edge_Density'] = np.full((target_size, target_size), ed, dtype=np.float32)

    stack = [calc.get(n, np.full((target_size, target_size), fill, dtype=np.float32)) for n in feature_list]
    return np.stack(stack, axis=0)


class ClassificationAugmentation:
    def __init__(self):
        self.rot_p, self.flip_p, self.noise_p, self.noise_std = 0.7, 0.7, 0.4, 0.05
        self.cut_p, self.cut_ratio, self.cut_fill = 0.5, 0.15, -1.0

    def __call__(self, img, target):
        img = img.astype(np.float32)
        if random.random() < self.rot_p: img = ndimage.rotate(img, random.choice([90, 180, 270]), axes=(1, 2),
                                                              reshape=False, order=1, mode='reflect')
        if random.random() < self.flip_p: img = np.flip(img, axis=2).copy()
        if random.random() < self.flip_p: img = np.flip(img, axis=1).copy()
        if random.random() < self.noise_p: img += np.random.normal(0, self.noise_std, img.shape)
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


def cutmix_data(imgs, lbls, alpha):
    if alpha <= 0: return imgs, lbls, lbls, 1.0
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


# --- Visualization & Analysis Helpers ---
def plot_classification_results(images, true_lbl, pred_lbl, names, mean, std, dir, prefix="", feats=None):
    os.makedirs(dir, exist_ok=True)
    feats = feats or ACTIVE_FEATURES
    idx = {n: feats.index(n) for n in ['Red', 'Green', 'Blue', 'NIR'] if n in feats}

    for i in range(min(images.shape[0], 10)):
        img = images[i].cpu().numpy()
        if mean is not None: img = img * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
        img = np.clip((img + 1) / 2, 0, 1)

        if {'Red', 'Green', 'Blue'} <= idx.keys():
            disp = np.stack([img[idx['Red']], img[idx['Green']], img[idx['Blue']]], -1)
        elif {'NIR', 'Red', 'Green'} <= idx.keys():
            disp = np.stack([img[idx['NIR']], img[idx['Red']], img[idx['Green']]], -1)
        else:
            disp = img[0]

        plt.figure(figsize=(4, 4))
        plt.imshow(disp, cmap='gray' if disp.ndim == 2 else None)
        t_name = names[true_lbl[i]] if true_lbl[i] < len(names) else str(true_lbl[i])
        p_name = names[pred_lbl[i]] if pred_lbl[i] < len(names) else str(pred_lbl[i])
        plt.title(f'T: {t_name}\nP: {p_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{prefix}_{i}.png"))
        plt.close()


def analyze_classification_results(true, pred, names, num, dir, suffix=""):
    print(f"\n--- Analysis ({suffix}) ---")
    acc = accuracy_score(true, pred)
    wf1 = f1_score(true, pred, average='weighted', zero_division=0)
    mf1 = f1_score(true, pred, average='macro', zero_division=0)
    kap = cohen_kappa_score(true, pred)
    mcc = matthews_corrcoef(true, pred)

    print(f"OA: {acc:.4f}, Weighted F1: {wf1:.4f}, Macro F1: {mf1:.4f}, Kappa: {kap:.4f}, MCC: {mcc:.4f}")

    cm = confusion_matrix(true, pred, labels=np.arange(num))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=names, yticklabels=names, cmap='Blues', cbar=False)
    plt.ylabel('True');
    plt.xlabel('Pred');
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"cm_{suffix}.png"))
    plt.close()

    return f1_score(true, pred, average=None, labels=np.arange(num), zero_division=0)


def calculate_dataset_stats(dataset, max_n=1000):
    print("Calc Stats...")
    ldr = DataLoader(dataset, batch_size=1, shuffle=False)
    feats = []
    for i, (img, _) in enumerate(tqdm(ldr, total=max_n)):
        if i >= max_n: break
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


def train(model, tr_ldr, val_ldr, dev, dir, n_cls, ep, name, wts, names, stats, lr, acc_steps, feats, cmix, cmix_a, ls):
    print(f"Start Train: {name}, Ep: {ep}, LR: {lr}")
    crit = nn.CrossEntropyLoss(weight=wts, label_smoothing=ls).to(dev)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=ep * len(tr_ldr))
    best_f1 = -1.0

    for e in range(ep):
        model.train()
        rl = 0.0
        pbar = tqdm(tr_ldr, desc=f"Ep {e + 1}/{ep}", leave=False)

        for i, (img, lbl) in enumerate(pbar):
            img, lbl = img.to(dev), lbl.to(dev)

            if cmix and img.size(0) > 1:
                img, la, lb, lam = cutmix_data(img, lbl, cmix_a)
                out = model(img)
                loss = lam * crit(out, la) + (1 - lam) * crit(out, lb)
            else:
                loss = crit(model(img), lbl)

            loss /= acc_steps
            loss.backward()

            if (i + 1) % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step();
                opt.zero_grad();
                sch.step()

            rl += loss.item() * acc_steps
            pbar.set_postfix(loss=rl / (i + 1))

        # Validation
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for img, lbl in val_ldr:
                out = model(img.to(dev))
                trues.extend(lbl.cpu().numpy())
                preds.extend(torch.max(out, 1)[1].cpu().numpy())

        wf1 = f1_score(trues, preds, average='weighted', zero_division=0)
        print(f"Ep {e + 1}: Val W-F1: {wf1:.4f}")

        if wf1 > best_f1:
            best_f1 = wf1
            torch.save(model.state_dict(), os.path.join(dir, f"{name}_best.pth"))

    return os.path.join(dir, f"{name}_best.pth")


if __name__ == '__main__':
    # --- Config ---
    csv_path = r"D:\big\model_classification_multi_class_optimized\filtered_full4.csv"
    tif_dir = r"D:\big\tif"
    out_dir = r"D:\big\model_classification_multi_class_optimized"
    batch_size, patch_size = 32, 64
    EPOCHS, MAX_LR = 120, 1e-2

    print(f"Setup: {device}, Batch: {batch_size}, Patch: {patch_size}, Feats: {len(ACTIVE_FEATURES)}")

    # --- Data Prep ---
    df = pd.read_csv(csv_path, encoding='gbk').dropna(subset=['x', 'y', 'label', 'image_name'])
    df['label'] = df['label'].astype(int)
    if EXCLUDED_CLASSES: df = df[~df['label'].isin(EXCLUDED_CLASSES)]

    tif_map = load_tif_mappings(tif_dir)
    df = df[df['image_name'].astype(str).str.replace(r'\.tif$', '', regex=True).isin(tif_map.keys())].copy()

    u_lbl = sorted(df['label'].unique())
    CLS_NAMES = [f'type_{l}' for l in u_lbl]
    CLS_MAP = {l: i for i, l in enumerate(u_lbl)}
    N_CLS = len(CLS_NAMES)
    print(f"Classes: {N_CLS} -> {CLS_MAP}")

    df['sid'] = df['label'].map(CLS_MAP)
    tr, te = train_test_split(df, test_size=0.2, stratify=df['sid'], random_state=42)
    tr, val = train_test_split(tr, test_size=0.125, stratify=tr['sid'], random_state=42)

    # --- Class Weights Calculation (Updated to Effective Number of Samples) ---
    print("\nCalculated Class Weights (Method: Effective Number of Samples, Beta=0.999):")
    beta = 0.999
    counts = tr['sid'].value_counts().sort_index().to_dict()
    samples_per_cls = [counts.get(i, 0) for i in range(N_CLS)]

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    # Handle possible division by zero if count is 0, though stratify usually prevents this
    effective_num = np.where(effective_num == 0, 1.0, effective_num)

    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * N_CLS  # Normalize

    wts_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(wts_tensor)

    # --- SMOTE ---
    print("SMOTE...")
    try:
        sm = SMOTE(random_state=42, k_neighbors=min(tr['sid'].value_counts().min() - 1, 5))
        idx_res, y_res = sm.fit_resample(np.arange(len(tr)).reshape(-1, 1), tr['sid'])
        tr = tr.iloc[idx_res.flatten()].copy()
    except Exception as e:
        print(f"SMOTE skipped: {e}")

    # --- Loaders ---
    print("Dataset Stats...")
    tmp_ds = ClassificationDataset(tr.iloc[:2000], tif_map, patch_size, None, N_CLS, CLS_MAP, feats=ACTIVE_FEATURES)
    stats = calculate_dataset_stats(tmp_ds)

    aug = ClassificationAugmentation()
    tr_ds = ClassificationDataset(tr, tif_map, patch_size, stats, N_CLS, CLS_MAP, aug, ACTIVE_FEATURES)
    val_ds = ClassificationDataset(val, tif_map, patch_size, stats, N_CLS, CLS_MAP, None, ACTIVE_FEATURES)
    te_ds = ClassificationDataset(te, tif_map, patch_size, stats, N_CLS, CLS_MAP, None, ACTIVE_FEATURES)

    tr_l = DataLoader(tr_ds, batch_size, True, num_workers=0, collate_fn=collate_fn_classification, drop_last=True)
    val_l = DataLoader(val_ds, batch_size * 2, False, num_workers=0, collate_fn=collate_fn_classification)
    te_l = DataLoader(te_ds, batch_size * 2, False, num_workers=0, collate_fn=collate_fn_classification)

    # --- Train & Eval ---
    model = TransFCNClassifier(len(ACTIVE_FEATURES), N_CLS).to(device)
    best = train(model, tr_l, val_l, device, out_dir, N_CLS, EPOCHS, "TransFCN", wts_tensor, CLS_NAMES, stats, MAX_LR,
                 4, ACTIVE_FEATURES, True, 1.0, 0.1)

    print("\nEvaluation...")
    model.load_state_dict(torch.load(best))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for img, lbl in tqdm(te_l):
            out = model(img.to(device))
            all_true.extend(lbl.cpu().numpy())
            all_pred.extend(torch.max(out, 1)[1].cpu().numpy())

    analyze_classification_results(np.array(all_true), np.array(all_pred), CLS_NAMES, N_CLS, out_dir, "test")

    # Save Metadata
    np.save(os.path.join(out_dir, "class_names.npy"), CLS_NAMES)
    if stats[0] is not None:
        np.save(os.path.join(out_dir, "norm_mean.npy"), stats[0])
        np.save(os.path.join(out_dir, "norm_std.npy"), stats[1])