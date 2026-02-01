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
from math import sqrt, log2
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

os.environ['TORCH_HOME'] = r'D:\big\Cache_Folder'
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RED_IDX, GREEN_IDX, BLUE_IDX, NIR_IDX, RED_EDGE_IDX = 0, 1, 2, 3, 4
ALL_IMAGE_FEATURES = [
    'Red', 'Green', 'Blue', 'NIR', 'Red_Edge', 'NDVI', 'SAVI', 'NDWI',
    'NDRE', 'PRI', 'GNDVI', 'EVI', 'OSAVI', 'ARVI', 'GCI', 'NDCI', 'RVI', 'VARI',
    'LBP_Spatial', 'LBP_Mean', 'LBP_StdDev', 'LBP_Entropy',
    'GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy',
    'GLCM_ASM', 'GLCM_Correlation', 'NIR_Skewness', 'NIR_Kurtosis', 'Edge_Density'
]
ACTIVE_FEATURES = [
    'VARI', 'RVI', 'LBP_Spatial', 'GLCM_Correlation', 'NDWI', 'LBP_Entropy', 'GCI',
    'LBP_Mean', 'NDRE', 'Edge_Density', 'PRI', 'ARVI', 'NIR_Kurtosis', 'NIR_Skewness',
    'LBP_StdDev', 'GLCM_Homogeneity', 'NDCI', 'OSAVI', 'GNDVI', 'SAVI'
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


set_seed(42)


def load_tif_mappings(tif_directory):
    ms_tif_map = {}
    if os.path.exists(tif_directory):
        for f in os.listdir(tif_directory):
            if f.lower().endswith('.tif'):
                base_name = re.sub(r'[_-]?result$', '', os.path.splitext(f)[0], flags=re.IGNORECASE)
                ms_tif_map[base_name] = os.path.join(tif_directory, f)
    return ms_tif_map


def calculate_image_channels(raw_patch_bands, target_size, epsilon=1e-8, feature_list=None):
    if feature_list is None: feature_list = ALL_IMAGE_FEATURES
    if raw_patch_bands is None or raw_patch_bands.shape[0] < 5:
        return np.full((len(feature_list), target_size, target_size), -1.0, dtype=np.float32)

    def _process_channel(channel_data, min_val=-1.0, max_val=1.0):
        processed = np.nan_to_num(channel_data, nan=(min_val + max_val) / 2.0, posinf=max_val, neginf=min_val)
        processed = np.clip(processed, min_val, max_val)
        return ((processed - min_val) / (max_val - min_val + epsilon) * 2 - 1).astype(np.float32)

    red = (raw_patch_bands[RED_IDX].astype(np.float32) / 65535.0).clip(0, 1)
    green = (raw_patch_bands[GREEN_IDX].astype(np.float32) / 65535.0).clip(0, 1)
    blue = (raw_patch_bands[BLUE_IDX].astype(np.float32) / 65535.0).clip(0, 1)
    nir = (raw_patch_bands[NIR_IDX].astype(np.float32) / 65535.0).clip(0, 1)
    red_edge = (raw_patch_bands[RED_EDGE_IDX].astype(np.float32) / 65535.0).clip(0, 1)

    gndvi = (nir - green) / (nir + green + epsilon)
    gndvi_8bit = ((np.nan_to_num(gndvi, nan=0.0).clip(-1, 1) + 1) * 127.5).astype(np.uint8)
    try:
        _, otsu_mask = cv2.threshold(gndvi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vegetation_mask = otsu_mask.astype(bool)
    except:
        vegetation_mask = gndvi > np.mean(gndvi)

    calc = {}
    calc['Red'], calc['Green'], calc['Blue'], calc['NIR'], calc['Red_Edge'] = _process_channel(red, 0,
                                                                                               1), _process_channel(
        green, 0, 1), _process_channel(blue, 0, 1), _process_channel(nir, 0, 1), _process_channel(red_edge, 0, 1)

    calc['NDVI'] = _process_channel(np.where(vegetation_mask, (nir - red) / (nir + red + epsilon), -1.0))
    calc['SAVI'] = _process_channel(np.where(vegetation_mask, ((nir - red) / (nir + red + 0.5 + epsilon)) * 1.5, -1.0))
    calc['NDWI'] = _process_channel(np.where(vegetation_mask, (green - nir) / (green + nir + epsilon), -1.0))
    calc['NDRE'] = _process_channel(np.where(vegetation_mask, (nir - red_edge) / (nir + red_edge + epsilon), -1.0))
    calc['PRI'] = _process_channel(np.where(vegetation_mask, (green - blue) / (green + blue + epsilon), -1.0))
    calc['GNDVI'] = _process_channel(np.where(vegetation_mask, gndvi, -1.0))
    calc['EVI'] = _process_channel(
        np.where(vegetation_mask, 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + epsilon), -1.0))
    calc['OSAVI'] = _process_channel(
        np.where(vegetation_mask, (1 + 0.16) * (nir - red) / (nir + red + 0.16 + epsilon), -1.0))
    calc['ARVI'] = _process_channel(
        np.where(vegetation_mask, (nir - (2 * red - blue)) / (nir + (2 * red - blue) + epsilon), -1.0))
    calc['GCI'] = _process_channel(np.where(vegetation_mask, (nir / (green + epsilon)) - 1, -1.0), -10, 10)
    calc['NDCI'] = _process_channel(np.where(vegetation_mask, (red_edge - blue) / (red_edge + blue + epsilon), -1.0))
    calc['RVI'] = _process_channel(np.where(vegetation_mask, nir / (red + epsilon), -1.0), 0, 10)
    calc['VARI'] = _process_channel(np.where(vegetation_mask, (green - red) / (green + red - blue + epsilon), -1.0))

    lbp = local_binary_pattern(gndvi_8bit, 8, 1, 'uniform')
    calc['LBP_Spatial'] = _process_channel(lbp, 0, 10)
    calc['LBP_Mean'] = np.full((target_size, target_size), _process_channel(np.array([np.mean(lbp)]), 0, 10)[0])
    calc['LBP_StdDev'] = np.full((target_size, target_size), _process_channel(np.array([np.std(lbp)]), 0, 5)[0])
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    calc['LBP_Entropy'] = np.full((target_size, target_size),
                                  _process_channel(np.array([sci_entropy(hist_lbp[hist_lbp > 0], base=2)]), 0,
                                                   log2(10))[0])

    glcm = graycomatrix(gndvi_8bit, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                        normed=True)
    for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'ASM', 'correlation']:
        val = np.mean(graycoprops(glcm, p))
        mx = 65025 if p == 'contrast' else (255 if p == 'dissimilarity' else 1)
        calc[f'GLCM_{p.capitalize()}'] = np.full((target_size, target_size),
                                                 _process_channel(np.array([val]), -1 if p == 'correlation' else 0, mx)[
                                                     0])

    calc['NIR_Skewness'] = np.full((target_size, target_size),
                                   _process_channel(np.array([skew(nir.ravel())]), -5, 5)[0])
    calc['NIR_Kurtosis'] = np.full((target_size, target_size),
                                   _process_channel(np.array([kurtosis(nir.ravel())]), -3, 20)[0])
    edges = canny(rescale_intensity(nir, out_range=(0, 255)).astype(np.uint8), sigma=1.0)
    calc['Edge_Density'] = np.full((target_size, target_size), _process_channel(np.array([np.mean(edges)]), 0, 1)[0])

    return np.stack([calc.get(name, np.full((target_size, target_size), -1.0)) for name in feature_list], axis=0)


class ClassificationDataset(Dataset):
    def __init__(self, data_array, labels, transform=None):
        self.data = data_array
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img, lbl = self.data[idx], self.labels[idx]
        if self.transform: img, lbl = self.transform(img, lbl)
        return torch.from_numpy(img).float(), torch.tensor(lbl, dtype=torch.long)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k = int(abs(log2(channel) + b) / gamma)
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                                  nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c))
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.eca = ECALayer(out_c)

    def forward(self, x): return F.relu(self.eca(self.conv(x) + self.shortcut(x)))


class TransFCNClassifier(nn.Module):
    def __init__(self, in_c, num_cls):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(256, num_cls))

    def forward(self, x):
        x = self.pool(self.enc1(self.init_conv(x)))
        x = self.pool(self.enc2(x))
        x = self.enc3(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1))


def cutmix_data(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0)).to(images.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    return images, labels, labels[rand_index], lam


def rand_bbox(size, lam):
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H), np.clip(cx + cut_w // 2, 0, W), np.clip(
        cy + cut_h // 2, 0, H)


if __name__ == '__main__':
    csv_path = r"-----.csv"
    tif_dir = r"------"
    out_dir = r"------"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, encoding='gbk').dropna(subset=['x', 'y', 'label', 'image_name'])
    tif_map = load_tif_mappings(tif_dir)
    df = df[df['image_name'].isin(tif_map.keys()) & (df['label'] != 0)].copy()

    unique_labels = sorted(df['label'].unique())
    label_map = {l: i for i, l in enumerate(unique_labels)}
    df['target'] = df['label'].map(label_map)

    print("Pre-loading images and calculating features...")
    all_imgs, all_lbls = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        with rasterio.open(tif_map[row['image_name']]) as src:
            r, c = src.index(row['x'], row['y'])
            win = rasterio.windows.Window(c - 32, r - 32, 64, 64)
            raw = src.read([1, 2, 3, 4, 5], window=win, boundless=True, fill_value=0)
            feat = calculate_image_channels(raw, 64, feature_list=ACTIVE_FEATURES)
            all_imgs.append(feat);
            all_lbls.append(row['target'])

    X, y = np.array(all_imgs), np.array(all_lbls)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train,
                                                      random_state=42)

    print("Applying SMOTE on flattened feature vectors...")
    N, C, H, W = X_train.shape
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(y_train)) - 1))
    X_train_reshaped, y_train_res = smote.fit_resample(X_train.reshape(N, -1), y_train)
    X_train_res = X_train_reshaped.reshape(-1, C, H, W)
    print(f"SMOTE complete. New training size: {len(y_train_res)}")

    train_loader = DataLoader(ClassificationDataset(X_train_res, y_train_res), batch_size=32, shuffle=True)
    val_loader = DataLoader(ClassificationDataset(X_val, y_val), batch_size=32)

    model = TransFCNClassifier(len(ACTIVE_FEATURES), len(unique_labels)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=100, steps_per_epoch=len(train_loader))

    best_f1 = 0
    for epoch in range(100):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, lbl_a, lbl_b, lam = cutmix_data(imgs, lbls)
            optimizer.zero_grad()
            out = model(imgs)
            loss = lam * criterion(out, lbl_a) + (1 - lam) * criterion(out, lbl_b)
            loss.backward();
            optimizer.step();
            scheduler.step()

        model.eval()
        v_true, v_pred = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                out = model(imgs.to(device))
                v_true.extend(lbls.numpy());
                v_pred.extend(out.argmax(1).cpu().numpy())

        f1 = f1_score(v_true, v_pred, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
        print(f"Epoch {epoch + 1} Val F1: {f1:.4f}")

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth")))
    model.eval()
    t_true, t_pred = [], []
    with torch.no_grad():
        for imgs, lbls in DataLoader(ClassificationDataset(X_test, y_test), batch_size=32):
            out = model(imgs.to(device))
            t_true.extend(lbls.numpy());
            t_pred.extend(out.argmax(1).cpu().numpy())

    print("\nFinal Test Report:")
    print(classification_report(t_true, t_pred, target_names=[str(l) for l in unique_labels]))