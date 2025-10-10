import torch
from torch.utils.data import DataLoader
from coloradar_heatmap import ColoradarHeatmapDataset
from models.cnn_setup import RadarCNN
import numpy as np
import matplotlib.pyplot as plt

#Dataset
DATA_ROOT = r"D:\ColoRadar"
TRAIN_SEQS = [r"12_21_2020_ec_hallways_run4\12_21_2020_ec_hallways_run4\single_chip\heatmaps",
              r"12_21_2020_ec_hallways_run3\12_21_2020_ec_hallways_run3\single_chip\heatmaps",
              r"12_21_2020_ec_hallways_run2\12_21_2020_ec_hallways_run2\single_chip\heatmaps",
              r"12_21_2020_ec_hallways_run1\12_21_2020_ec_hallways_run1\single_chip\heatmaps",
              r"12_21_2020_ec_hallways_run0\12_21_2020_ec_hallways_run0\single_chip\heatmaps",
              r"12_21_2020_arpg_lab_run0\12_21_2020_arpg_lab_run0\single_chip\heatmaps",
              r"12_21_2020_arpg_lab_run1\12_21_2020_arpg_lab_run1\single_chip\heatmaps",
              r"12_21_2020_arpg_lab_run2\12_21_2020_arpg_lab_run2\single_chip\heatmaps",
              r"12_21_2020_arpg_lab_run3\12_21_2020_arpg_lab_run3\single_chip\heatmaps"]

ds = ColoradarHeatmapDataset(
    DATA_ROOT,
    sequences=TRAIN_SEQS,
    pattern="data/*.bin",
    heatmap_shape=(2, 256, 256),
    normalization="zscore",
)

print("Total samples:", len(ds))
print("First few paths:")
for i in range(min(5, len(ds))):
    print("  ", ds.samples[i][0])


loader = DataLoader(ds, batch_size=2, shuffle=False)

batch = next(iter(loader))
images = batch["image"]
print(f"Loaded batch: {tuple(images.shape)}")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RadarCNN(in_channels=images.shape[1], num_classes=2)
model.to(device)
model.eval()

# Forward pass
with torch.no_grad():
    logits = model(images.to(device))
print(f"Output logits: {tuple(logits.shape)}")

probs = torch.softmax(logits, dim=1)
print("Probs (first batch):", probs.cpu().numpy())

# Visualize heatmap wth
img0 = images[0, 0].cpu().numpy()
plt.figure(figsize=(6, 4))
plt.imshow(np.log1p(img0), cmap='hot')  # log scale apparently helps reveal detail
plt.title("Radar heatmap (ch0, log-scaled)")
plt.tight_layout()
plt.show()

if images.shape[1] > 1:
    img1 = images[0, 1].cpu().numpy()  # channel 1  Doppler
    plt.figure(figsize=(6, 4))
    plt.imshow(img1, cmap='coolwarm')
    plt.title("Radar heatmap (ch1)")
    plt.tight_layout()
    plt.show()
