"""
FingerFlex training and testing on the added Miller subjects
(scripted from Lightning_BCI-autoencoder.ipynb).

Faithfully reproduces the notebook's AutoEncoder1D training (30 epochs,
lr=8.42e-5, Adam wd=1e-6), with SUBJECT replaced by a Miller subject, the
wandb/plotly visualization removed, and the best checkpoint auto-loaded.
SEED (second CLI argument) enables the three-seed protocol.

Usage (from the fingerflex/ root):
    python code/train_miller.py jc 0
"""
import sys
import pathlib
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "ht"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_EPOCHS = 30
DOWNSAMPLE_FS = 100
WAVELET_NUM = 40
SAMPLE_LEN = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = f"data/processed/{SUBJECT}"
print(f"SUBJECT={SUBJECT}  DEVICE={DEVICE}  DATA_DIR={DATA_DIR}", flush=True)


# ============================================================
# Data module
# ============================================================
class EcogFingerflexDataset(Dataset):
    def __init__(self, ecog_path, flex_path, sample_len, train=False):
        self.ecog_data = np.load(ecog_path).astype("float32")
        self.fingerflex_data = np.load(flex_path).astype("float32")
        self.duration = self.ecog_data.shape[2]
        self.sample_len = sample_len
        self.stride = 1
        self.ds_len = (self.duration - self.sample_len) // self.stride
        self.train = train

    def __len__(self):
        return self.ds_len

    def __getitem__(self, index):
        s = index * self.stride
        e = s + self.sample_len
        return self.ecog_data[..., s:e], self.fingerflex_data[..., s:e]


class EcogFingerflexDatamodule(pl.LightningDataModule):
    def __init__(self, sample_len, data_dir, batch_size=128, add_name=""):
        super().__init__()
        self.data_dir = data_dir
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.add_name = add_name

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            self.train = EcogFingerflexDataset(
                f"{self.data_dir}/train/ecog_data{self.add_name}.npy",
                f"{self.data_dir}/train/fingerflex_data{self.add_name}.npy",
                self.sample_len, train=True)
            self.val = EcogFingerflexDataset(
                f"{self.data_dir}/val/ecog_data{self.add_name}.npy",
                f"{self.data_dir}/val/fingerflex_data{self.add_name}.npy",
                self.sample_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)


# ============================================================
# Model
# ============================================================
def correlation_metric(x, y):
    return torch.mean(nn.CosineSimilarity(dim=-1, eps=1e-08)(x, y))


def corr_metric(x, y):
    return float(np.corrcoef(x, y)[0, 1])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, p_conv_drop=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                bias=False, padding="same")
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self.stride = stride

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.drop(x)
        x = self.downsample(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super().__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode="linear", align_corners=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    def __init__(self, n_electrodes=30, n_freqs=16, n_channels_out=21,
                 channels=[8, 16, 32, 32], kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], dilation=[1, 1, 1]):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs * n_electrodes
        self.n_channels_out = n_channels_out
        self.model_depth = len(channels) - 1
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        self.downsample_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_sizes[i],
                      stride=strides[i], dilation=dilation[i])
            for i in range(self.model_depth)])
        channels = [ch for ch in channels[:-1]] + channels[-1:]
        self.upsample_blocks = nn.ModuleList([
            UpConvBlock(scale=strides[i],
                        in_channels=channels[i + 1] if i == self.model_depth - 1 else channels[i + 1] * 2,
                        out_channels=channels[i],
                        kernel_size=kernel_sizes[i])
            for i in range(self.model_depth - 1, -1, -1)])
        self.conv1x1_one = nn.Conv1d(channels[0] * 2, self.n_channels_out, kernel_size=1, padding="same")

    def forward(self, x):
        batch, elec, n_freq, time = x.shape
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        skip = []
        for i in range(self.model_depth):
            skip.append(x)
            x = self.downsample_blocks[i](x)
        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat((x, skip[-1 - i]), dim=1)
        return self.conv1x1_one(x)


class BaseEcogFingerflexModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 8.42e-5

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cosine_dst_train", corr, on_step=False, on_epoch=True, prog_bar=True)
        return 0.5 * loss + 0.5 * (1.0 - corr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cosine_dst_val", corr, on_step=False, on_epoch=True, prog_bar=True)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)


# ============================================================
# Callbacks
# ============================================================
class ValidationCallback(Callback):
    def __init__(self, val_x, val_y, fg_num):
        super().__init__()
        self.val_x = val_x.T
        self.val_y = val_y.T
        self.fg_num = fg_num

    def on_validation_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            SIZE = 64
            bound = self.val_x.shape[0] // SIZE * SIZE
            X_test = self.val_x[:bound]
            y_test = self.val_y[:bound]
            x_batch = torch.from_numpy(X_test).float().to(pl_module.device).T
            x_batch = torch.unsqueeze(x_batch, 0)
            y_hat = pl_module.model(x_batch)[0].cpu().detach().numpy()
            STRIDE = 1
            y_prediction = y_hat.T[::int(STRIDE * (DOWNSAMPLE_FS / 100)), :]
            y_prediction = scipy.ndimage.gaussian_filter1d(y_prediction.T, sigma=6).T
            y_test = y_test[::int(STRIDE * (DOWNSAMPLE_FS / 100)), :]
            corrs = [corr_metric(y_prediction[:, roi], y_test[:, roi]) for roi in range(self.fg_num)]
            corr_mean = float(np.mean(corrs))
            pl_module.log("corr_mean_val", corr_mean, on_step=False, on_epoch=True, prog_bar=True)


class TestCallback:
    def __init__(self, val_x, val_y, fg_num, subject):
        self.val_x = val_x.T
        self.val_y = val_y.T
        self.fg_num = fg_num
        self.subject = subject

    def test(self, pl_module):
        pl_module.eval()
        with torch.no_grad():
            SIZE = 64
            bound = self.val_x.shape[0] // SIZE * SIZE
            X_test = self.val_x[:bound]
            y_test = self.val_y[:bound]
            x_batch = torch.from_numpy(X_test).float().to(pl_module.device).T
            x_batch = torch.unsqueeze(x_batch, 0)
            y_hat = pl_module.model(x_batch)[0].cpu().detach().numpy()
            STRIDE = 1
            y_prediction = y_hat.T[::int(STRIDE * (DOWNSAMPLE_FS / 100)), :]
            y_prediction = scipy.ndimage.gaussian_filter1d(y_prediction.T, sigma=1).T
            y_test = y_test[::int(STRIDE * (DOWNSAMPLE_FS / 100)), :]
            pathlib.Path("res_npy").mkdir(parents=True, exist_ok=True)
            np.save(f"res_npy/prediction_{self.subject}.npy", y_prediction)
            np.save(f"res_npy/true_{self.subject}.npy", y_test)
            corrs = [corr_metric(y_prediction[:, roi], y_test[:, roi]) for roi in range(self.fg_num)]
            corr_mean = float(np.mean(corrs))
            print(f"[{self.subject}] test per-finger r: {[round(c, 3) for c in corrs]}  avg_r={corr_mean:.4f}", flush=True)


# ============================================================
# Main flow
# ============================================================
def main():
    pl.seed_everything(SEED, workers=True)
    print(f"SEED={SEED}", flush=True)
    probe = np.load(f"{DATA_DIR}/train/ecog_data.npy")
    CHANNELS_NUM = probe.shape[0]
    del probe
    print(f"Detected {CHANNELS_NUM} channels for {SUBJECT}", flush=True)

    hp = dict(channels=[32, 32, 64, 64, 128, 128],
              kernel_sizes=[7, 7, 5, 5, 5],
              strides=[2, 2, 2, 2, 2],
              dilation=[1, 1, 1, 1, 1],
              n_electrodes=CHANNELS_NUM,
              n_freqs=WAVELET_NUM,
              n_channels_out=5)
    model = AutoEncoder1D(**hp).to(DEVICE)
    wrapper = BaseEcogFingerflexModel(model)
    dm = EcogFingerflexDatamodule(sample_len=SAMPLE_LEN, data_dir=DATA_DIR, add_name="")

    ecog_val = np.load(f"{DATA_DIR}/val/ecog_data.npy")
    flex_val = np.load(f"{DATA_DIR}/val/fingerflex_data.npy")

    ckpt_cb = ModelCheckpoint(save_top_k=1, monitor="corr_mean_val", mode="max",
                              dirpath="checkpoints",
                              filename=f"{SUBJECT}-seed{SEED}-{{epoch:02d}}-{{corr_mean_val:.4f}}")
    val_cb = ValidationCallback(ecog_val, flex_val, 5)

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      devices=1, max_epochs=MAX_EPOCHS, logger=True,
                      callbacks=[val_cb, ckpt_cb], enable_progress_bar=False)
    trainer.fit(wrapper, dm)

    best_path = ckpt_cb.best_model_path
    print(f"best checkpoint: {best_path}", flush=True)
    trained = BaseEcogFingerflexModel.load_from_checkpoint(best_path, model=AutoEncoder1D(**hp)).to(DEVICE)
    TestCallback(ecog_val, flex_val, 5, f"{SUBJECT}_seed{SEED}").test(trained)


if __name__ == "__main__":
    main()
