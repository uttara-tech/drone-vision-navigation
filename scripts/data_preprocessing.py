import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


# ========================================================
# 1. DATA PREPROCESSING: UZH-FPV Multimodal (Img+IMU)
# ========================================================

class ViTDataset(Dataset):
    

    # UZH-FPV MultiModal Preprocessing Pipeline
    # # Current image size: (640, 480), Target image size: (256,256)
    # - 256x256 → 16x16 patches (256 tokens)
    # - IMU: 16-step windows [B, 16, 6]

    def __init__(self,dataset_csv,imu_csv,image_dir,transform=None):
        self.data_df = pd.read_csv(dataset_csv)
        self.imu_df = pd.read_csv(imu_csv)
        self.image_dir = image_dir
        self.transform = transform 

    def __len__(self):
        return len(self.data_df)
    
    preprocess = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(256,256)),
        ])
    
        
    def __getitem__(self, index):
        img_path = str(self.data_df['img_path'].iloc[index])
        image = Image.open(img_path).convert('L')

        img_ts = str(self.data_df['timestamp'].iloc[index]).split('.')[0]
        img_ts = float(img_ts) / 1e9                                                # Convert to seconds

        img_start_ts = self.data_df['imu_start_ts'].iloc[index]
        img_end_ts = self.data_df['imu_end_ts'].iloc[index]

        mask = (self.imu_df['timestamp'] >= img_start_ts) & (self.imu_df['timestamp'] <= img_end_ts)

        imu_window = self.imu_df[mask]

        s = len(imu_window)

        if len(imu_window) > 0:                                                     # Selecting all IMU rows that fall within this window
            imu_sample = imu_window.iloc[:, 1:s].mean().values.astype('float32')    # Taking the average (mean) of Acc and Gyro over the window
        else:                                                                       # Fallback 
            mid_ts = (img_start_ts + img_end_ts) / 2
            closest_idx = (self.imu_df.iloc[:, index] - mid_ts).abs().idxmin()
            imu_sample = self.imu_df.iloc[closest_idx, index:s].values.astype('float32')

        target_pose = self.data_df.loc[index, ['pos_x', 'pos_y', 'pos_z']].values.astype('float32')     #target position (x,y,z translation) in 3D space corresponding to current index
            
        img_tensor = self.preprocess(image)

        #Normalizing
        img_mean = img_tensor.mean()
        img_std = img_tensor.std()

        img_tensor = v2.functional.normalize(img_tensor, mean=[img_mean], std=[img_std])

        return img_tensor, torch.tensor(imu_sample), torch.tensor(target_pose)