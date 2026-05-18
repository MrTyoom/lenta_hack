import glob
import os
import shutil
from omegaconf import OmegaConf

def top1_crops(cfg: OmegaConf) -> None:
    os.makedirs(cfg.output_folder, exist_ok=True)
    
    all_imgs = sorted(glob.glob(os.path.join(cfg.input_folder, "**/top1*.jpg"), recursive=True))

    for ind, img_path in enumerate(all_imgs):
        shutil.copy(img_path, cfg.output_folder)