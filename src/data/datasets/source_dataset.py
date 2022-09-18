"""Dataset for loading source data"""
import torch
import torch.utils.data
from PIL import Image
import os
from ..transforms import get_transforms
class SourceDataset(torch.utils.data.Dataset):  # load from offline samples
    def __init__(self,cfg,split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for source dataset".format(
            split)
        self.cfg = cfg
        self.data_path = '/remote-home/share/VPT/samples/imagen/imagen'
        self.img_list = os.listdir(self.data_path)
        self.transform = get_transforms('train', cfg.DATA.CROPSIZE)

    def __getitem__(self,idx):
        img_path = os.path.join(self.data_path,self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)

## to do: online sampling of ImageNet
