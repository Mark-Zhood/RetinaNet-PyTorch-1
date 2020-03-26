import torch.utils.data
import numpy as np
from PIL import Image
from torchvision import transforms as tvtsf
import torch.nn.functional as F
from utils.box_tools import wh2xy, assign_anchors, xy2wh, box2loc,create_anchors
from config import cfg
from torch.utils.data import Dataset
import glob


class ListDataset(Dataset):
    def __init__(self,path, is_train=True):
        self.class_names = cfg.class_name
        self.is_train = is_train
        with open(path) as f:
            self.img_paths = f.readlines()
        self.label_paths = [path.replace('JPGImages', 'labels').replace('.jpg', '.txt') for path in self.img_paths]
        self.anchors_xywh = create_anchors()
        self.anchors_xyxy = wh2xy(self.anchors_xywh)

    def __getitem__(self, index):
        img_path = self.img_paths[index].rstrip()
        label_path = self.label_paths[index].rstrip()
        label_data = (np.loadtxt(label_path).reshape(-1, 5))
        target_label = label_data[:, 0].astype(np.int64)
        target_boxes = label_data[:, 1:].astype(np.float32)
        img = tvtsf.ToTensor()(Image.open(img_path).convert("RGB"))
        c, h, w = img.shape
        # 将坐标修改为相对形式的坐标
        target_boxes[:, 0::2] /= w
        target_boxes[:, 1::2] /= h
        img = F.interpolate(img.unsqueeze(0), size=(cfg.height, cfg.height), mode="nearest").squeeze(0)
        img = tvtsf.Normalize(mean=[0.13428119, 0.13828206, 0.20883734], std=[0.19393602, 0.16601577, 0.16852522])(img)
        target_boxes = torch.from_numpy(target_boxes)
        target_labels = torch.from_numpy(target_label)
        # 为每个anchor赋予拟合box及label,以及后面的修正系数
        target_boxes, target_labels = assign_anchors(target_boxes, target_labels, self.anchors_xyxy)
        target_boxes = xy2wh(target_boxes)
        target_locs = box2loc(target_boxes, self.anchors_xywh)
        return img, target_locs, target_labels, img_path

    def __len__(self):
        return len(self.label_paths)

    def _get_annotation(self, img_name):
        w, h = Image.open(img_name).size
        label_path = img_name.replace('JPGImages', 'labels').replace('.jpg', '.txt')
        label_data = (np.loadtxt(label_path).reshape(-1, 5))
        labels = label_data[:, 0].astype(np.int64)
        boxes = label_data[:, 1:].astype(np.float32)
        return boxes, labels, w, h


# 为测试图片准备
class ImageFolder(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob("%s/*.*" % folder_path)

    def __getitem__(self, index):
        img_path = self.files[index]
        # 这里使用convert是防止使用png图片或其他格式时会有多个通道而引起的报错,
        img = tvtsf.ToTensor()(Image.open(img_path).convert("RGB"))
        img = F.interpolate(img.unsqueeze(0), size=(600, 600), mode="nearest").squeeze(0)
        img = tvtsf.Normalize(mean=[0.13428119, 0.13828206, 0.20883734], std=[0.19393602, 0.16601577, 0.16852522])(img)
        return img, img_path

    def __len__(self):
        return len(self.files)