import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class NuScenesDataset(data.Dataset):
    """nuScenes 数据集加载器，适配 STEPS 预处理后的结构"""
    def __init__(self, data_path, filenames, height, width, frame_ids, is_train=False, img_ext='.jpg'):
        super(NuScenesDataset, self).__init__()
        self.data_path = data_path
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.is_train = is_train
        self.img_ext = img_ext  # 实际没用到，因为 file_list.txt 已带后缀

        # filenames: 每行一个场景名（如 "scene-0992"）
        # scene_list = [line.strip() for line in filenames]
        scene_list = [line.strip().split()[0] for line in filenames]

        self.samples = []  # 存放 (场景名, 帧索引) 对
        for scene in scene_list:
            scene_dir = os.path.join(self.data_path, 'sequences', scene)
            if not os.path.isdir(scene_dir):
                continue
            # 读取该场景的图片列表
            with open(os.path.join(scene_dir, 'file_list.txt'), 'r') as f:
                frame_files = [line.strip() for line in f.readlines()]
            # 取能构成连续三帧的中间帧
            for i in range(1, len(frame_files) - 1):
                self.samples.append((scene, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene_name, frame_idx = self.samples[index]
        scene_dir = os.path.join(self.data_path, 'sequences', scene_name)

        # 读取该场景的图片名列表
        with open(os.path.join(scene_dir, 'file_list.txt'), 'r') as f:
            frame_files = [line.strip() for line in f.readlines()]

        inputs = {}

        # 加载所有需要的帧（前、后帧等）
        for f_i in self.frame_ids:
            if f_i == "s":  # 跳过立体帧
                continue
            img_name = frame_files[frame_idx + f_i]
            img_path = os.path.join(scene_dir, img_name)
            color = pil_loader(img_path)
            # 缩放到指定尺寸
            color = color.resize((self.width, self.height), resample=Image.LANCZOS)
            color_tensor = torch.from_numpy(np.array(color)).permute(2, 0, 1).float() / 255.0
            inputs[("color", f_i, 0)] = color_tensor
            inputs[("color_aug", f_i, 0)] = color_tensor  # 先复制，稍后增强

        # 读取内参 K
        intrinsic_path = os.path.join(scene_dir, 'intrinsic.npy')
        K = np.load(intrinsic_path).astype(np.float32)  # (3,3)

        # 调整内参以适应缩放后的尺寸（nuScenes 原图为 1600x900）
        orig_h, orig_w = 900, 1600
        scale_x = self.width / orig_w
        scale_y = self.height / orig_h
        K_scaled = K.copy()
        K_scaled[0, :] *= scale_x
        K_scaled[1, :] *= scale_y

        K_tensor = torch.from_numpy(K_scaled).float()
        inputs[("K", 0)] = K_tensor
        inputs[("inv_K", 0)] = torch.inverse(K_tensor)

        # 训练时进行随机左右翻转（数据增强）
        if self.is_train and np.random.rand() > 0.5:
            # 翻转所有图像
            for f_i in self.frame_ids:
                if f_i == "s":
                    continue
                flipped = torch.flip(inputs[("color", f_i, 0)], dims=[2])
                inputs[("color", f_i, 0)] = flipped
                inputs[("color_aug", f_i, 0)] = flipped
            # 调整内参的 cx
            K_tensor[0, 2] = self.width - 1 - K_tensor[0, 2]
            inputs[("K", 0)] = K_tensor
            inputs[("inv_K", 0)] = torch.inverse(K_tensor)

        return inputs