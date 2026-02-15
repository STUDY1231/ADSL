from torch.utils.data import Dataset
from .data_utils import read_image
class MemoryDataset(Dataset):
    """Dataset for initializing memory bank with image features"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        # 处理 PID 的重新映射
        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        # 取出每张图片的路径、ID、摄像头ID等信息
        if len(self.img_items[index]) > 3:
            img_path, pid, camid, did = self.img_items[index]
        else:
            img_path, pid, camid = self.img_items[index]
            did = ''  # 无额外信息时填充空字符串

        # 读取图片并应用预处理
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # 如果需要重新标注 PID，则进行映射
        if self.relabel:
            pid = self.pid_dict[pid]

        # 返回内存库需要的变量：特征和 PID
        return {
            "images": img,  # 特征图像
            "targets": pid,  # 行人 ID（PID）
            "img_path": img_path  # 返回图片路径
        }



