import os
import torch
import sys
import collections.abc as container_abcs

# from torch._six import container_abcs, string_classes, int_classes
int_classes = int
string_classes = str
from torch.utils.data import DataLoader
from utils import comm
import random
import torchvision.transforms as T
from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms
from data.MemoryDataset import MemoryDataset
_root = os.getenv("REID_DATASETS", "../../data")


def build_reid_train_loader(cfg):
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('*'*100)
        print('Hmm, Big Debugger is watching me')
        print('*'*100)
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms(cfg, is_train=True, is_fake=False)
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    _root = cfg.DATASETS.ROOT_DIR
    for d in cfg.DATASETS.TRAIN:
        if d == 'CUHK03_NP':
            dataset = DATASET_REGISTRY.get('CUHK03')(root=_root, cuhk03_labeled=False)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if len(dataset.train[0]) < 4:
            for i, x in enumerate(dataset.train):
                add_info = {}  # dictionary

                if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                    add_info['domains'] = dataset.train[i][2]
                    camera_all.append(dataset.train[i][2])
                else:
                    add_info['domains'] = int(domain_idx)
                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(add_info)
                dataset.train[i] = tuple(dataset.train[i])
        domain_idx += 1
        train_items.extend(dataset.train)

    #################### 插入：检查每个ID的图片数 ≥4 ####################
    from collections import defaultdict
    id_count = defaultdict(int)
    for img_item in train_items:
        pid = img_item[1]  # 假设pid在索引1的位置
        id_count[pid] += 1
    min_count = min(id_count.values())
    assert min_count >=2, f"存在ID图片数不足: {min_count}"
    ##################################################################
    train_set = CommDataset(train_items, train_transforms, relabel=True)

    train_loader = make_sampler(
        train_set=train_set,
        num_batch=cfg.SOLVER.IMS_PER_BATCH,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=cfg.DATALOADER.DROP_LAST,
        flag1=cfg.DATALOADER.NAIVE_WAY,
        flag2=cfg.DATALOADER.DELETE_REM,
        cfg = cfg)
    #################### 插入：验证第一个batch的ID分布 ####################
    # 检查第一个batch是否符合num_instance=4的要求
    for first_batch in train_loader:
        batch_ids = first_batch['targets']  # 假设targets是行人ID
        unique_ids, counts = torch.unique(batch_ids, return_counts=True)

        print(f"\n验证batch的ID分布:")
        print(f"数据集ID总数: {len(id_count)}")
        print(f"最小ID图片数: {min_count}")
        print(f"当前batch的ID数量: {len(unique_ids)}")
        print(f"当前batch的ID分布示例: {unique_ids[:5]}...")  # 打印前5个ID
        print(f"每个ID的图片数: {counts.tolist()}")  # 应全为4
        assert torch.all(counts == cfg.DATALOADER.NUM_INSTANCE), \
            f"采样错误: ID图片数应为{cfg.DATALOADER.NUM_INSTANCE}, 实际为{counts.tolist()}"
        break  # 仅验证第一个batch
    ###################################################################

    return train_loader

def get_train_items(cfg):
    train_items = []
    domain_idx = 0
    _root = cfg.DATASETS.ROOT_DIR

    for d in cfg.DATASETS.TRAIN:
        if d == 'CUHK03_NP':
            dataset = DATASET_REGISTRY.get('CUHK03')(root=_root, cuhk03_labeled=False)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)

        if comm.is_main_process():
            dataset.show_train()

        if len(dataset.train[0]) < 4:
            for i in range(len(dataset.train)):
                add_info = {}

                if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                    add_info['domains'] = dataset.train[i][2]
                else:
                    add_info['domains'] = int(domain_idx)

                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(add_info)
                dataset.train[i] = tuple(dataset.train[i])

        domain_idx += 1
        train_items.extend(dataset.train)

    return train_items
################  get_memory_loader   ###############
def get_memory_loader(cfg,train_items):
    """
      获取内存库加载器，用于提取特征并初始化内存库。
      :param cfg: 配置参数
      :param dataset: 数据集对象
      :return: memory_loader
      """
    # 简单的图像预处理：仅进行resize和normalize操作
    normalizer = T.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])

    # 只进行基本的图像预处理，不进行增广操作
    test_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    # 创建内存库数据加载器
    memory_loader = DataLoader(
        MemoryDataset(train_items,  transform=test_transformer),
        batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS, shuffle=False, pin_memory=True
    )
    return memory_loader


def build_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True, shuffle=False, only_gallery=False, only_query=False, eval_time=False):
    test_transforms = build_transforms(cfg, is_train=False)
    _root = cfg.DATASETS.ROOT_DIR
    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    if flag_test:
        if only_gallery:
            test_items = dataset.gallery
        elif only_query:
            test_set = CommDataset([random.choice(dataset.query)], test_transforms, relabel=False)
            return test_set
        else:
            test_items = dataset.query + dataset.gallery
        if shuffle: # only for visualization
            random.shuffle(test_items)
    else:
        test_items = dataset.train

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, list):
        out_g = []
        out_pt1 = []
        out_pt2 = []
        out_pt3 = []
        # out = torch.stack(elem, dim=0)
        for i, tensor_list in enumerate(batched_inputs):
            out_g.append(tensor_list[0])
            out_pt1.append(tensor_list[1])
            out_pt2.append(tensor_list[2])
            out_pt3.append(tensor_list[3])
        out = torch.stack(out_g, dim=0)
        out_pt1 = torch.stack(out_pt1, dim=0)
        out_pt2 = torch.stack(out_pt2, dim=0)
        out_pt3 = torch.stack(out_pt3, dim=0)
        return out, out_pt1, out_pt2, out_pt3


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, cfg=None):

    if flag1:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader