from collections import OrderedDict
import time
import torch
from utils.to_torch import to_torch
from utils.meter import AverageMeter
def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
       feat = model(inputs)
    return feat.cpu()


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()

    with torch.no_grad():
        time.sleep(2)
        print("Feature extraction ...")
        for i, batch in enumerate(data_loader):
            data_time.update(time.time() - end)

            imgs = batch["images"].cuda()
            pids = batch["targets"]
            # 如果你的 MemoryDataset 还返回 'img_path' 作为 key，使用它作为 fname
            fnames = batch["img_path"]  # 或者 batch["img_path"]，看你数据结构

            outputs = extract_cnn_feature(model, imgs)
            # print(f"model outputs len: {len(outputs)}")

            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels
