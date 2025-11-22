import logging
import os
import random
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter
from .intensity_attack import IntensityAttacker, generate_part_aware_mask


def part_attention_vit_do_train_with_amp(cfg,
                                         model,
                                         train_loader,
                                         val_loader,
                                         optimizer,
                                         scheduler,
                                         loss_fn,
                                         num_query, local_rank,
                                         patch_centers=None,
                                         pc_criterion=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training')
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    # ============ 初始化强度攻击器和超参数 ============
    attacker = IntensityAttacker(n_points=20, data_range=(-1, 1)).to(device)
    delta = 3.0       # 攻击强度
    lambda1 = 0.5     # 对抗损失权重
    lambda2 = 0.2     # cosine对齐损失权重
    patch_size = 16   # patch大小，与cfg.MODEL.STRIDE_SIZE一致

    total_loss_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    pc_loss_meter = AverageMeter()
    part_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    cos_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512)
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    # 初始化centers
    if cfg.MODEL.PC_LOSS:
        print('initialize the centers')
        model.train()
        for i, informations in enumerate(train_loader):
            with torch.no_grad():
                input = informations['images'].cuda(non_blocking=True)
                vid = informations['targets']
                camid = informations['camid']
                path = informations['img_path']
                _, _, layerwise_feat_list, _, _ = model(input)
                patch_centers.get_soft_label(path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('initialization done')

    best_mAP = 0.0
    best_index = 1
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        part_loss_meter.reset()
        adv_loss_meter.reset()
        cos_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            camid = informations['camid']
            img_path = informations['img_path']
            t_domains = informations['others']['domains']

            img = img.to(device)
            target = vid.to(device)
            target_cam = camid.to(device)
            t_domains = t_domains.to(device)
            attacker.reset_parameters()

            for param in model.parameters():
                param.requires_grad = False
            attacker.rho.requires_grad = True

            B, C, H, W = img.shape
            mask = generate_part_aware_mask(
                batch_size=B,
                img_height=H,
                img_width=W,
                patch_size=patch_size,
                selected_part=None,
                device=device
            )

            img_adv = attacker(img, mask)
            score_adv, _, layerwise_feat_list_adv, _, _ = model(img_adv)

            patch_agent_adv, position_adv = patch_centers.get_soft_label(
                img_path, layerwise_feat_list_adv[-1], vid=vid, camid=camid
            )
            feat_adv = torch.stack(layerwise_feat_list_adv[-1], dim=0)
            feat_adv = feat_adv[:, ::1, :]
            ploss_for_attacker, _ = pc_criterion(feat_adv, patch_agent_adv, position_adv,
                                                 patch_centers, vid=target, camid=target_cam)

            attacker.zero_grad()
            ploss_for_attacker.backward()

            with torch.no_grad():
                grad = attacker.rho.grad
                if grad is not None and grad.norm() > 0:
                    attacker.rho.data = delta * grad / (grad.norm() + 1e-8)
            for param in model.parameters():
                param.requires_grad = True
            attacker.rho.requires_grad = False
            optimizer.zero_grad()

            with amp.autocast(enabled=False):
                score, layerwise_global_feat, layerwise_feat_list, part_scores, part_feats = model(img)
                img_adv_final = attacker(img, mask)
                score_adv, layerwise_global_feat_adv, layerwise_feat_list_adv, part_scores_adv, part_feats_adv = model(img_adv_final)
                patch_agent, position = patch_centers.get_soft_label(img_path, layerwise_feat_list[-1], vid=vid, camid=camid)
                l_ploss = cfg.MODEL.PC_LR

                if cfg.MODEL.PC_LOSS:
                    feat = torch.stack(layerwise_feat_list[-1], dim=0)
                    feat = feat[:, ::1, :]
                    ploss_clean, all_posvid = pc_criterion(feat, patch_agent, position, patch_centers,
                                                           vid=target, camid=target_cam)
                    reid_loss_clean = loss_fn(score, layerwise_global_feat[-1], target,
                                              all_posvid=all_posvid, soft_label=cfg.MODEL.SOFT_LABEL,
                                              soft_weight=cfg.MODEL.SOFT_WEIGHT, soft_lambda=cfg.MODEL.SOFT_LAMBDA)
                else:
                    ploss_clean = torch.tensor([0.]).cuda()
                    reid_loss_clean = loss_fn(score, layerwise_global_feat[-1], target, soft_label=cfg.MODEL.SOFT_LABEL)
                part_loss_clean = torch.tensor(0.0).cuda()
                for p_idx in range(3):
                    part_loss_i = loss_fn(part_scores[p_idx], part_feats[p_idx], target, soft_label=False)
                    part_loss_clean = part_loss_clean + part_loss_i
                patch_agent_adv, position_adv = patch_centers.get_soft_label(img_path, layerwise_feat_list_adv[-1],
                                                                             vid=vid, camid=camid)

                if cfg.MODEL.PC_LOSS:
                    feat_adv = torch.stack(layerwise_feat_list_adv[-1], dim=0)
                    feat_adv = feat_adv[:, ::1, :]
                    # Adversarial CILL clustering loss
                    ploss_adv, all_posvid_adv = pc_criterion(feat_adv, patch_agent_adv, position_adv,
                                                             patch_centers, vid=target, camid=target_cam)
                else:
                    ploss_adv = torch.tensor([0.]).cuda()
                part_loss_adv = torch.tensor(0.0).cuda()
                for p_idx in range(3):
                    part_loss_i = loss_fn(part_scores_adv[p_idx], part_feats_adv[p_idx], target, soft_label=False)
                    part_loss_adv = part_loss_adv + part_loss_i
                cos_loss = torch.tensor(0.0).cuda()
                for p_idx in range(3):
                    cos_sim = F.cosine_similarity(part_feats[p_idx], part_feats_adv[p_idx], dim=1)
                    cos_loss = cos_loss + (1 - cos_sim).mean()
                cos_loss = cos_loss / 3.0
                total_loss = (
                    reid_loss_clean +
                    l_ploss * ploss_clean +
                    part_loss_clean +
                    lambda1 * (l_ploss * ploss_adv + part_loss_adv) +
                    lambda2 * cos_loss
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            total_loss_meter.update(total_loss.item(), img.shape[0])
            reid_loss_meter.update(reid_loss_clean.item(), img.shape[0])
            acc_meter.update(acc, 1)
            pc_loss_meter.update(ploss_clean.item(), img.shape[0])
            part_loss_meter.update(part_loss_clean.item(), img.shape[0])
            adv_loss_meter.update((ploss_adv + part_loss_adv).item(), img.shape[0])
            cos_loss_meter.update(cos_loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iter[{}/{}] total:{:.3f} reid:{:.3f} pc:{:.3f} part:{:.3f} adv:{:.3f} cos:{:.3f} Acc:{:.3f} Lr:{:.2e}"
                    .format(epoch, n_iter + 1, len(train_loader), total_loss_meter.avg,
                            reid_loss_meter.avg, pc_loss_meter.avg, part_loss_meter.avg,
                            adv_loss_meter.avg, cos_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/pc_loss', pc_loss_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/part_loss', part_loss_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/adv_loss', adv_loss_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))
                tbWriter.add_scalar('train/cos_loss', cos_loss_meter.avg, n_iter + 1 + (epoch - 1) * len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                tbWriter.add_scalar('val/mAP', mAP, epoch)

        if epoch % checkpoint_period == 0:
            if best_mAP < mAP:
                best_mAP = mAP
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(best_index))
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print('load weights from {}_{}.pth'.format(cfg.MODEL.NAME, best_index))
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)

    del_list = os.listdir(log_path)
    for fname in del_list:
        if '.pth' in fname:
            os.remove(os.path.join(log_path, fname))
            print('removing {}. '.format(os.path.join(log_path, fname)))
    print('saving final checkpoint.\nDo not interrupt the program!!!')
    torch.save(eval_model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
    print('done!')


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    t0 = time.time()
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP