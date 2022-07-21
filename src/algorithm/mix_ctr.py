import torch
import sys
import wandb
import numpy as np
from .mixup import mixup_data, mixup_criterion
from .utils import equal_batch_size, cosine_similarity


#two contrastive loss
#first contrastive loss in all data: postive pair: images from same domains, and images from different domains
#second contrastive loss: mixup
def mix_ctr_learning(args, model, optimizers, schedulers, trainset, valset,
                     testset):
    model.train()
    c_loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(args.mix_ctr_epochs):
        skip_steps = 0
        for i, batch in enumerate(trainset):
            x = batch['img'].to(args.device)
            y = batch['label'].to(args.device)
            d = batch['domain'].to(args.device)

            total_loss = torch.tensor(0.0).to(args.device)
            same_ctr_loss = torch.tensor(0.0).to(args.device)
            diff_ctr_loss = torch.tensor(0.0).to(args.device)
            same_hinge_loss = torch.tensor(0.0).to(args.device)
            diff_hinge_loss = torch.tensor(0.0).to(args.device)

            feat = model.conv_features(x)
            feat = model.dense_features(feat)
            yhat = model.classifier(feat)
            c_loss = c_loss_fn(yhat, y)
            total_loss += c_loss * args.mix_ctr_cls_loss_w

            total_mix_loss, total_hinge_loss = mix_ctr(args, feat, y, d, model,
                                                       c_loss_fn)
            total_loss += total_mix_loss * args.mix_ctr_mix_loss_w
            total_loss += total_hinge_loss * args.mix_hinge_loss_w


def getMixCtrLoss(args, fir_dom_feat, fir_dom_cls, fir_dom_img, sec_dom_feat,
                  sec_dom_cls, sec_dom_img, thi_dom_feat, thi_dom_cls,
                  thi_dom_img, model, c_loss_fn):
    #mixup first domain with sec domain
    fir_dom_feat, fir_dom_cls, fir_dom_img, sec_dom_feat, sec_dom_cls, sec_dom_img = equal_batch_size(
        fir_dom_feat, fir_dom_cls, fir_dom_img, sec_dom_feat, sec_dom_cls,
        sec_dom_img)
    fir_dom_feat, fir_dom_cls, fir_dom_img, thi_dom_feat, thi_dom_cls, thi_dom_img = equal_batch_size(
        fir_dom_feat, fir_dom_cls, fir_dom_img, thi_dom_feat, thi_dom_cls,
        thi_dom_img)
    assert fir_dom_feat.shape[0] == thi_dom_feat.shape[
        0] and fir_dom_feat.shape[0] == sec_dom_feat.shape[0]
    mix_firSec_dom_img, lam = mixup_data(fir_dom_img, sec_dom_img)
    # calculate first domain and second domain for mixup class loss
    mix_firSec_dom_feat = model.dense_features(
        model.conv_features(mix_firSec_dom_img))
    mix_firSec_dom_yhat = model.classifier(mix_firSec_dom_feat)
    mix_firSec_dom_loss = mixup_criterion(c_loss_fn, mix_firSec_dom_yhat,
                                          fir_dom_cls, sec_dom_cls, lam)
    mix_firSec_dom_loss = mix_firSec_dom_loss
    # calculate first domain and firSecMix domain distance (positive)
    firSec_pos_dist = cosine_similarity(fir_dom_feat, mix_firSec_dom_feat)
    firSec_pos_dist = firSec_pos_dist / args.tau
    # calculate first domain and third domain distance (negative)
    firThi_neg_dist = cosine_similarity(fir_dom_feat, thi_dom_feat)
    firThi_neg_dist = firThi_neg_dist / args.tau

    hinge_loss = -1 * torch.sum(firSec_pos_dist - torch.log(
        torch.exp(firSec_pos_dist) + torch.exp(firThi_neg_dist)))

    return mix_firSec_dom_loss, hinge_loss


def mix_ctr(args, img, feat, y, d, model, c_loss_fn):
    total_mix_loss = 0
    total_hinge_loss = 0
    for y_d in range(args.num_src_domains):
        fir_dom_indices = d[:] == y_d
        fir_dom_feat = feat[fir_dom_indices]
        fir_dom_cls = y[fir_dom_indices]
        fir_dom_img = img[fir_dom_indices]
        sec_dom_indices = d[:] == ((y_d + 1) % args.num_src_domains)
        sec_dom_feat = feat[sec_dom_indices]
        sec_dom_cls = y[sec_dom_indices]
        sec_dom_img = img[sec_dom_indices]
        thi_dom_indices = d[:] == ((y_d + 2) % args.num_src_domains)
        thi_dom_feat = feat[thi_dom_indices]
        thi_dom_cls = y[thi_dom_indices]
        thi_dom_img = img[thi_dom_indices]
        if fir_dom_feat.shape[0] == 0 or sec_dom_feat.shape[
                0] == 0 or thi_dom_feat.shape[0] == 0:
            continue
        mix_firSec_dom_loss, firSec_dom_hinge_loss = getMixCtrLoss(
            args, fir_dom_feat, fir_dom_cls, fir_dom_img, sec_dom_feat,
            sec_dom_cls, sec_dom_img, thi_dom_feat, thi_dom_cls, thi_dom_img,
            model, c_loss_fn)
        mix_firThi_dom_loss, firThi_dom_hinge_loss = getMixCtrLoss(
            args, fir_dom_feat, fir_dom_cls, fir_dom_img, thi_dom_feat,
            thi_dom_cls, thi_dom_img, sec_dom_feat, sec_dom_cls, sec_dom_img,
            model, c_loss_fn)
        total_mix_loss += (mix_firSec_dom_loss + mix_firThi_dom_loss) / 2
        total_hinge_loss += (firSec_dom_hinge_loss + firThi_dom_hinge_loss) / 2
    total_mix_loss = total_mix_loss / args.num_src_domains
    total_hinge_loss = total_hinge_loss / args.num_src_domains
    return total_mix_loss, total_hinge_loss