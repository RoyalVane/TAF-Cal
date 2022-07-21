import torch
import sys
import wandb
import numpy as np


def train_ctr_phase(args,
                    model,
                    optimizers,
                    trainset,
                    schedulers,
                    domain_discriminator=None,
                    d_loss_fn=None):
    model.train()
    if domain_discriminator is not None:
        domain_discriminator.train()
    for epoch in range(args.ctr_pretrain_epochs):
        penalty_same_ctr = 0
        penalty_diff_ctr = 0
        penalty_same_hinge = 0
        penalty_diff_hinge = 0
        skip_pos_neg_feat = 0
        skip_pos_feat_c = 0
        if domain_discriminator is not None:
            p = epoch / 25
            supression = (2.0 / (1. + np.exp(-args.supression_decay * p)) - 1)
            print(f'supression={supression:.2f}')
            beta = 1 * args.domain_adversary_weight
            beta *= supression
            print(f'supression={beta:.2f}')
            domain_discriminator.set_beta(beta)
        for i, batch in enumerate(trainset):
            x = batch['img'].to(args.device)
            y = batch['label'].to(args.device)
            d = batch['domain'].to(args.device)
            a = batch['augmented'].to(args.device)

            loss_e = torch.tensor(0.0).to(args.device)

            same_ctr_loss = torch.tensor(0.0).to(args.device)
            diff_ctr_loss = torch.tensor(0.0).to(args.device)
            same_hinge_loss = torch.tensor(0.0).to(args.device)
            diff_hinge_loss = torch.tensor(0.0).to(args.device)

            feat = model.conv_features(x)
            feat = model.dense_features(feat)
            if domain_discriminator is not None:
                dhat = domain_discriminator(feat)
                d_loss = d_loss_fn(dhat, d)
            #contrastive loss
            same_neg_counter = 1
            diff_neg_counter = 1

            for y_d in range(args.num_src_domains):
                pos_indices = d[:] == y_d
                neg_indices = d[:] != y_d
                pos_feat = feat[pos_indices]
                pos_y = y[pos_indices]
                neg_feat = feat[neg_indices]
                neg_y = y[neg_indices]
                if pos_feat.shape[0] == 0 or neg_feat.shape[0] == 0:
                    skip_pos_neg_feat += 1
                    continue
                for y_ci in range(args.num_classes):
                    pos_indices_ci = pos_y[:] == y_ci
                    pos_feat_ci = pos_feat[pos_indices_ci]
                    # neg_indicies_ci = neg_y[:] == y_ci
                    # neg_feat_ci = neg_feat[neg_indicies_ci]
                    # if pos_feat_ci.shape[0] == 0 or neg_feat_ci.shape[0] == 0:
                    #     skip_pos_neg_feat += 1
                    #     continue

                    neg_dist = embedding_dist(
                        pos_feat_ci, neg_feat, args.tau,
                        xent=True)  #TODO: change neg_feat to neg_feat_ci
                    if torch.sum(torch.isnan(neg_dist)):
                        print('Neg Dist Nan')
                        sys.exit()
                    for y_cj in range(args.num_classes):
                        if y_ci != y_cj:
                            pos_indices_cj = pos_y[:] == y_cj
                            pos_feat_cj = pos_feat[pos_indices_cj]
                            if pos_feat_ci.shape[0] == 0 or pos_feat_cj.shape[
                                    0] == 0:
                                skip_pos_feat_c += 1
                                continue
                            pos_dist = 1 - embedding_dist(
                                pos_feat_ci, pos_feat_cj, metric='cos')
                            pos_dist = pos_dist / args.tau
                            # pos_dist = torch.mean(pos_dist, dim=1)
                            if torch.sum(torch.isnan(pos_dist)):
                                print('Pos Dist Nan')
                                sys.exit()
                            if torch.sum(
                                    torch.isnan(
                                        torch.log(
                                            torch.mean(torch.exp(pos_dist),
                                                       dim=1) + neg_dist))):
                                print('Xent Nan')
                                sys.exit()
                            diff_hinge_loss += -1 * torch.sum(
                                torch.mean(pos_dist, dim=1) - torch.log(
                                    torch.mean(torch.exp(pos_dist), dim=1) +
                                    neg_dist))
                            diff_ctr_loss = torch.sum(neg_dist)
                            diff_neg_counter += pos_dist.shape[0]
            same_ctr_loss = same_ctr_loss / same_neg_counter
            diff_ctr_loss = diff_ctr_loss / diff_neg_counter
            same_hinge_loss = same_hinge_loss / same_neg_counter
            diff_hinge_loss = diff_hinge_loss / diff_neg_counter

            penalty_same_ctr += float(same_ctr_loss)
            penalty_diff_ctr += float(diff_ctr_loss)
            penalty_same_hinge += float(same_hinge_loss)
            penalty_diff_hinge += float(diff_hinge_loss)
            print(f"penalty_diff_hinge:{diff_hinge_loss}")
            loss_e += (
                (epoch - args.penalty_s) /
                (args.ctr_pretrain_epochs - args.penalty_s)) * diff_hinge_loss
            loss_e += diff_hinge_loss
            if domain_discriminator is not None:
                loss_e += d_loss
                print(f"d_loss:{d_loss}")
            print(f"loss_e:{loss_e}")
            if args.wandb is not None:
                wandb.log({
                    f"{args.heldout}_diff_hinge_loss":
                    diff_hinge_loss,
                    f"{args.heldout}_diff_ctr_loss":
                    diff_ctr_loss,
                    f"{args.heldout}_loss_e":
                    loss_e,
                    f"{args.heldout}_d_loss":
                    d_loss if domain_discriminator is not None else None,
                    f"{args.heldout}_d_suppression":
                    supression if domain_discriminator is not None else None
                })
            for optim in optimizers:
                optim.zero_grad()
            loss_e.backward()
            for optim in optimizers:
                optim.step()

        print(
            f'penalty_same_ctr:{penalty_same_ctr}, penalty_diff_ctr:{penalty_diff_ctr} \n, penalty_same_hinge:{penalty_same_hinge}, penalty_diff_hinge:{penalty_diff_hinge}'
        )
        print(f'avg_penalty_diff_hinge:{penalty_diff_hinge/i}')
        print(
            f'skip_pos_neg_feat:{skip_pos_neg_feat},skip_pos_feat_c:{skip_pos_feat_c}'
        )
        print('Done Training for epoch: ', epoch)
        if args.wandb is not None:
            wandb.log({
                f"{args.heldout}_epoch":
                epoch,
                f"{args.heldout}_penalty_diff_ctr":
                penalty_diff_ctr,
                f"{args.heldout}_penalty_diff_hinge":
                penalty_diff_hinge,
                f"{args.heldout}_avg_penalty_diff_hinge":
                (penalty_diff_hinge / i),
                f"{args.heldout}_skip_pos_neg_feat":
                skip_pos_neg_feat,
                f"{args.heldout}_skip_pos_feat_c":
                skip_pos_feat_c
            })
        if not args.end2end:
            if (epoch + 1) % args.save_model_freq == 0:
                print(
                    f'Saving the contrastive pretrained model at epoch {epoch+1}'
                )
                torch.save(
                    model.state_dict(),
                    f'{args.save_dir_}/ep{epoch+1}_ctr_pretrained_model_{args.heldout}.pt'
                )

    return model


def embedding_dist(x1, x2, tau=0.05, xent=False, metric='cos'):
    #the code is adapted from https://github.com/microsoft/robustdg/blob/514a3d92c8bf55d839a36ed0af654a63480dca8c/utils/helper.py#L80
    if xent:
        #X1 denotes the batch of anchors while X2 denotes all the negative matches
        #Broadcasting to compute loss for each anchor over all the negative matches

        #Only implemnted if x1, x2 are 2 rank tensors
        if len(x1.shape) != 2 or len(x2.shape) != 2:
            print(
                'Error: both should be rank 2 tensors for NT-Xent loss computation'
            )

        #Normalizing each vector
        ## Take care to reshape the norm: For a (N*D) vector; the norm would be (N) which needs to be shaped to (N,1) to ensure row wise l2 normalization takes place
        if torch.sum(torch.isnan(x1)):
            print('X1 is nan')
            sys.exit()

        if torch.sum(torch.isnan(x2)):
            print('X1 is nan')
            sys.exit()

        eps = 1e-8

        norm = x1.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x1 = x1 / torch.max(norm, temp)

        if torch.sum(torch.isnan(x1)):
            print('X1 Norm is nan')
            sys.exit()

        norm = x2.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x2 = x2 / torch.max(norm, temp)

        if torch.sum(torch.isnan(x2)):
            print('Norm: ', norm, x2)
            print('X2 Norm is nan')
            sys.exit()

        # Boradcasting the anchors vector to compute loss over all negative matches
        x1 = x1.unsqueeze(1)
        cos_sim = torch.sum(x1 * x2, dim=2)
        cos_sim = cos_sim / tau

        if torch.sum(torch.isnan(cos_sim)):
            print('Cos is nan')
            sys.exit()

        loss = torch.mean(torch.exp(cos_sim), dim=1)

        if torch.sum(torch.isnan(loss)):
            print('Loss is nan')
            sys.exit()

        return loss
    elif metric == 'cos':
        return cosine_similarity(x1, x2)


def cosine_similarity(x1, x2):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    return 1 - cos(x1.unsqueeze(1), x2)


def ctr_with_erm(args, epoch, feat, y, d):
    loss_e = torch.tensor(0.0).to(args.device)
    diff_ctr_loss = torch.tensor(0.0).to(args.device)
    diff_hinge_loss = torch.tensor(0.0).to(args.device)
    skip_pos_neg_feat = 0
    skip_pos_feat_c = 0
    diff_neg_counter = 1
    for y_d in range(args.num_src_domains):
        pos_indices = d[:] == y_d
        neg_indices = d[:] != y_d
        pos_feat = feat[pos_indices]
        pos_y = y[pos_indices]
        neg_feat = feat[neg_indices]
        neg_y = y[neg_indices]
        if pos_feat.shape[0] == 0 or neg_feat.shape[0] == 0:
            skip_pos_neg_feat += 1
            continue
        for y_ci in range(args.num_src_domains):
            pos_indices_ci = pos_y[:] == y_ci
            pos_feat_ci = pos_feat[pos_indices_ci]

            neg_dist = embedding_dist(pos_feat_ci,
                                      neg_feat,
                                      args.tau,
                                      xent=True)

            if torch.sum(torch.isnan(neg_dist)):
                print('Neg Dist Nan')
                sys.exit()
            for y_cj in range(args.num_classes):
                if y_ci != y_cj:
                    pos_indices_cj = pos_y[:] == y_cj
                    pos_feat_cj = pos_feat[pos_indices_cj]
                    if pos_feat_ci.shape[0] == 0 or pos_feat_cj.shape[0] == 0:
                        skip_pos_feat_c += 1
                        continue
                    pos_dist = 1 - embedding_dist(
                        pos_feat_ci, pos_feat_cj, metric='cos')
                    pos_dist = pos_dist / args.tau
                    # pos_dist = torch.mean(pos_dist, dim=1)
                    if torch.sum(torch.isnan(pos_dist)):
                        print('Pos Dist Nan')
                        sys.exit()
                    if torch.sum(
                            torch.isnan(
                                torch.log(
                                    torch.mean(torch.exp(pos_dist), dim=1) +
                                    neg_dist))):
                        print('Xent Nan')
                        sys.exit()
                    diff_hinge_loss += -1 * torch.sum(
                        torch.mean(pos_dist, dim=1) - torch.log(
                            torch.mean(torch.exp(pos_dist), dim=1) + neg_dist))
                    diff_ctr_loss = torch.sum(neg_dist)
                    diff_neg_counter += pos_dist.shape[0]

    diff_ctr_loss = diff_ctr_loss / diff_neg_counter
    diff_hinge_loss = diff_hinge_loss / diff_neg_counter
    loss_e += ((epoch - args.penalty_s) /
               (args.ctr_pretrain_epochs - args.penalty_s)) * diff_hinge_loss
    return diff_hinge_loss
