import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
   
    def parsing_loss_bk(self, preds, target):
        h, w = target[0].size(1), target[0].size(2)

        target[1] = torch.clamp(target[1], 0, 1)
        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target[0])
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])

        # loss for edge
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)

        return loss
        
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)
        loss = 0

        # loss for parsing
        preds_parsing = preds
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target)
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target)


        return loss
    def forward(self, preds, target):  
        loss = self.parsing_loss_bk(preds, target) 
        return loss
    
class CriterionCrossEntropyEdgeParsing_boundary_attention_loss(nn.Module):
    """Weighted CE2P loss for face parsing.
    
    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """
    def __init__(self, loss_weight=[1.0, 1.0, 1.0, 1.0], ignore_index=255, num_classes=11):
        super(CriterionCrossEntropyEdgeParsing_boundary_attention_loss, self).__init__()
        self.ignore_index = ignore_index   
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index) 
        self.loss_weight = loss_weight
          
    def forward(self, pred, target):
        # pred: seg_pred
        # target: seg_label
        h, w = target.size(1), target.size(2)

        scale_parse = F.interpolate(input=pred, size=(h, w), mode='bilinear') # parsing
        loss_parse = self.criterion(scale_parse, target)

        return loss_parse

class DiscriminativeLoss(nn.Module):

    def __init__(self, n_classes, alpha=.05, beta=1., ignore_label=255):
        super(DiscriminativeLoss, self).__init__()

        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, embedding, label):

        # embedding: (n, c, h', w')
        # label: (n, h, w)
        # alpha: scalar, maximum intra-cluster radius
        # beta: scalar, minimum inter-cluster radius

        label[label == 255] = 0

        N, C, H, W = embedding.size()
        label = F.interpolate(label.unsqueeze(1).float(), size=(H, W)).squeeze().long()
        one_hot_label = torch.nn.functional.one_hot(label.view(-1), self.n_classes).view(N, H * W, -1).permute(0, 2,
                                                                                                          1)  # (n, K, h * w)
        count = one_hot_label.view(N, self.n_classes, -1).sum(2)  # (n, K)
        mask = (count > 0).int()  # (n, K)
        one_hot_label = one_hot_label.unsqueeze(2).repeat(1, 1, C, 1)  # (n, K, c, h * w)
        embedding = F.normalize(embedding, dim=1)
        embedding = embedding.unsqueeze(1).repeat(1, self.n_classes, 1, 1, 1).view(N, self.n_classes, C, -1)  # (n, K, c, h * w)
        embedding = embedding * one_hot_label
        embedding_mean = embedding.sum(3) / (count.unsqueeze(2) + 1)  # (n, K, c)
        intra_dist = (embedding_mean.unsqueeze(3) - embedding) * one_hot_label  # (n, K, c, h * w)
        l2_intra_dist = (intra_dist ** 2).sum(2)  # (n, K, h * w)
        intra_mask = l2_intra_dist > self.alpha
        l2_intra_dist = (l2_intra_dist - self.alpha) * intra_mask
        l2_intra_dist = l2_intra_dist[:, 1:].sum(2).sum(1) / (intra_mask[:, 1:].sum(2).sum(1) + 1)

        inter_dist = embedding_mean.unsqueeze(1).repeat(1, self.n_classes, 1, 1) - embedding_mean.unsqueeze(2).repeat(1, 1, self.n_classes, 1)  # (n, K, K, c)
        l2_inter_dist = (inter_dist ** 2).sum(3)  # (n, K, K)
        l2_inter_dist = l2_inter_dist * mask.unsqueeze(1).repeat(1, self.n_classes, 1)
        l2_inter_dist = l2_inter_dist * mask.unsqueeze(2).repeat(1, 1, self.n_classes)
        inter_mask = (l2_inter_dist < self.beta) * (l2_inter_dist > 0)
        l2_inter_dist = (self.beta - l2_inter_dist) * inter_mask
        l2_inter_dist = l2_inter_dist[:, 1:, 1:].reshape(N, -1).sum(1) / (inter_mask[:, 1:].sum(2).sum(1) + 1) / 2
        #l2_inter_dist = l2_inter_dist[:, 1:, 1:].reshape(N, -1).sum(1) / (mask[:, 1:].sum(1) * mask[:, 1:].sum(1))  # (n)

        return l2_intra_dist.mean(0), l2_inter_dist.mean(0)

