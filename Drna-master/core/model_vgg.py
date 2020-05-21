from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import vgg
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(512, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)

class aug_net(nn.Module):
    def __init__(self, NUM_CLASSES, M=32, net = None):
        super(aug_net, self).__init__()
        self.num_classes = NUM_CLASSES
        self.M = M
        self.num_features = 2048
        self.expansion = 1

        self.features = net.get_features()
        self.expansion = self.features[-1][-1].expansion
        self.num_features = 512
        self.attentions = nn.Conv2d(self.num_features * self.expansion, self.M, kernel_size=1, bias=False)
        self.fc = nn.Linear(self.M * self.num_features * self.expansion, self.num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        feature_maps = self.features(x)
        attention_maps = self.attentions(feature_maps)
        k_indices = np.random.randint(self.M, size=batch_size)
        attention_map = torch.zeros(batch_size, -1).to(torch.device("cuda"))
        for i in range(batch_size):
            attention_map[i] = attention_maps[i, k_indices[i]:k_indices[i] + 1, ...]
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)
        attention_map = attention_map.view(batch_size, -1)
        attention_map_max, _ = attention_map.max(dim=1, keepdim=True)
        attention_map_min, _ = attention_map.min(dim=1, keepdim=True)
        attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)
        attention_map = attention_map.view(batch_size, -1)

        return attention_map

class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        # self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model = vgg.vgg16(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.classifier = nn.Linear(512, 2341)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.aug_net = aug_net()
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 2341)
        #self.concat_net = nn.Linear(2560, 2341)
        self.partcls_net = nn.Linear(512, 2341)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.upsample(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                   align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*2341
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        aug_logits = self.aug_net(part_features)
        # part_logits have the shape: B*N*2341
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, aug_logits, concat_logits, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
