from torch import nn
import torch


class predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 19
        self.num_anchors = 9
        self.make_headers()
        self.reset_parameters()

    def forward(self, features):
        pred_scores = []
        pred_locs = []
        batch_size = features[0].size(0)
        # 把五个特征图分别进行定位与分类卷积操作,然后cat合并
        for feature in features:
            # 当使用permute transpose 或者.T这种改变tensor的stride时可以使用 reshape() 代替 .contiguous().view()
            pred_scores.append(self.cls_headers(feature).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            pred_locs.append(self.reg_headers(feature).permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
        # (75*75+38*38+19*19+10*10+5*5)*9 = 67995
        pred_scores = torch.cat(pred_scores, dim=1)
        pred_locs = torch.cat(pred_locs, dim=1)
        return pred_scores, pred_locs

    def make_headers(self):
        cls_headers = []
        reg_headers = []

        for _ in range(4):
            cls_headers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            cls_headers.append(nn.ReLU(inplace=True))

            reg_headers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            reg_headers.append(nn.ReLU(inplace=True))

        cls_headers.append(nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, stride=1, padding=1))
        reg_headers.append(nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, stride=1, padding=1))

        self.cls_headers = nn.Sequential(*cls_headers)
        self.reg_headers = nn.Sequential(*reg_headers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
