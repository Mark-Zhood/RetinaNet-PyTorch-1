from model.Retinanet import RetinaNet
from dataset import ListDataset
import torch
import visdom
from tqdm import tqdm
from config import cfg
from utils.eval_tools import Eval
from utils.loss import multiboxloss
from terminaltables import AsciiTable
from torch.utils.data import DataLoader


train_dataset=ListDataset(cfg.train_dir, is_train=True)
test_dataset = ListDataset(cfg.val_dir, is_train=False)

if __name__ == '__main__':
    data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    model = RetinaNet(cfg.res_name).cuda()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=0.9,weight_decay=cfg.weight_decay)
    if cfg.pretrain:
        model.load_state_dict(torch.load(cfg.load_path))

    loss_func = multiboxloss()
    vis = visdom.Visdom(env='RetinaNet')
    mAP = 0
    for epoch in range(1, cfg.epoch):
        for images, boxes, labels, image_names in tqdm(data_loader):
            images, boxes, labels = images.cuda(), boxes.cuda(), labels.cuda()
            pred_score, pred_loc = model(images)
            reg_loss, cls_loss = loss_func(pred_score, pred_loc, labels, boxes)
            reg_loss = reg_loss.mean()
            cls_loss = cls_loss.mean()
            loss = reg_loss + cls_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 绘制loss曲线
        vis.line(X=torch.tensor([epoch]),Y=torch.tensor([reg_loss]), win='reg_loss', update=None if epoch==1 else 'append', opts={'title': 'reg_loss'})
        vis.line(X=torch.tensor([epoch]),Y=torch.tensor([cls_loss]), win='cls_loss', update=None if epoch==1 else 'append', opts={'title': 'cls_loss'})
        vis.line(X=torch.tensor([epoch]), Y=torch.tensor([loss]), win='loss', update=None if epoch==1 else 'append', opts={'title': 'loss'})
        # 在验证集上计算mAP,以及输出 Precision Recall AP F1-score等指标
        eval_result = Eval(model=model, test_dataset=test_dataset)
        ap_table = [["Index", "Class name", "Precision", "Recall", "AP", "F1-score"]]
        for p, r, ap, f1, cls_id in zip(*eval_result):
            ap_table += [[cls_id, cfg.class_name[cls_id], "%.3f" % p, "%.3f" % r, "%.3f" % ap, "%.3f" % f1]]
        print('\n' + AsciiTable(ap_table).table)
        eval_map = round(eval_result[2].mean(),4)
        print("Epoch %d/%d ---- mAP:%.4f Loss:%.4f" % (epoch, cfg.epoch, eval_map, loss))
        vis.line( X=torch.tensor([epoch]), Y=torch.tensor([eval_map]),win='map', update=None if epoch == 1 else 'append',
             opts={'title': 'map'})
        # 如果某epoch的eval_map大于之前最大mAP,则保存该epoch的权重
        if eval_map > mAP:
            mAP = eval_map
            torch.save(model.state_dict(),'map_%s.pt' % mAP)