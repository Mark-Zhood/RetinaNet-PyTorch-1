class Config:
    env = 'RetinaNet'                                   # Visdom 可视化环境名称
    train_dir = r'data\train.txt'  # 训练集  以下文件路径均为相对路径
    val_dir = r'data\val.txt'      # 验证集
    test_dir = r'test'                       # 测试图片文件夹
    resnet_path = r'D:\py_pro\RetinaNet-PyTorch\resnet50-19c8e357.pth'   # resnet的预训练权重,默认resnet50
    load_path = r'map_0.9652.pt'  # 基于此模型权重训练
    video_path = r'zzz.avi'
    pretrain = False         # 是否基于已有的权重基础上继续训练
    height = 500            # 网络输入尺寸
    num_workers = 2         # 取决于你的cpu核数,比如9400F是六核的,建议2~4之间会比较好
    test_num_workers = 2    # 同上
    res_name = 'resnet50'   # 选用哪种ResNet
    use_adam = True         # 是否使用Adam优化方式,如果使用Adam的话需要一个相对于SGD更小的学习率比如 1e-4
    weight_decay = 0.0005   # 权重衰减系数
    lr_decay = 0.1          # 每隔指定epoch学习率下降的倍数
    lr = 1e-4               # 初始学习率
    epoch = 20              # 训练的轮数
    batch_size = 4

    # center_variance(xy)和size_variance(wh)是可以调整loc损失在整体loss中的比例
    # 参考https://github.com/weiliu89/caffe/issues/629
    center_variance = 0.1
    size_variance = 0.2

    iou_threshold = 0.5     # assign_anchors方法中的判断正负样本的IOU参数
    # 类别置信度阈值,网络输出最终结果时会过滤掉小于此值的pred_box
    # 也可以作为权衡recall和precision的指标,该值越大,recall越大,precision越小.反之同样
    score_threshold = 0.9   # NMS时 过滤掉score低于此值的pred_box
    iou_nms = 0.45          # 此值为最后进行NMS操作时其中的IOU参数

    alpha = 0.25            # focal loss 阿尔法参数 float 或者 list ,用于调整正负样本的权重 背景:正样本 = alpha:(1-alpha)
    gamma = 2               # focal loss 伽马参数  ,用于控制难易分类样本的权重
    neg_pos_ratio = 3       # 负正例比例,用于hard_negative_mining中的参数
    class_name = ("__background__", "WhitehairedBanshee", "UndeadSkeleton", "WhitehairedMonster", "SlurryMonster",
                  "MiniZalu", "Dopelliwin", "ShieldAxe", "SkeletonKnight", "Zalu", "Cyclone", "SlurryBeggar",
                  "Gerozaru", "Catalog", "InfectedMonst", "Gold", "StormRider", "Close", "Door",)
    num_class = len(class_name)


cfg = Config()
