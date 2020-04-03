代码源自 https://github.com/yatengLG/Retinanet-Pytorch

在此基础上进行修改与注释(删除了数据增强部分),因为我发现对于部分场景下的任务.数据增强反而会降低模型的mAP,以及会放慢收敛的速度

还有一些比如config文件,数据预处理部分,以及一些函数的修改及其位置的调整

所做的一切目的都是为了代码更简洁便于理解

训练:config文件配置好之后直接运行train即可训练(同时每个epoch都可以验证mAP)

测试:运行dection_image.py或者dection_video.py文件即可

###### NMS阶段会耗费大量的时间,导致此版本代码的FPS约为10.原始论文中FPS为14.