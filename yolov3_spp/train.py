import datetime
import argparse

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyp):
    #选择GPU设备，在opt参数中可以修改，目前默认使用GPU组中的第1块GPU
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    wdir = "weights" + os.sep  # weights dir，权重保存的文件夹
    best = wdir + "best.pt"  #最好的权重保存的位置和名称（best.pt）
    # results_file保存了最终模型的结果，mAP等等
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 一系列的赋值，将opt中的参数传入到train函数中
    cfg = opt.cfg
    data = opt.data # data/my_data.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    # 如果batch_size=4，则accumulate=16，每16步更新一次参赛，在此之前累计梯度
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # 初始训练权重，weights/yolov3-spp-ultralytics-512.pt
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数，最小图像是原来的1/32
    gs = 32  # (pixels) grid size，特征图缩小的最大倍数，训练自己的数据集要使得其尺寸是32的整数倍
    # 判断图像是否是32倍数，如果不是则抛出异常
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    # 当未进行尺寸放缩时，最大最小grid的尺寸相等
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs

    if multi_scale: #多尺度训练
        # 尺寸放缩后，最大尺寸和最小尺寸，最大为原来的1.5倍，最小为0.667倍
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 以下两行函数：将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size，训练图片尺寸初始化为最大尺寸
        # 打印出图片尺寸的最大最小范围
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    # init_seeds()  # 初始化随机种子，保证结果可复现
    data_dict = parse_data_cfg(data) #解析data数据集
    train_path = data_dict["train"] #训练集对应的txt
    test_path = data_dict["valid"] #验证集对应的txt
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # 物体类别，coco数据集是80类
    # 根据类别数和图像大小，挑战分类损失参数+物体置信度损失参数
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    # Remove previous results，移除以前的结果，如移除results20210515-152935.txt
    for f in glob.glob(results_file):
        os.remove(f)

    # Initialize model，将模型文件cfg传入Daekent函数，返回最终模型，并将模型放入设备中
    model = Darknet(cfg).to(device)

    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        # 总共训练3x2=6个parameters
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)
    else:
        # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
        # 若要训练全部权重，删除以下代码
        darknet_end_layer = 74  # only yolov3spp cfg
        # Freeze darknet53 layers
        # 总共训练21x3+3x2=69个parameters
        for idx in range(darknet_end_layer + 1):  # [0, 74]
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # optimizer
    # 将所有需要梯度的参数找出来并加入列表
    pg = [p for p in model.parameters() if p.requires_grad]
    # 使用SGD随机优化算法，将需要更新梯度的参数列表pg传入
    # 通过hyp字典，将模型的超参数传入（学习率、momentum冲量、权重衰减策略）
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    scaler = torch.cuda.amp.GradScaler() if opt.amp else None #进行混合精度训练

    start_epoch = 0 #起始epoch
    best_map = 0.0 #起始最佳map为0
    if weights.endswith(".pt") or weights.endswith(".pth"): #如果是以pt和pth结尾的训练权重
        ckpt = torch.load(weights, map_location=device) #加载模型，并map到cpu或gpu设备中

        # load model
        try:
            # 以字典形式保存模型需要的参数
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False) #将参数导入model中
        except KeyError as e:
            # 如果预训练权重的参数结构和model的结构不同，那么就报错
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e #报错

        # load optimizer，载入优化器
        # 优化器参数也是可学习的，需要将参数载入（比如已经训练了一段时间了，需要载入）
        # 优化器对象(connector .optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys(): #如果有best_map值，就将其导入（这是在模型已经训练了一段时间）
                best_map = ckpt["best_map"]

        # load results，训练结果
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # epochs，断点训练后，epoch改变
        start_epoch = ckpt["epoch"] + 1
        # epochs=10
        # start_epoch=29+1=30
        # 此时说明训练完毕，只需额外微调10轮
        if epochs < start_epoch: #如果已经训练完了，额外微调
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs，epochs=10+30=40

        if opt.amp and "scaler" in ckpt: #加载混合精度训练参数
            scaler.load_state_dict(ckpt["scaler"])

        del ckpt #删除ckpt文件

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 学习率曲线，cos曲线
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) #将函数应用到学习器中
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始，学习率开始更新的epochs

    # Plot lr schedule，画出学习率曲线
    # y = []
    # for _ in range(epochs): #跑epochs次
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # model.yolo_layers = model.module.yolo_layers

    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸，进行多尺度训练
    # 训练数据集
    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=True,
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=opt.rect,  # rectangular training
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls)

    # 验证集的图像尺寸指定为img_size(512)，因为是要验证，所以使用原图尺寸
    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

    # dataloader，nw是num—worker，取三者最小值
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # 训练集dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    # 验证集dataloader
    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # Model parameters，模型数据
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 计算每个类别的目标个数，并计算每个类别的比重
    # model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # start training
    # caching val_data when you have plenty of memory(RAM)
    # coco = None
    # 遍历数据集，将标签信息读取一遍
    coco = get_coco_api_from_dataset(val_dataset)

    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    # 开始训练
    for epoch in range(start_epoch, epochs):
        # 得到损失和学习率
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True,
                                               scaler=scaler)
        # update scheduler
        scheduler.step() #更新学习率

        if opt.notest is False or epoch == epochs - 1: #一次验证
            # evaluate on the test dataset
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            # write into tensorboard
            # 利用tensorboard进行绘制
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # write into txt
            # 将当前的指标写入txt文件
            with open(results_file, "a") as f:
                # 记录coco的12个指标加上训练总损失和lr
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # update best mAP(IoU=0.50:0.95)
            # 更新最佳损失
            if coco_mAP > best_map:
                best_map = coco_mAP

            #保存模型参数
            # False时，每次都会保存模型参数
            # True时，只会保存最佳模型参数
            if opt.savebest is False:
                # save weights every epoch
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
            else:
                # only save best weights
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, best.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 训练轮数，默认30轮
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    # cfg是yolov3-spp的网络结构参数文件，详细记录了yolov3每一层的卷积或者结构的stride、size、padding值等等
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help="*.cfg path")
    # data文件里面存在着训练、验证的txt名称
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    # hyp是训练需要使用的超参数
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    # 是否选用多尺度预测
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    # 数据集部分讲解此参数
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # 是否只保存最佳权重，默认是，若关闭，则每次验证都会保存
    parser.add_argument('--savebest', type=bool, default=True, help='only save best checkpoint')
    # 选择是否只在最后一次进行验证模型
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # 是否将图片加入缓存中，加入后读取文件速度会变快，加快训练速度
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    # 权重，预训练权重，如果训练了几个epoch后，又要继续训练，就将此文件改成上次训练保存的文件
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    # 所有类别的名称，对于coco数据集，则是80个类别
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    # 选择设备，默认第一款Gpu
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # 冻结训练，先训练head，在训练全部网络结构
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    # 是否使用混合精度训练(需要GPU支持混合精度，pytorch版本>1.6)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    opt = parser.parse_args(args=[])

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    #打开超参数，并且加载
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
