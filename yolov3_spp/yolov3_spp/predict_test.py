# 用得到的网络，对图片进行一次预测

import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]

    #文件路径，并且判断文件是否存在
    cfg = "cfg/yolov3-spp.cfg"  # 改成生成的.cfg文件
    weights = "weights/best.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = "my_yolo_dataset\\val\\images\\000000000009.jpg" #测试图片
    # 判断文件是否存在，否则报错
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    # 加载类别文件，将每个类别的名称和编码放入字典中
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file) #加载josn文件
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()} #生成字典

    input_size = (img_size, img_size) #输入图片的尺寸

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #设置设备

    model = Darknet(cfg, img_size) #实例化模型
    model.load_state_dict(torch.load(weights, map_location=device)["model"]) #载入模型权重
    model.to(device) #将模型放入设备中

    model.eval() #模型改为验证模式，不会进行反向传播，只会进行前向推理和预测
    with torch.no_grad(): #禁止网络在训练过程中梯度更新
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device) #空图片
        model(img) #先用空图片进行正向传播，因为网络第一次预测，需要加载很多数值，速度较慢，所以选择空图片测试

        # 注意opencv读取的图片是BGR格式，不是RGB格式，需要转换
        img_o = cv2.imread(img_path)  # BGR，读取图片
        assert img_o is not None, "Image Not Found " + img_path #报错，图片不存在

        # 图片尺寸调整，并不是变成512*512，而是长边变成512，短边等比扩大（用0填充）
        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416，通道移到最前面来
        img = np.ascontiguousarray(img) #图像是不是连续的

        img = torch.from_numpy(img).to(device).float() #图片变成tensor格式
        img /= 255.0  # scale (0, 255) to (0, 1)，缩放
        img = img.unsqueeze(0)  # add batch dimension，新增一个batch维度

        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]  # only get inference result，得到模型的输出
        t2 = torch_utils.time_synchronized()
        print(t2 - t1) #花费时间计算

        # 对输出进行非极大值抑制处理
        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print(t3 - t2)

        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections，将预测图片的尺寸缩放成原尺寸，因为刚才进行了缩放
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)

        # 预测框、得分、类别信息
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        # draw_box方法进行绘制预测框
        img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
        plt.imshow(img_o)
        plt.show()

        img_o.save("test_result.jpg") #将得到的图片进行保存


if __name__ == "__main__":
    main()
