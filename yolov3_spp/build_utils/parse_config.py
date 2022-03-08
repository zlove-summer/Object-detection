import os
import numpy as np

# 解析cfg配置文件，将配置文件搭建成字典形式

def parse_model_cfg(path: str): #path是cfg文件路径
    # 检查文件是否以.cfg结尾、文件是否存在
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 读取文件信息
    with open(path, "r") as f:
        lines = f.read().split("\n") # 以换行符"\n"进行分割

    # 去除空行和注释行
    # 如果当前行不为空，并且当前行不以#开头，则将其存入lines列表当中
    lines = [x for x in lines if x and not x.startswith("#")]

    # 遍历lines，去除每行开头和结尾的空格符
    # strip去除首尾特定符号函数，如果不传参数，则默认去除空格和换行符
    lines = [x.strip() for x in lines]

    mdefs = []  # module definitions
    for line in lines: #遍历lines
        if line.startswith("["):  # this marks the start of a new block，表示开始一个block
            mdefs.append({}) # 添加一个字典
            # mdefs[-1]是最后一个层（也就是最后加入的层），加入type键值
            # [convolutional]，line[1:-1]取出“convolutional”
            mdefs[-1]["type"] = line[1:-1].strip()  # 记录module类型
            # 如果是卷积模块，设置默认不使用BN(普通卷积层后面会重写成1，最后的预测层conv保持为0)
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0
        else: #不是以“[”开通，就是一系列参数比如size、stride、pad等
            key, val = line.split("=") #用“=”号将两者分开，分别为键、值
            # 去除key、val前面的空格
            key = key.strip()
            val = val.strip()

            if key == "anchors":
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")  # 将空格去除
                # val去除空格后变成这样：10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326
                # 用","进行分割，分割后reshape成(9,2)形状
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
            # key为其他值:"from", "layers", "mask"
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                mdefs[-1][key] = [int(x) for x in val.split(",")] #对值进行分割，以整形类型放入列表
            else: #其它值，比如pad等等
                # TODO: .isnumeric() actually fails to get the float case
                if val.isnumeric():  # return int or float 如果是数值的情况
                    # if (int(val) - float(val)) == 0
                    # 如果val是int类型，那么int(val) - float(val)的值为0
                    # 否则，int(val) - float(val)的值不为0
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string  是字符的情况，直接传入字典中

    # check all fields are supported，所有支持的网络操作
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    # 遍历检查每个模型的配置
    for x in mdefs[1:]:  # 0对应net配置，不需要使用到
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported: #如果不支持，则报错
                raise ValueError("Unsupported fields:{} in cfg".format(k))

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

if __name__=="__main__":
    a=parse_model_cfg("..\cfg\yolov3-spp.cfg")
    print("end")
