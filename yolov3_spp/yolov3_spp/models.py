from build_utils.layers import *
from build_utils.parse_config import *

# 搭建网络模块
ONNX_EXPORT = False

# 通过cfg文件得到模型
def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)，将第一个模块弹出，因为是yolo参数字典，不需要使用到
    output_filters = [3]  # input channels，输出特征矩阵的channels，3是输入图片的channels
    module_list = nn.ModuleList() # ModuleList将列表转成nn操作
    # routs统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs): # enumerate枚举
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            # stride参数，可能出现x和y分别具有自己的步幅的情况，所以要考虑此情况
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int): #卷积层尺寸是整数
                # 添加卷积层
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else: #报错，卷积核尺寸不是整数
                raise TypeError("conv2d filter size must be int type.")

            if bn: #如果使用bn层
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor（最后三个用于特征图输出）
                routs.append(i)  # detection output (goes into yolo layer)，将当前层的索引加入routs

            if mdef["activation"] == "leaky": #加入激活函数LeakyReLU
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d": #在yolov3不存在单独的BatchNorm2d
            pass

        elif mdef["type"] == "maxpool": #最大池化
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample": #上采样
            # stride里面的值就是上采样率
            # 是否要导出ONNX，默认不导出，忽略
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"]) #传入上采样率

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            # 如果l小于0，则不做改变
            # 如果l大于0（正着数），则要+1，因为output_filters第一个是输入通道3
            # 但是filter都是负值，所以问题不大
            # 如果layers不止一个值的话，则用sum函数求和
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # l>0则直接加入routs，否则i+l倒回到需要加的层
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers) # 使用FeatureConcat创建模块

        elif mdef["type"] == "shortcut": #残差相加模块
            layers = mdef["from"]
            filters = output_filters[-1] #上一层的channels
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0]) #从那一层进行输出
            # weights_type参数没有使用到
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            #  anchors有9个，通过mask进行控制选择哪几个anchor
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index]) #哪个yolo的预测头，选择对应的stride（缩放比例）

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try: #初始化Conv2d的偏置，来自于focal论文
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules) #将模块加入模块列表中
        output_filters.append(filters) #将filters加入的output_filters

    routs_binary = [False] * len(modules_defs) #建立一个bool矩阵，长度是modules_defs中模块个数
    for i in routs: #如果需要用到，则将其值赋值为True
        routs_binary[i] = True
    return module_list, routs_binary

# yolov3的三个输出层
class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors) #将anchors变成tensor格式
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)，每个特征图上anchors的个数
        self.nc = nc  # number of classes (80)，类别数
        # 对于每个anchor预测nc+5个参数。
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 因为anchor的值都是相对于原图的大小，所以要将anchors大小缩放到grid尺度，缩放倍数就是stride
        self.anchor_vec = self.anchors / self.stride

        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        # 进行这种格式是为了后续进行broacast
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng #将ng分布给nx和ny
        self.ng = torch.tensor(ng, dtype=torch.float) #将ng变成tensor格式

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            # 得到一个二维网格图
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            # 用stack将x与y的坐标进行拼接
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device: #如果anchor—vec与当前设备不相同，则将其放入设备当中
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p): #p是预测参数
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device) #重新生成参数

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        # 利用contiguous使得内存连续
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training: #训练模式，直接返回p
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            # 计算边界框的gird网格中的xy位置，io前两个是x，y
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            # 计算边界框的gird网格中的wh位置，io第3、4个是w，h
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度，乘以缩放比例
            torch.sigmoid_(io[..., 4:]) #对obj和classes预测值通过sigmoid函数
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

# yolo-spp模型搭建
class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        # 这里传入的img_size只在导出ONNX模型时起作用，img_size在训练时不起作用
        super(Darknet, self).__init__()
        # 如果只传入一个值（int），则变成[img_size,img_size]形式
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件，得到模型列表，模型每一个参数是一个字典
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构，调用create_modules函数，一层一层去搭建
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印每个层的详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False): #一次前向传播函数
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x) #不在上面的情况，直接将x进行正向传播即可

            out.append(x if self.routs[i] else []) #routs保存对应的模块
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train，训练模式直接返回前向传播的值
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)

# 获取网络中三个"YOLOLayer"模块对应的索引
def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    # 方法，遍历module_list，如果当前模块的名称是YOLOLayer，则将其索引加入列表中
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



