### 炼丹步骤

1 建立my_yolo_dataset文件夹，将数据集复制至此yolo-spp文件夹下my_yolo_dataset文件夹

2 修改数据集结构（为了和代码中路径相同，也可以根据相关文件进行修改）

```
├── my_yolo_dataset 自定义数据集根目录
│         ├── train   训练集目录
│         │     ├── images  训练集图像目录
│         │     └── labels  训练集标签目录 
│         └── val    验证集目录
│               ├── images  验证集图像目录
│               └── labels  验证集标签目录       
```

3 对自己的数据集，建立my_data_label.names，方法就是将每一类作为一行建立一个txt文件，最后修改文件后缀为.name格式。比如要对以下8个类别进行目标检测，就将其复制为到txt文件中，修改名称为my_data_label.names，

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
```

4 运行calculate_dataset.py文件，这两个文件会将my_yolo_dataset中的图片进行遍历，创建my_train_data.txt、my_val_data.txt两个文件，这两个文件中存储着需要训练或者验证的图片的路径。calculate_dataset.py还会创建mydata.data，其中记录了记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径。

5 下载或者更改权重文件，将权重保存在weights文件夹，在train.py权重路径中修改为对应的名称。如果是第一次训练，选择使用预训练权重，若有中断再继续训练，将train.py的权重路径更改为保留的权重。

6 选择epoch、batch_size等参数，然后运行train.py文件。
