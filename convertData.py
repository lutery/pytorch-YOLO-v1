# 实现增加TK100数据集转换为yolov1的数据集格式，文件内容格式：图片路径 x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
# 用于存储转换结果的列表
import json
import numpy as np

json_file_path = r"F:\Projects\datasets\oc\TK100\data\annotations.json"
with open(json_file_path, "r") as file:
    json_data = json.load(file)

train_yolo_format_data = []
test_yolo_format_data = []

# 类别名称到 ID 的映射
types = json_data["types"]
category_to_id = {category: idx for idx, category in enumerate(types)}

def convert_to_yolo_format(image_path, objects):
    yolo_data_line = image_path
    for obj in objects:
        x1 = obj["bbox"]["xmin"]
        y1 = obj["bbox"]["ymin"]
        x2 = obj["bbox"]["xmax"]
        y2 = obj["bbox"]["ymax"]
        class_id = category_to_id[obj["category"]]

        # 添加到行数据中
        yolo_data_line += f" {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(class_id)}"

    return yolo_data_line

# 解析 JSON 数据并转换格式
for img_id, img_info in json_data["imgs"].items():
    image_path = img_info["path"]
    objects = img_info["objects"]

    if (len(objects) <= 0):
        continue

    # 开始构建 YOLO 格式的字符串
    if 'test' in image_path:
        # 将完整的行添加到列表中
        test_yolo_format_data.append(convert_to_yolo_format(image_path, objects))
    elif 'train' in image_path:
        train_yolo_format_data.append(convert_to_yolo_format(image_path, objects))
    else:
        continue


# 将结果分为测试集（90%）和训练集（10%）
with open('tk100.txt', 'w') as file:
    for line in train_yolo_format_data:
        file.write(line + "\n")

with open('tk100-test.txt', 'w') as file:
    for line in test_yolo_format_data:
        file.write(line + "\n")