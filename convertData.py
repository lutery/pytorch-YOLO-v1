# 实现增加TK100数据集转换为yolov1的数据集格式，文件内容格式：图片路径 x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
# 用于存储转换结果的列表
import json
import numpy as np

json_file_path = ""
with open(json_file_path, "r") as file:
    json_data = json.load(file)

yolo_format_data = []

# 类别名称到 ID 的映射
types = json_data["types"]
category_to_id = {category: idx for idx, category in enumerate(types)}

# 解析 JSON 数据并转换格式
for img_id, img_info in json_data["imgs"].items():
    image_path = img_info["path"]
    objects = img_info["objects"]

    if (len(objects) <= 0):
        continue

    # 开始构建 YOLO 格式的字符串
    yolo_data_line = image_path

    for obj in objects:
        x1 = obj["bbox"]["xmin"]
        y1 = obj["bbox"]["ymin"]
        x2 = obj["bbox"]["xmax"]
        y2 = obj["bbox"]["ymax"]
        class_id = category_to_id[obj["category"]]

        # 添加到行数据中
        yolo_data_line += f" {x1} {y1} {x2} {y2} {class_id}"

    # 将完整的行添加到列表中
    yolo_format_data.append(yolo_data_line)

# 将结果分为测试集（90%）和训练集（10%）
np.random.default_rng(42).shuffle(yolo_format_data)
with open('tk100.txt', 'w') as file:
    for line in yolo_format_data[:int(len(yolo_format_data) * 0.9)]:
        file.write(line + "\n")

with open('tk100-test.txt', 'w') as file:
    for line in yolo_format_data[int(len(yolo_format_data) * 0.9)]:
        file.write(line + "\n")