import os
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18


# 加载模型
use_resnet = True
if use_resnet:
    model = resnet50()
else:
    model = vgg16_bn()

model.load_state_dict(torch.load(r'M:\Projects\openSource\python\pytorch-YOLO-v1\best.pth'))


# 定义预处理函数
preprocess = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# 读取文件夹内的所有图片
image_dir = r'F:\Projects\datasets\oc\TK100\data\test'
image_files = os.listdir(image_dir)

# 定义保存图片的文件夹
save_dir = 'M:\\Projects\\openSource\\python\\pytorch-YOLO-v1\\testimg'
os.makedirs(save_dir, exist_ok=True)

for image_file in image_files:
    # 读取图片
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    # 对图片进行预处理
    image_tensor = preprocess(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    # 使用模型进行预测
    with torch.no_grad():
        preds = model(image_tensor)

    # 对预测结果进行处理，这部分取决于你的模型和你的需求
    # 这里只是一个示例，你可能需要修改这部分代码
    boxes1 = preds[..., :4]
    confidences1 = preds[..., 4]
    boxes2 = preds[..., 5:9]
    confidences2 = preds[..., 10]
    class_probs = preds[..., 11:]
    class_preds = torch.argmax(class_probs, dim=-1)
    draw = ImageDraw.Draw(image)
    for i in range(boxes1.shape[0]):
        if confidences1[i] > 0.5:
            x, y, w, h = boxes1[i]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            draw.rectangle(((x1, y1), (x2, y2)), outline='red')
            draw.text((x1, y1), str(class_preds[i].item()), fill='red')

        if confidences2[i] > 0.5:
            x, y, w, h = boxes2[i]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            draw.rectangle(((x1, y1), (x2, y2)), outline='red')
            draw.text((x1, y1), str(class_preds[i].item()), fill='red')

    # 保存图片
    image.save(os.path.join(save_dir, image_file))