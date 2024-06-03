import cv2
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load('best.pth')

# 定义预处理函数
preprocess = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# 打开视频文件
cap = cv2.VideoCapture('path_to_your_video')

while(cap.isOpened()):
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 对帧进行预处理
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image = preprocess(image)
    image = torch.unsqueeze(image, 0)

    # 使用模型进行预测
    with torch.no_grad():
        preds = model(image)

    # 对预测结果进行处理，这部分取决于你的模型和你的需求
    # 这里只是一个示例，你可能需要修改这部分代码
    boxes = preds[..., :4]
    confidences = preds[..., 4]
    class_probs = preds[..., 5:]
    class_preds = torch.argmax(class_probs, dim=-1)
    for i in range(boxes.shape[0]):
        if confidences[i] > 0.5:
            x, y, w, h = boxes[i]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(class_preds[i].item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()