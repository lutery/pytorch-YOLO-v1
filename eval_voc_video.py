#encoding:utf-8
#
#created by xiongzihua
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from predict import *
from collections import defaultdict
from tqdm import tqdm
import cv2

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

if __name__ == '__main__':
    #test_eval()

    preds = defaultdict(list)
    image_list = [] #image path list

    print('---start test---')
    model = resnet50()
    #             #nn.Linear(4096, 4096),
    #             #nn.ReLU(True),
    #             #nn.Dropout(),
    #             nn.Linear(4096, 1470),
    #         )
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()

    cap = cv2.VideoCapture("D:\Projects\datasets\oc\TK100\data\test.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("D:\Projects\datasets\oc\TK100\data\test-result.mp4", fourcc, 30, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        result = predict_gpu_opencvimg(model, frame)
        for left_up,right_bottom,class_name,_,prob in result:
            color = Color[VOC_CLASSES.index(class_name) % len(Color)]
            cv2.rectangle(frame,left_up,right_bottom,color,2)
            label = class_name+str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1]- text_size[1])
            cv2.rectangle(frame, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(frame, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('---start evaluate---')