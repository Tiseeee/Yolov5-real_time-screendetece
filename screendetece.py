import torch
import numpy as np
import win32gui, win32con, cv2
from grabscreen import grab_screen   # 本地文件grabscreen.py中的定义
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
 
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
 
 
# 可调参数q
conf_thres = 0.25
iou_thres = 0.45
thickness = 3
x, y = (1080, 900)
re_x, re_y = (1080, 900)
 
 
 
def LoadModule():
    device = select_device('')
    weights = 'D:\\vc product\\yolov5-master\\best.pt'  #设置自己的权重
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    return model
 
 
model = LoadModule()
 
#CV2初始化放在while外面，避免重复的初始化显示区域的大小
cv2.namedWindow('windows', cv2.WINDOW_NORMAL)
cv2.resizeWindow('windows', 1080,600) #窗口分辨率
 
while True:
    names = model.names
    img0 = grab_screen(region=(460, 300, 2000, 1200))  #屏幕坐标
    im = letterbox(img0, 640, stride=32, auto=True)[0]        # 缩放
    im = im.transpose((2, 0, 1))[::-1]                                  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)                                       # 将图像数组转换为连续的内存布局
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()                        # uint8 to fp16/32
    im /= 255                                                           # 0 - 255 to 0.0 - 1.0，输入数据的归一化操作
    if len(im.shape) == 3:
        im = im[None]                                                   #将单张图像扩展为一个批次维度
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False,
                               max_det=1000)                            #经过NMS跟IOU之后最多保留1000个结果
    boxs=[]
    for i, det in enumerate(pred):  # per image
        im0 = img0.copy()
        s = ' '
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]             # 提取whwh
        imc = img0  # for save_crop
 
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()                      # 检测出现的次数
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "    #目标出现不止一次，追加s表示复数
 
        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()    #格式，一维变二维再一维，归一化处理
            line = (cls, *xywh)                                 # label format，添加标签组成新元组
            box = ('%g ' * len(line)).rstrip() % line           #拼接成一个字符串`box`
            box = box.split(' ')
            boxs.append(box)                                    #拆分成一个列表形式的`box`，然后将其添加到`boxs`列表中
        if len(boxs):
            for i, det in enumerate(boxs):
                _, x_center, y_center, width, height = det      #解包
                x_center, width = re_x * float(x_center), re_x * float(width)   #前面在计算gn时进行了归一化
                y_center, height = re_y * float(y_center), re_y * float(height)
                top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                color = (0, 0, 255)  # RGB
                cv2.rectangle(img0, top_left, bottom_right, color, thickness=thickness)
 
                # 添加以下代码用于显示类别文本
                category_text = names[int(cls)]
                text_size, _ = cv2.getTextSize(category_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                text_x = top_left[0]
                text_y = top_left[1] - 5 if top_left[1] - 5 > 0 else top_left[1] + text_size[1] + 5
                cv2.putText(img0, category_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow()
        break
    cv2.imshow('windows', img0)
    HWND = win32gui.FindWindow(None, "windows")
    win32gui.SetWindowPos(HWND, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)