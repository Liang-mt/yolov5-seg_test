import numpy as np
import torch

import cv2

from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow,
    check_requirements, colorstr, cv2, increment_path, non_max_suppression,
    print_args, scale_boxes, scale_segments, strip_optimizer
)
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox


class yolov5_seg():
    def __init__(self,
                 weights="./yolov5s-seg.pt",
                 data="./data/coco128.yaml",
                 imgsz=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.4,
                 max_det=1000,
                 device='cpu',
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 visualize=False,
                 line_thickness=3,
                 half=False,
                 dnn=False,
                 vid_stride=1,
                 retina_masks=False):
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        self.retina_masks = retina_masks

    def infer(self, image):
        bs = 1
        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(
            self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half
        )
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt = [0.0, 0.0, 0.0]

        img = letterbox(image, self.imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        t1 = time_sync()

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        t2 = time_sync()
        dt[0] += t2 - t1

        pred, proto = model(im, augment=self.augment, visualize=self.visualize)[:2]
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
            max_det=self.max_det, nm=32
        )
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            im0 = image
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
            if len(det):
                if self.retina_masks:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                segments = [
                    scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                    for x in reversed(masks2segments(masks))
                ]
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() / 255 if self.retina_masks else im[i]
                )

                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    seg = segments[j].reshape(-1)
                    line = (cls, *seg)
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        return im0


if __name__ == "__main__":
    img = cv2.imread("./images/bus.jpg")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = yolov5_seg()
    model.device = device
    img = model.infer(img)
    cv2.imshow("1", img)
    cv2.waitKey(0)