import torch
import cv2
import sys
from pathlib import Path
import pathlib 
temp = pathlib.PosixPath 
pathlib.PosixPath = pathlib.WindowsPath

bestfile='bester.py' #WRITE YOUR FİLE NAME HERE

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import smart_inference_mode, select_device
from ultralytics.utils.plotting import Annotator, colors

@smart_inference_mode()
def webcam_detect(weights='bester.pt'):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz[1])
    
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break
            
        #  BGR → RGB + RESIZE + NORMALIZE
        im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (imgsz[1], imgsz[0]))  # (width, height)
        im = im.transpose((2, 0, 1))  # HWC → CHW
        im = torch.from_numpy(im).to(device).float() / 255.0
        im = im[None]  # Batch dimension
        
        pred = model(im)
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)
        
        # Results
        annotator = Annotator(im0, line_width=3)
        if len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)
            for *xyxy, conf, cls in reversed(det):
                if conf > 0.25:  # Confidence filter
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, colors(int(cls), True))
        
        cv2.imshow('YOLO Webcam', annotator.result())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        webcam_detect('bester.pt')
    except KeyboardInterrupt:
        print("\nDurduruldu")
    finally:
        cv2.destroyAllWindows()
        pathlib.PosixPath = temp

pathlib.PosixPath = temp
