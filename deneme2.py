import torch
import cv2
import sys
from pathlib import Path
import pathlib

# Windows path compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

WEIGHTS_FILE = 'bester.pt'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import smart_inference_mode, select_device
from ultralytics.utils.plotting import Annotator, colors

class WebcamDetector:
    def __init__(self, weights=WEIGHTS_FILE, device='', conf_thres=0.25, iou_thres=0.45):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Get optimal image size
        self.imgsz = check_img_size((640, 640), s=self.stride)
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model inference"""
        # BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        img_resized = cv2.resize(img_rgb, (self.imgsz[1], self.imgsz[0]))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).to(self.device)
        img_tensor = img_tensor.float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        return img_tensor.unsqueeze(0)
    
    def process_detections(self, pred, original_frame):
        """Process and draw detections on frame"""
        annotator = Annotator(original_frame, line_width=2)
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.imgsz, det[:, :4], original_frame.shape)
                
                for *xyxy, conf, cls in reversed(det):
                    if conf > self.conf_thres:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, colors(int(cls), True))
        
        return annotator.result()

@smart_inference_mode()
def webcam_detect(weights=WEIGHTS_FILE):
    detector = WebcamDetector(weights)
    
    # Use DirectShow on Windows
    backend = cv2.CAP_DSHOW
    
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.imgsz[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.imgsz[1])
    
    print("Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Preprocess
            img_tensor = detector.preprocess_frame(frame)
            
            # Inference
            with torch.no_grad():
                pred = detector.model(img_tensor)
                pred = non_max_suppression(pred, detector.conf_thres, 
                                          detector.iou_thres, max_det=1000)
            
            # Process results
            result_frame = detector.process_detections(pred, frame)
            
            # Display
            cv2.imshow('YOLO Webcam Detection', result_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        webcam_detect(WEIGHTS_FILE)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        pathlib.PosixPath = temp