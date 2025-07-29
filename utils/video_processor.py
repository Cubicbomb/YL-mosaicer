import cv2
import os

class VideoProcessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(input_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = os.path.join(output_dir, "output_blurred.mp4")
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
    
    def process_video(self, model, blur_function):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测行人 (COCO数据集中类别0为行人)
            results = model(frame, classes=[0])
            
            # 处理每个检测结果
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 对检测到的区域进行模糊
                    frame = blur_function(frame, (x1, y1, x2, y2))
            
            self.out.write(frame)
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()