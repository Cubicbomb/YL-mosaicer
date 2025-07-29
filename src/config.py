import os

class Config:
    def __init__(self):
        # 模型配置
        self.model_path = "yolov8n.pt"  # 使用轻量级nano模型
        self.confidence_threshold = 0.5
        
        # 路径配置
        self.input_dir = os.path.join(os.getcwd(), "input_videos")
        self.output_dir = os.path.join(os.getcwd(), "output_videos")
        self.input_video_name = "input.mp4"
        
        # 模糊配置
        self.blur_strength = 25  
        # 高斯模糊核大小，控制行人区域的模糊程度
        # 数值越大模糊效果越强，建议取值范围：15-40
        # 推荐值：25（平衡模糊效果和计算速度）
        
        # 创建目录
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)