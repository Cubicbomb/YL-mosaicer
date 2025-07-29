import sys
from pathlib import Path
# Add root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
import os
from ultralytics import YOLO
from config import Config
from utils.video_processor import VideoProcessor
from utils.blur_utils import blur_region

def main():
    """
    主函数：执行视频行人模糊处理流程
    
    输入输出视频路径、模糊程度等，
    配置请前往src/config.py文件

    输入视频请放入input_videos文件夹并命名为input.mp4
    输出视频将保存在output_videos文件夹
    """
    # 加载配置
    config = Config()
    
    # 检查输入视频是否存在
    input_video_path = os.path.join(config.input_dir, config.input_video_name)
    if not os.path.exists(input_video_path):
        print(f"错误: 未找到输入视频文件 {input_video_path}")
        print("请将视频文件放入input_videos文件夹并命名为input.mp4")
        return
    
    # 加载YOLOv8模型
    model = YOLO(config.model_path)
    
    # 初始化视频处理器
    video_processor = VideoProcessor(input_video_path, config.output_dir)
    
    # 处理视频
    print("开始处理视频...")
    video_processor.process_video(model, lambda frame, coords: blur_region(frame, coords, config.blur_strength))
    print(f"视频处理完成！结果保存在: {video_processor.output_path}")

if __name__ == "__main__":
    main()