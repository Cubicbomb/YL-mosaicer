import sys
from pathlib import Path
# Add root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
import cv2
import os
import glob
from ultralytics import YOLO
from config import Config
from utils.video_processor import VideoProcessor
from utils.blur_utils import blur_region


def main():
    """
    主函数：执行视频行人模糊处理流程
    
    输入输出视频路径、模糊程度等，
    配置请前往src/config.py文件

    输入视频请放入input_videos文件夹
    输出视频将保存在output_videos文件夹
    """
    # 加载配置
    config = Config()
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 获取input_videos文件夹中所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    input_video_paths = []
    
    for ext in video_extensions:
        input_video_paths.extend(glob.glob(os.path.join(config.input_dir, f'*{ext}')))
    
    if not input_video_paths:
        print(f"错误: 未在 {config.input_dir} 文件夹中找到任何视频文件")
        print("请将视频文件放入input_videos文件夹")
        return
    
    print(f"找到 {len(input_video_paths)} 个视频文件，开始批量处理...")
    
    # 加载YOLOv8模型
    model = YOLO(config.model_path)
    
    # 批量处理每个视频
    for idx, input_video_path in enumerate(input_video_paths, 1):
        print(f"\n处理第 {idx}/{len(input_video_paths)} 个视频: {os.path.basename(input_video_path)}")
        
        # 初始化视频处理器
        video_processor = VideoProcessor(input_video_path, config.output_dir)
        
        # 处理视频
        video_processor.process_video(model, lambda frame, coords: blur_region(frame, coords, config.blur_strength))
        print(f"视频处理完成！结果保存在: {video_processor.output_path}")

if __name__ == "__main__":
    main()