import cv2

def blur_region(frame, coordinates, blur_strength=25):
    x1, y1, x2, y2 = coordinates
    # 确保坐标在有效范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    # 提取区域并应用高斯模糊
    region = frame[y1:y2, x1:x2]
    blurred_region = cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)
    frame[y1:y2, x1:x2] = blurred_region
    return frame