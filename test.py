from glob import glob
import os
import cv2

def image_to_video():
    file = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/exps/indoor/neus/scene0050_00/pred_40_test10_compa3/rendering/mesh'  # 图片目录
    output = '/home/du/Proj/3Dv_Reconstruction/NeuRIS/exps/indoor/neus/scene0050_00/pred_40_test10_compa3/vedio.mp4'  # 生成视频路径
    num = os.listdir(file)  # 生成图片目录下以图片名字为内容的列表
    height = 480
    weight = 640
    fps = 30
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    file = sorted(glob(f'{file}/*.jpg'))
    for i in range(len(num)):

        frame = cv2.imread(file[i])
        videowriter.write(frame)

    videowriter.release()

image_to_video()
