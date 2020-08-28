import cv2
import os


def frames_to_video(frames_dir, fps, size, start_index, end_index, save_path):  
    """
    [start_index, end_index+1)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #  'xxx.avi'
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)  # size 不匹配，将导致生成的文件损坏（无法打开）
    for i in range(start_index, end_index+1):
        if os.path.isfile("%s/%d.jpg"%(frames_dir, i)):
            frame = cv2.imread("%s/%d.jpg"%(frames_dir, i))
            print("%s/%d.jpg"%(frames_dir, i))
            print(frame.shape)
            videoWriter.write(frame)
    videoWriter.release()
    return

if __name__ == "__main__":
    frames_dir = 'outputs/head-detection-frames-part_a-0.1-outputs'
    frame_count = len(os.listdir(frames_dir))
    frame_tmp = os.path.join(frames_dir, os.listdir(frames_dir)[1])
    frame_size = cv2.imread(frame_tmp).shape[:2][::-1]  # (304, 640, 3) → (304, 640) → (640, 304)
    result_file = frames_dir + '.mp4'
    frames_to_video(frames_dir, fps=25, size=frame_size, start_index=1, end_index = frame_count, save_path=result_file)
    pass