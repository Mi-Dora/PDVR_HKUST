import cv2


def save_video(frame_list, save_path, fps=30,  greyscale=False):
    """
    Args:
        frame_list: (list of ndarray or ndarray) frame list
        save_path: video path to save, with video name and suffix
        fps: frame per second
        greyscale: whether save grey scale video
    """
    if greyscale:
        H, W = frame_list[0].shape
    else:
        H, W, _ = frame_list[0].shape
    size = (int(W), int(H))
    encoder = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    if greyscale:
        writer.open(save_path, encoder, fps=fps, frameSize=size, isColor=False)
    else:
        writer.open(save_path, encoder, fps=fps, frameSize=size, isColor=True)

    for frame in frame_list:
        writer.write(frame)

    print(save_path + ' saved.')
    writer.release()




