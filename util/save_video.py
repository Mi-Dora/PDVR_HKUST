import cv2


def save_video(list_file, save_path, fps=30,  greyscale=False):
    """
    Args:
        frame_list: (list of ndarray or ndarray) frame list
        save_path: video path to save, with video name and suffix
        fps: frame per second
        greyscale: whether save grey scale video
    """
    with open(list_file, 'r') as f:
        lines = f.readlines()
    img = cv2.imread(lines[0])
    if greyscale:
        H, W = img.shape
    else:
        H, W, _ = img.shape
    size = (int(W), int(H))
    encoder = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    if greyscale:
        writer.open(save_path, encoder, fps=fps, frameSize=size, isColor=False)
    else:
        writer.open(save_path, encoder, fps=fps, frameSize=size, isColor=True)

    for line in lines:
        frame = cv2.imread(line)
        writer.write(frame)

    print(save_path + ' saved.')
    writer.release()




