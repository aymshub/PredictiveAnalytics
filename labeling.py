import argparse
import json
import os
from pathlib import Path
import ffmpeg

COCO_CLASSES = {
    'person': 0,
    'car': 1,
}

def extract_video_frames(video_path: Path, output_path: Path, frame_rate: int):
    os.makedirs(output_path, exist_ok=True)
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(output_path / 'frame_%06d.jpg'),
                qscale=2,  # Adjust image quality as needed
                r=frame_rate  # Adjust frame rate as needed
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Error extracting frames: {e.stderr.decode()}")

def labelstudio_labels_to_yolo(labelstudio_labels_json_path: str, output_dir_path: str, frame_skip: int, index_video: int = 0) -> None:
    ls_project = json.load(open(labelstudio_labels_json_path))[index_video]

    frames_count = ls_project['annotations'][0]['result'][0]['value']['framesCount']
    yolo_labels = [[] for _ in range(frames_count - frame_skip)]
    for instance in ls_project['annotations'][0]['result']:
        class_id = COCO_CLASSES.get(instance['value']['labels'][0], -1)
        if class_id == -1:
            continue
        for i, keypoint in enumerate(instance['value']['sequence'][:-1]):
            start_point = keypoint
            end_point = instance['value']['sequence'][i + 1]
            start_frame = start_point['frame']
            end_frame = end_point['frame']

            n_frames_between = end_frame - start_frame
            delta_x = (end_point['x'] - start_point['x']) / n_frames_between
            delta_y = (end_point['y'] - start_point['y']) / n_frames_between
            delta_width = (end_point['width'] - start_point['width']) / n_frames_between
            delta_height = (end_point['height'] - start_point['height']) / n_frames_between

            x = start_point['x'] + start_point['width'] / 2
            y = start_point['y'] + start_point['height'] / 2
            width = start_point['width']
            height = start_point['height']
            for frame in range(start_frame, end_frame):
                if frame >= frame_skip:
                    yolo_labels[frame - frame_skip].append([class_id, x / 100, y / 100, width / 100, height / 100])
                x += delta_x + delta_width / 2
                y += delta_y + delta_height / 2
                width += delta_width
                height += delta_height

            epsilon = 1e-5
            assert abs(x - end_point['x'] - end_point['width'] / 2) <= epsilon, f'x does not match: {x} vs {end_point["x"] + end_point["width"] / 2}'
            assert abs(y - end_point['y'] - end_point['height'] / 2) <= epsilon, f'y does not match: {y} vs {end_point["y"] + end_point["height"] / 2}'
            assert abs(width - end_point['width']) <= epsilon, f'width does not match: {width} vs {end_point["width"]}'
            assert abs(height - end_point['height']) <= epsilon, f'height does not match: {height} vs {end_point["height"]}'

        last_keypoint = instance['value']['sequence'][-1]
        if last_keypoint['frame'] >= frame_skip and last_keypoint['frame'] < frames_count:
            yolo_labels[last_keypoint['frame'] - frame_skip].append([class_id, last_keypoint['x'] / 100, last_keypoint['y'] / 100, last_keypoint['width'] / 100, last_keypoint['height'] / 100])

    os.makedirs(output_dir_path, exist_ok=True)
    for frame, frame_labels in enumerate(yolo_labels):
        if frame % 100 == 0:
            print(f'Writing labels for frame {frame + 1}')
        padded_frame_number = str(frame + 1).zfill(6)
        file_path = Path(output_dir_path) / f'frame_{padded_frame_number}.txt'
        text = ''
        for label in frame_labels:
            text += ' '.join(map(str, label)) + '\n'
        with open(file_path, 'w') as f:
            f.write(text)

    print(f'Done. Wrote labels for {frames_count - frame_skip} frames.')

    for frame in range(frames_count - frame_skip, frames_count):
        padded_frame_number = str(frame + 1).zfill(6)
        frame_file = Path(output_dir_path) / f'frame_{padded_frame_number}.jpg'
        if frame_file.exists():
            os.remove(frame_file)
        label_file = Path(output_dir_path) / f'frame_{padded_frame_number}.txt'
        if label_file.exists():
            os.remove(label_file)
    print(f'Deleted (skipped) the {frame_skip} first labels and last frames.')

def split_train_val(output_path: Path, ratio: float = 0.8):
    os.makedirs(output_path / 'train' / 'images', exist_ok=True)
    os.makedirs(output_path / 'train' / 'labels', exist_ok=True)
    os.makedirs(output_path / 'val' / 'images', exist_ok=True)
    os.makedirs(output_path / 'val' / 'labels', exist_ok=True)

    img_files = sorted([f for f in os.listdir(output_path / 'frames') if f.endswith('.jpg')])
    lbl_files = sorted([f for f in os.listdir(output_path / 'frames') if f.endswith('.txt')])
    count_train = round(len(img_files) * ratio)

    for file in img_files[:count_train]:
        os.rename(output_path / 'frames' / file, output_path / 'train' / 'images' / file)
    for file in img_files[count_train:]:
        os.rename(output_path / 'frames' / file, output_path / 'val' / 'images' / file)
    for file in lbl_files[:count_train]:
        os.rename(output_path / 'frames' / file, output_path / 'train' / 'labels' / file)
    for file in lbl_files[count_train:]:
        os.rename(output_path / 'frames' / file, output_path / 'val' / 'labels' / file)
    os.rmdir(output_path / 'frames')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label-studio-json-export', help='Path to label-studio project export in JSON format', required=True)
    parser.add_argument('-v', '--video-file', help='Path to video file associated with project export', required=True)
    parser.add_argument('-o', '--output-folder', help='Path to output folder', required=True)
    parser.add_argument('-s', '--skip-frames', type=int, help='Number of frames to skip', required=True)
    parser.add_argument('-r', '--frame-rate', type=int, help='Frame rate of the video', required=True)
    args = parser.parse_args()

    EXPORT_PATH = Path(args.label_studio_json_export)
    VIDEO_PATH = Path(args.video_file)
    OUTPUT_PATH = Path(args.output_folder)
    FRAME_SKIP = args.skip_frames
    FRAME_RATE = args.frame_rate

    extract_video_frames(VIDEO_PATH, OUTPUT_PATH / 'frames', FRAME_RATE)
    labelstudio_labels_to_yolo(EXPORT_PATH, OUTPUT_PATH / 'frames', FRAME_SKIP)
    split_train_val(OUTPUT_PATH)
