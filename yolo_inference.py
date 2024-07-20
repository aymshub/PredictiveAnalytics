#!/usr/bin/env python

import argparse
import logging
import os
import signal
import cv2
from ultralytics import YOLO

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

running = True

task_suffixes = {
    'detect': '',
    'segment': '-seg',
    'classify': '-cls',
    'pose': '-pose',
}

COCO_CLASSES = {
    'brocoli': 0,
    'blank': 1,
    'person': 2,
    'bicycle': 3,
    'car': 4,
}

def model_name(size, task):
    suffix = task_suffixes[task]
    return f'yolov8{size}{suffix}.pt'

def do_inferencing(source, model_size, task, preview, class_list):
    global running
    # use USB camera
    if source.isdigit():
        source = int(source)
    # get video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    # get details from video source
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video source details: width={width}, height={height}, fps={fps}, frame_count={frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('output6.mp4', fourcc, fps, (width, height), isColor=True)

    logger.info("Start inferencing")

    # instantiate pretrained yolo model
    model = YOLO(model_name(model_size, task))
    logger.info(f"Loaded YOLO model: {model_name(model_size, task)}")

    # live preview window?
    if preview:
        try:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("output", 800, 600)
        except cv2.error as e:
            logger.error("Failed to create preview window: {}".format(e))
            preview = False

    while cap.isOpened() and running:
        if os.path.exists("stopfile"):
            logger.info("Stop file detected. Stopping...")
            running = False
            break
        
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read next frame")
            break
        
        prediction = model.predict(frame, verbose=False, tracker='bytetrack.yaml')
        res_plotted = prediction[0].plot()
        output.write(res_plotted)
        logger.info(f"Processed frame {cap.get(cv2.CAP_PROP_POS_FRAMES)} / {frame_count}")
        
        if preview:
            cv2.imshow('output', res_plotted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 

    # be nice to your operating system
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    logger.info("Stopped inferencing")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('videosource', help='any video source opencv understands, e.g. 0,1,... for usb cams, "rtsp://..." for RTSP streams, /path/video.mp4 for video file')
    arg_parser.add_argument('-m', '--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', help='the size of the model to use (nano, small, medium, large, xlarge); defaults to "nano"')
    arg_parser.add_argument('-t', '--task', choices=['detect', 'segment', 'classify', 'pose'], default='detect', help='the task to perform; defaults to "detect"')
    arg_parser.add_argument('-p', '--preview', default=False, action='store_true', help='whether to show a live preview window, reduces performance slightly. If the window is in focus, press "q" to exit.')
    args = arg_parser.parse_args()

    # shutdown script properly
    def signal_handler(signum, _):
        signame = signal.Signals(signum).name
        logger.warning(f'Received {signame}. Exiting...')
        global running
        running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)    

    do_inferencing(args.videosource, args.model_size, args.task, args.preview, COCO_CLASSES)
