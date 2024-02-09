from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from utils import LineCounter, LineCounterAnnotator, Point
from tqdm.notebook import tqdm
import numpy as np
import cv2
from ultralytics import YOLO

# File paths for YOLO models
MODEL_POSE_PATH = "yolov8n-pose.pt"
MODEL_OBJECT_PATH = "yolov8n.pt"

# Initialize YOLO models
model_pose = YOLO(MODEL_POSE_PATH)
model_object = YOLO(MODEL_OBJECT_PATH)

# Class names dictionary from the object detection model
CLASS_NAMES_DICT = model_object.model.names

# Class ID for the "ball"
CLASS_ID_BALL = [32]

SOURCE_VIDEO_PATH = "dataset/juggling3.mp4"
TARGET_VIDEO_PATH = "dataset/juggling3_result.mp4"

# Create VideoInfo instance from the source video
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Create a frame generator for the source video
frame_generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# Initialize BoxAnnotator and LineCounterAnnotator instances
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)
line_counter = LineCounter(start=Point(x=0, y=0), end=Point(x=0, y=0))
line_annotator = LineCounterAnnotator(thickness=3, text_thickness=3, text_scale=2)

# Open the target video file for writing
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # OpenCV window initialization
    window_name = "Juggle Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, video_info.width, video_info.height)

    # Loop over video frames
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        results_poses = model_pose.track(frame, persist=True)
        annotated_frame = results_poses[0].plot()
        keypoints = results_poses[0].keypoints.xy.int().cpu().tolist()
        bboxes = results_poses[0].boxes.xyxy.cpu().numpy()

        results_object = model_object.track(frame, persist=True, conf=0.1)
        tracker_ids = results_object[0].boxes.id.int().cpu().numpy() if results_object[0].boxes.id is not None else None
        detections = Detections(
            xyxy=results_object[0].boxes.xyxy.cpu().numpy(),
            confidence=results_object[0].boxes.conf.cpu().numpy(),
            class_id=results_object[0].boxes.cls.cpu().numpy().astype(int),
            tracker_id=tracker_ids
        )

        # Filter out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID_BALL for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # Update line
        line_counter.update_line(bboxes[0], keypoints[0])
        # Update line counter
        line_counter.update(detections=detections)

        # Print the in_count attribute
        print("Juggle Count:", line_counter.in_count)

        # Annotate and display frame
        labels = [
            f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, track_id
            in detections
        ]
        annotated_frame = box_annotator.annotate(frame=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

        # Ensure the frame has valid dimensions before displaying
        if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
            cv2.imshow(window_name, annotated_frame)
            sink.write_frame(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
