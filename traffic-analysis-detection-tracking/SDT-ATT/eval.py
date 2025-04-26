#python traffic-analysis-detection-tracking/SDT-ATT/eval.py

####3###

import torch
from torch.utils.data import DataLoader
from model import SDTATTModel
from dataloader import SDTATTDataset 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 
import os 

# ---- Config ----
BATCH_SIZE = 1
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_NEIGHBORS = 3  # Change according to your dataset Curreent numpy from datasets.py has 5 neighbors
FUTURE_LEN = 300    #30 frames is only 1sec and let us try for 5sec so we can set it to 150 frames

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_DATA_PATH = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")  # update if needed

NUM_SAMPLES_TO_PREDICT=5


# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Dataset ----
dataset = SDTATTDataset(N_DATA_PATH)

print("Dataset details:")
print(f"Number of samples: {len(dataset)}")
print(f"Sample keys: {dataset[0].keys()}")  # Check keys in the first sample
print(f"Sample shape: {dataset[0]['tv_hist'].shape}, {dataset[0]['nv_sp'].shape}, {dataset[0]['nv_dp'].shape}")


#########################To predict a specific sample, set frame_id and track_id#########################
frame_id = 238 # Set to None if you want to search by track_id only
track_id = 8
  # Set to None if you want to search by frame_id only


"""

V ID: 293, Center Frame: 978.0, Neighbors: [219 248 234 298 294]
TV ID: 293, Center Frame: 979.0, Neighbors: [219 248 298 294]
TV ID: 293, Center Frame: 980.0, Neighbors: [291 219 304 248 234]
TV ID: 293, Center Frame: 981.0, Neighbors: [291 219 304 234 298]
TV ID: 293, Center Frame: 987.0, Neighbors: [291 304 298]
TV ID: 293, Center Frame: 988.0, Neighbors: [291 289 219 304 248]
TV ID: 293, Center Frame: 989.0, Neighbors: [289 219 304 248 234]
TV ID: 293, Center Frame: 990.0, Neighbors: [289 219 304 234 298]
TV ID: 293, Center Frame: 991.0, Neighbors: [291 289 304 248 234]
TV ID: 293, Center Frame: 992.0, Neighbors: [291 289 219 304 248]
TV ID: 293, Center Frame: 993.0, Neighbors: [291 289 248 234 298]
TV ID: 293, Center Frame: 999.0, Neighbors: [289 304 248 234 294]
TV ID: 293, Center Frame: 1000.0, Neighbors: [304 248 234 294]
TV ID: 293, Center Frame: 1001.0, Neighbors: [304 234 294]

"""

print(f"Predicting for frame_id: {frame_id}, track_id: {track_id}")

sample = dataset.get_sample_by_frame_and_track(frame_id, track_id)


def SDTATT_predict():

    # ---- Prepare Model ----
    model = SDTATTModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_neighbors=NUM_NEIGHBORS,
        future_len=FUTURE_LEN
    ).to(device)

    model.eval()

    # ---- Move data to device ----
    tv_hist = sample['tv_hist'].unsqueeze(0).to(device)  # shape: [1, T, 2]
    nv_sp = sample['nv_sp'].unsqueeze(0).to(device)      # shape: [1, N, T, 2]
    nv_dp = sample['nv_dp'].unsqueeze(0).to(device)      # shape: [1, N, T, 2]

    # ---- Inference ----
    with torch.no_grad():
        output = model(tv_hist, nv_sp, nv_dp)  # shape: [1, future_len, 2]

    # Assume you have access to the last known position (e.g., from tv_hist)
    last_pos = tv_hist[0, -1]  # Shape: [2], last frame of history
    pred_rel = output[0]       # Shape: [30, 2] from model

    #
    print("Past vehcile Trajectory:")
    print(tv_hist.cpu().numpy())  # [T, 2]

    # Convert relative to absolute
    pred_abs = torch.cumsum(pred_rel, dim=0) + last_pos  # [30, 2]
    print("Absolute Predicted Future Trajectory Shape:")
    print(pred_abs.cpu().numpy())
    return pred_abs, sample['center_frame'], sample['tv_id']

## Output predicted would be of a particular fram_id and track_id meaning the vehicle so that we can visualize it on the video

pred_abs, center_frame, track_id = SDTATT_predict()


# ---- Load Video and Get Properties ----
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
VIDEO_PATH = os.path.join(PARENT_DIR, "data", "Lane_C_Video.mp4")  # update if needed

# ---- Load Video and Get Properties ----
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video resolution: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")
cap.release()

#------ Trained YOLO Object detection and tracking---------
# Process frame with YOLO
import supervision as sv
from ultralytics import YOLO
source_weights_path= r"D:\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\best.pt"
tracker= sv.ByteTrack()
conf_threshold = 0.2
iou_threshold = 0.5
model= YOLO(source_weights_path)


# ---- Draw prediction on the first frame ----
video = cv2.VideoCapture(VIDEO_PATH)
ret, frame = video.read()

print("Image shape:", frame.shape)  # Should be (H, W, 3)
#print("Prediction coordinates (first 5):", pred_abs.numpy()[:5])

if not ret:
    print("Error: Could not read the first frame.")
    exit()

for (x, y) in pred_abs.numpy().astype(int):
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)


display_frame = cv2.resize(frame, (1024, 540)) 
cv2.imshow(f"Prediction on Sample Frame No", display_frame)
cv2.waitKey(2)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
video.release() 

# ---- Write Output Video with Prediction Overlay ---
video = cv2.VideoCapture(VIDEO_PATH)

start_frame = int(center_frame) + 1 # Start overlaying right after the center frame
end_frame = start_frame + FUTURE_LEN
frame_idx = 0

# Output video writer - write only FUTURE_LEN frames
out = cv2.VideoWriter(
    r"D:\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\output_SDTATT1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)
START_BUFFER= 300 #frames
END_BUFFER= 300 #frames
print(f"Saving frames from {start_frame} to {end_frame}...")

with tqdm(total=START_BUFFER+FUTURE_LEN+END_BUFFER, desc="Processing Future Trajectory Frames", unit="frame") as pbar:
    video = cv2.VideoCapture(VIDEO_PATH)

    predicted_started = False
predicted_maneuver = "Lane Keep"  # Placeholder — use actual if available

print("neighbor vehicle ids", sample.get("nv_ids"))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    if start_frame - START_BUFFER <= frame_idx < end_frame + END_BUFFER:
        results = model(frame, verbose=False, conf=conf_threshold, iou=iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            det_id = int(detections.tracker_id[i])
     
            # Label Target Vehicle
            if det_id == track_id:
                label = "TV"
                color = (0, 255, 0)  # Green
            # Label Neighbor Vehicles (check if in sample nv_ids)
           
            elif det_id in sample.get("nv_ids", []):
                label = "NV"
                color = (0, 165, 255)  # Orange
            else:
                label = f"ID:{det_id}"
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        idx = frame_idx - start_frame

        if idx >= 0:
            predicted_started = True

        # ---- Prediction visualization ----
        if 0 <= idx < len(pred_abs):
            x, y = pred_abs[int(idx)].numpy().astype(int)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

            for i in range(1, idx + 1):
                pt1 = tuple(pred_abs[i - 1].numpy().astype(int))
                pt2 = tuple(pred_abs[i].numpy().astype(int))
                cv2.line(frame, pt1, pt2, color=(0, 200, 0), thickness=2)

        # ---- Draw TV history as bounding boxes ----
        tv_hist_np = sample['tv_hist'].cpu().numpy().astype(int)
        for pt in tv_hist_np:
            cv2.circle(frame, tuple(pt), 2, (255, 0, 0), -1)  # small blue dot

        # ---- Draw NV history ----
        nv_hist_np = sample['nv_sp'].cpu().numpy().astype(int)
        for n in range(nv_hist_np.shape[0]):
            for pt in nv_hist_np[n]:
                cv2.circle(frame, tuple(pt), 2, (0, 140, 255), -1)  # dark orange dot

        # ---- Display Model Info & Maneuver ----
        if predicted_started:
            cv2.putText(frame, "SDT-ATT Prediction", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.putText(frame, f"Predicted Maneuver: {predicted_maneuver}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        out.write(frame)
        pbar.update(1)

    elif frame_idx >= end_frame + END_BUFFER:
        break

    frame_idx += 1

    """
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if start_frame -START_BUFFER <= frame_idx < end_frame+END_BUFFER:
            results = model(
                frame, verbose=False, conf=conf_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            # Draw bounding boxes and track IDs
            for i in range(len(detections)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                track_id = detections.tracker_id[i]
                label = f"ID: {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            idx = frame_idx - start_frame

            if 0 <= idx < len(pred_abs):
                # Draw current prediction point
                x, y = pred_abs[int(idx)].numpy().astype(int)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

                # Draw trajectory line up to this point
                for i in range(1, idx + 1):
                    pt1 = tuple(pred_abs[i - 1].numpy().astype(int))
                    pt2 = tuple(pred_abs[i].numpy().astype(int))
                    cv2.line(frame, pt1, pt2, color=(0, 200, 0), thickness=2)

            # Draw target vehicle's past trajectory (blue)
            tv_hist_np = sample['tv_hist'].cpu().numpy().astype(int)
            for i in range(1, len(tv_hist_np)):
                pt1 = tuple(tv_hist_np[i - 1])
                pt2 = tuple(tv_hist_np[i])
                cv2.line(frame, pt1, pt2, color=(255, 0, 0), thickness=2)

            # Draw neighboring vehicles' past trajectories (orange)
            nv_hist_np = sample['nv_sp'].cpu().numpy().astype(int)  # Shape: [N, T, 2]
            num_neighbors = nv_hist_np.shape[0]
            for n in range(num_neighbors):
                for t in range(1, nv_hist_np.shape[1]):
                    pt1 = tuple(nv_hist_np[n, t - 1])
                    pt2 = tuple(nv_hist_np[n, t])
                    cv2.line(frame, pt1, pt2, color=(0, 165, 255), thickness=2)  # orange

            for n in range(num_neighbors):
                end_pt = tuple(nv_hist_np[n, -1])
                if 0 <= end_pt[0] < width and 0 <= end_pt[1] < height:
                    cv2.circle(frame, end_pt, radius=3, color=(0, 140, 255), thickness=-1)

            out.write(frame)
            pbar.update(1)

        elif frame_idx >= end_frame+300:
            break  # No need to continue

        frame_idx += 1
      """
video.release()
out.release()
print("Short trajectory video saved!")

