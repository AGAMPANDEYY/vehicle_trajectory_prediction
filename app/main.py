from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
import pandas as pd
import cv2
import shutil
import uuid

from app.modules.video_processor import VideoProcessor
from app.modules.sdtatt import SDTATT_predict_all, SDTATT_predict_vehicle, overlay_sdtatt_prediction
from app.modules.pet import PETPipeline


load_dotenv()

app = FastAPI(title="Academic Traffic Research Platform")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Templates & Static
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
# after your existing mounts
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# In-memory task storage
task_status = {}
task_params = {}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    video: UploadFile = File(None),             # now optional
    tracking_csv: UploadFile = File(None),      # new optional
    choice: str = Form("all"),
    frame_id: int = Form(None),
    vehicle_id: int = Form(None),
):
    task_id = str(uuid.uuid4())
    params = {"choice": choice, "frame_id": frame_id, "vehicle_id": vehicle_id}

    # save video only if provided
    if video:
        vid_name = f"input_{task_id}.mp4"
        vid_path = os.path.join(UPLOAD_DIR, vid_name)
        with open(vid_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        params["video_path"] = vid_path
        input_url = f"/uploads/{vid_name}"
    else:
        params["video_path"] = None
        input_url = None

    # save tracking CSV only if provided
    if tracking_csv:
        csv_name = f"tracking_{task_id}.csv"
        csv_path = os.path.join(UPLOAD_DIR, csv_name)
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(tracking_csv.file, f)
        params["tracking_csv_path"] = csv_path
    else:
        params["tracking_csv_path"] = None

    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Waiting to start",
        "input_video_url": input_url,
    }
    task_params[task_id] = params

    return templates.TemplateResponse(
        "processing.html",
        {"request": request, "task_id": task_id, "input_video_url": input_url}
    )

from fastapi import FastAPI, BackgroundTasks

def _do_work(task_id: str):
    try:
        print(f"--- starting work for task {task_id} ---")
        params = task_params[task_id]
        video_path = params["video_path"]
        csv_override = params["tracking_csv_path"]

        task_status[task_id].update({"status": "processing", "progress": 0})

        # 1) DETECTION & TRACKING (skip if CSV was uploaded)
        if csv_override is None:
            task_status[task_id].update({"message":"Detecting & Tracking","progress":20})
            processed_video = os.path.join(OUTPUT_DIR, f"processed_{task_id}.mp4")
            vp = VideoProcessor(
                source_weights_path=os.getenv("YOLO_WEIGHTS"),
                source_video_path=video_path,
                target_video_path=processed_video,
                lstm_model_path=os.getenv("LSTM_MODEL_CHKPT"),
            )
            vp.process_video()
            tracking_csv = os.path.join(OUTPUT_DIR, f"tracking_{task_id}.csv")
        else:
            task_status[task_id].update({"message":"Using uploaded CSV","progress":20})
            tracking_csv = csv_override
            processed_video = None

        # 2) SDT-ATT
        task_status[task_id].update({"message":"Predicting Trajectories","progress":50})
        future_csv = os.path.join(OUTPUT_DIR, f"future_{params['choice']}_{task_id}.csv")
        data_npy   = os.getenv("SDTATT_DATA")
        checkpoint = os.getenv("SDTATT_CHECKPOINT")
        if params["choice"]=="all":
            SDTATT_predict_all(data_npy, tracking_csv, checkpoint, future_csv)
        else:
            pred,center_frame,track_id = SDTATT_predict_vehicle(params["frame_id"], params["vehicle_id"])
            pd.DataFrame(pred).to_csv(future_csv, index=False)
            tracking_df= pd.read_csv(r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\combined_tracking_data (1).csv")
            VIDEO_PATH=r"C:\Agam\Work\vehicle_trajectory_prediction\app\uploads\Lane_C_Video.mp4"
            overlay_sdtatt_prediction(params["frame_id"], params["vehicle_id"], pred, center_frame, VIDEO_PATH, tracking_df)


        # 3) PET ANALYSIS
        task_status[task_id].update({"message":"Running PET","progress":75})
        tracking_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\combined_tracking_data (1).csv"
        # Changed tracking path to the detected and tracked csv and not SDTATT output since that gives error as of now
        zone_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\zones.csv"
        
        # 3) PET conflict‐zones video (skip if already exists)
        conflict_vid=r"C:\Agam\Work\vehicle_trajectory_prediction\app\outputs\pet.mp4"
        if not os.path.exists(conflict_vid):
            task_status[task_id].update({"message":"Generating Conflict Zones","progress":75})
            # reuse the same inputs you already saved
            pet = PETPipeline(tracking_path=tracking_path, zone_path=zone_path, video_path=video_path)
            pet.run()


        # 4) FINALIZE
        task_status[task_id].update({"message":"Finalizing","progress":90})
        #heatmap = pet.generate_heatmap()
        #cv2.imwrite(os.path.join(OUTPUT_DIR, f"heatmap_{task_id}.png"),
                #cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX).astype("uint8"))

        # Build final result dict   f"/outputs/{os.path.basename(overlay_vid)}"
        result = {
        "tracking_csv_url":   f"/outputs/combined_tracking_data (1).csv",
        "processed_video_url": f"/outputs/output_video.mp4",
        "future_csv_url":      f"/outputs/{os.path.basename(future_csv)}",
        "error_csv_url":       f"/outputs/trajectory_errors.csv",
        "overlay_sdtatt_url":  f"/outputs/output_SDTATT.mp4",
        "heatmap_url":         f"/outputs/pet_img.jpeg",
        "conflict_video_url":  f"/outputs/pet.mp4" 
        }
        task_status[task_id].update({
        "status":   "completed",
        "progress": 100,
        "message":  "Done",
        "result":   result
        })

    except Exception as e:
        task_status[task_id].update({
            "status": "error",
            "progress": 100,
            "message": f"Failed: {str(e)}"
        })

@app.get("/status/{task_id}")
def get_status(task_id: str):
    status = task_status.get(task_id)
    if not status:
        return JSONResponse({"error": "Unknown task"}, status_code=404)
    return status

@app.get("/result/{task_id}", response_class=HTMLResponse)
def result(request: Request, task_id: str):
    status = task_status.get(task_id, {})
    if status.get("status") != "completed":
        return HTMLResponse("<h3>Task not completed yet or failed.</h3>", status_code=400)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, **status["result"]}
    )

@app.post("/run/{task_id}")
async def run_task(task_id: str, background_tasks: BackgroundTasks):
    # 1) schedule the heavy work
    background_tasks.add_task(_do_work, task_id)
    # 2) immediately return so the browser can start polling
    return {"status":"started"}