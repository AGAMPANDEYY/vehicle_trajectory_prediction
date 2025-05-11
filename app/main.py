# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import shutil
from app.modules.video_processor import VideoProcessor
from app.modules.sdtatt import SDTATT_predict_all, SDTATT_predict_vehicle
from app.modules.pet import PETPipeline

app = FastAPI(title="Academic Traffic Research Platform")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
def process(
    request: Request,
    video: UploadFile = File(...),
    choice: str = Form("all"),
    frame_id: int = Form(None),
    vehicle_id: int = Form(None)
):
    # Save uploaded video
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Run VideoProcessor to generate tracking CSV
    processed_video = os.path.join(OUTPUT_DIR, f"processed_{video.filename}")
    vp = VideoProcessor(
        source_weights_path=os.getenv("YOLO_WEIGHTS"),
        source_video_path=video_path,
        target_video_path=processed_video,
        lstm_model_path=os.getenv("LSTM_MODEL")
    )
    vp.process_video()
    tracking_csv = os.path.join(os.path.dirname(processed_video), "combined_tracking_data.csv")

    # SDT-ATT prediction
    data_npy = os.getenv("SDTATT_DATA")
    checkpoint = os.getenv("SDTATT_CHECKPOINT")
    data_dir = os.path.join(OUTPUT_DIR, video.filename)
    os.makedirs(data_dir, exist_ok=True)
    if choice == "all":
        future_csv = os.path.join(data_dir, "future_all.csv")
        SDTATT_predict_all(data_npy, tracking_csv, checkpoint, future_csv)
    else:
        if frame_id is None or vehicle_id is None:
            return HTMLResponse("<h3>Error: frame_id and vehicle_id required for single.</h3>", status_code=400)
        pred, _, _ = SDTATT_predict_vehicle(frame_id, vehicle_id, data_npy, checkpoint)
        # save single CSV similarly...
        future_csv = os.path.join(data_dir, "future_single.csv")
        # ... conversion code omitted for brevity

    # Prepare PET input
    pet_input = os.path.join(data_dir, "pet_input.csv")
    df = pd.read_csv(future_csv)
    # ... formatting steps
    df.to_csv(pet_input, index=False)

    # Run PET
    pet = PETPipeline(tracking_path=pet_input, zone_path=os.getenv("ZONE_CSV"), video_path=video_path)
    pet.run()
    pet_csv = os.path.join(data_dir, "pet_results.csv")
    pet.pet_df.to_csv(pet_csv, index=False)

    heatmap = pet.generate_heatmap()
    heatmap_path = os.path.join(data_dir, "heatmap.png")
    cv2.imwrite(heatmap_path, cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "processed_video": processed_video,
        "tracking_csv": tracking_csv,
        "future_csv": future_csv,
        "pet_csv": pet_csv,
        "heatmap": heatmap_path
    })
