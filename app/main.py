# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routers import detection, trajectory, conflict_zone

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(detection.router, prefix="/detect", tags=["Detection"])
app.include_router(trajectory.router, prefix="/trajectory", tags=["Trajectory"])
app.include_router(conflict_zone.router, prefix="/conflict", tags=["ConflictZone"])

@app.get("/")
def index(request):
    return templates.TemplateResponse("index.html", {"request": request})

# app/routers/detection.py
from fastapi import APIRouter, UploadFile, File
import shutil
from app.services.detector import run_detection

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"app/static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    csv_path = run_detection(file_path)
    return {"tracking_csv": csv_path}

# app/routers/trajectory.py
from fastapi import APIRouter, Form
from app.services.kalman_lstm import predict_future as kl_predict
from app.services.sdt_att import predict_future as sa_predict

router = APIRouter()

@router.post("/predict")
def predict(frame_id: int = Form(...), track_id: int = Form(...),
            method: str = Form("sdtatt"), csv_path: str = Form(...)):
    if method == "kalman_lstm":
        out_video = kl_predict(csv_path, frame_id, track_id)
    else:
        out_video = sa_predict(csv_path, frame_id, track_id)
    return {"output_video": out_video}

# app/routers/conflict_zone.py
from fastapi import APIRouter, Form
from app.services.pet_conflict import compute_conflict_zones

router = APIRouter()

@router.post("/run")
def run_pet(csv_path: str = Form(...)):
    conflict_video = compute_conflict_zones(csv_path)
    return {"conflict_video": conflict_video}