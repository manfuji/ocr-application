from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import HTMLResponse, JSONResponse
import easyocr
import cv2
import numpy as np
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.handlers.ocr_handler import *
app = FastAPI()

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Mount the "static" folder to serve static files like CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform OCR on the image
        extracted_text = perform_ocr(image)

        return templates.TemplateResponse("result.html", {"request": request, "extracted_text": extracted_text})
    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "error": str(e)})


@app.post("/api/uploadfile/")
async def create_api_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform OCR on the image
        extracted_text = perform_ocr(image)

        return JSONResponse(content={"extracted_text": extracted_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Add a route to serve static files like CSS


@app.get("/static/{filename}", response_class=HTMLResponse)
async def serve_static(request: Request, filename: str):
    return templates.TemplateResponse("static/" + filename, {"request": request})
