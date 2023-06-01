from fastapi import FastAPI, File, UploadFile
from segmentation import get_yolov5, get_image_from_bytes
import uvicorn
import os
import io
import base64
import cv2
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
from multiprocessing import cpu_count, freeze_support

LOCAL_WORKSPACE_PATH = '.'

paths = {
    'PRED_PATH': LOCAL_WORKSPACE_PATH + '/preds'
}

model = get_yolov5()


app = FastAPI(
    title="JKFenner Oil Seal Spring Detection Tool",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="3.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@app.get("/api/v1.0/check-status")
async def root():
    return {"Alive": True}


@app.post("/jkfenner/spring/detect/image")
async def detect_object_json(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    results = model.predict(source=image)
    res_plotted = results[0].plot()
    im_array = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Reduce quality (preserve aspect ratio) - default starts at 95
    _, im_arr = cv2.imencode('.jpg', im_array, encode_param)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


@app.post("/jkfenner/spring/detect/json")
async def detect_object_json(file: UploadFile = File(...)):

    # Datetime For Filename
    now = datetime.now()
    now = str(now)
    now = now.replace('-', '').replace(' ', '_').replace(':', '').replace('.', '_')
    image = load_image_into_numpy_array(await file.read())
    results = model(image)
    res_plotted = results[0].plot()
    for r in results:
        detection_classes = r.boxes.cls.tolist()
    not_ok_count = detection_classes.count(0.0)
    ok_count = detection_classes.count(1.0)
    
    inspection_result = True

    if (ok_count) == 0 and (not_ok_count == 0):
        inspection_result = False     
    elif (ok_count > 0) and (not_ok_count == 0):
        inspection_result = True  
    elif (ok_count) == 0 and (not_ok_count > 0):
        inspection_result = False
    elif (ok_count > 0) and (not_ok_count > 0):
        inspection_result = False
    
            
    local_pred_im = Image.fromarray(res_plotted)
    local_pred_im.save(paths['PRED_PATH'] + '/' + f'{now}.jpg') 

    im_array = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Reduce quality (preserve aspect ratio) - default starts at 95
    _, im_arr = cv2.imencode('.jpg', im_array, encode_param)  # im_arr: image in Numpy one-dim array format.    
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    return {"inspection_result" : inspection_result,
            "not_ok_count": not_ok_count,
            "ok_count" : ok_count, 
            "b64": im_b64}