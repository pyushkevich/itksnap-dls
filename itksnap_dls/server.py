from typing import Union, Callable, List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, Response, HTTPException, Form
from fastapi.routing import APIRoute
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from .session import session_manager
from .segment import SegmentSession
import io
import SimpleITK as sitk
import torch
import tempfile
import os
import logging
import base64
import numpy as np
import gzip
import time
import json


class ValidationErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                form = await request.form()
                print(f"Headers: {dict(request.headers)}")
                print(f"Query Params: {dict(request.query_params)}")
                print(f"Form: {form}")
                return await original_route_handler(request)
            except RequestValidationError as exc:
                body = await request.body()
                detail = {"errors": exc.errors(), "body": body.decode()}
                raise HTTPException(status_code=422, detail=detail)

        return custom_route_handler

app = FastAPI()
# app.router.route_class = ValidationErrorLoggingRoute

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/start_session")
def start_session():
    
    # Create a segmentation session    
    seg = SegmentSession('/data/pauly2/tk/nninter/test/mymodels/nnInteractive_v1.0')
    
    # Associate it with the id
    session_id = session_manager.create_session(seg)
    
    return {"session_id": session_id}


@app.post("/upload_nifti/{session_id}")
async def upload_nifti(session_id: str, filename:str, myfile: UploadFile = File(...)):
    
    # Get the current segmentator session
    seg = session_manager.get_session(session_id)
    if seg is None:
       return {"error": "Invalid session"}

    # Read file into memory
    contents = await myfile.read()
    
    # Write to a temporary location
    with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as temp_file:
        fn_temp = temp_file.name
        temp_file.write(contents)
        temp_file.flush()
        temp_file.close()
        
        # Load NIFTI using SimpleITK
        sitk_image = sitk.ReadImage(fn_temp)
        seg.set_image(sitk_image)

        # Store in session
        return {"message": "NIFTI file uploaded and stored in GPU memory"}        


def read_sitk_image(contents, metadata):
    print(f'arr_gz size: {len(contents)}, first byte: {contents[0]:d}, second byte: {contents[1]:d}')
    contents_raw = gzip.decompress(contents)
    array = np.frombuffer(contents_raw, dtype=np.float32)

    # Reshape the array
    print(f'Metadata: {metadata}')
    metadata_dict = json.loads(metadata)
    dim = metadata_dict['dimensions'][::-1]
    array = array.reshape(dim)
    
    # Load the NIFTI image
    sitk_image = sitk.GetImageFromArray(array, isVector=False)
    print(f'Received image of shape {sitk_image.GetSize()}')
    
    return sitk_image
    
    
@app.post("/upload_raw/{session_id}")
async def upload_raw(session_id: str, file: UploadFile = File(...), metadata: str = Form(...)):
    
    # Get the current segmentator session
    seg = session_manager.get_session(session_id)
    if seg is None:
       return {"error": "Invalid session"}

    # Read file into memory
    contents_gzipped = await file.read()
    sitk_image = read_sitk_image(contents_gzipped, metadata)
    
    # Load NIFTI using SimpleITK
    seg.set_image(sitk_image)

    # Store in session
    return {"message": "NIFTI file uploaded and stored in GPU memory"}        


@app.get("/process_point_interaction/{session_id}")
def handle_point_interaction(session_id: str, x: int, y: int, z: int, foreground: bool = False):
    
    # Get the current segmentator session
    seg = session_manager.get_session(session_id)
    if seg is None:
       return {"error": "Invalid session"}
   
    # Handle the interaction
    t0 = time.perf_counter()
    seg.add_point_interaction([x,y,z], include_interaction=foreground)
    t1 = time.perf_counter()
    
    # Base64 encode the segmentation result
    arr = np.where(sitk.GetArrayFromImage(seg.get_result()) > 0, 1, 0).astype(np.int8)
    arr_gz = gzip.compress(arr.tobytes())
    print(f'arr_gz size: {len(arr_gz)}, first byte: {arr_gz[0]:d}, second byte: {arr_gz[1]:d}')
    arr_b64 = base64.b64encode(arr_gz) #.decode("utf-8")
    t2 = time.perf_counter()
    
    print(f'handle_point_interaction timing:')
    print(f'  t[nnInteractive] = {t1-t0:.6f}')
    print(f'  t[encode] = {t2-t1:.6f}')
    return { "status": "success", "result": arr_b64 }
    

@app.post("/process_scribble_interaction/{session_id}")
async def handle_scribble_interaction(session_id: str, 
                                      file: UploadFile = File(...), 
                                      metadata: str = Form(...), 
                                      foreground: bool = False):
    
    # Get the current segmentator session
    seg = session_manager.get_session(session_id)
    if seg is None:
       return {"error": "Invalid session"}
   
    # Read squiggle image into memory
    contents_gzipped = await file.read()
    sitk_image = read_sitk_image(contents_gzipped, metadata)

    # Handle the interaction
    t0 = time.perf_counter()
    seg.add_scribble_interaction(sitk_image, include_interaction=foreground)
    t1 = time.perf_counter()
    
    # Base64 encode the segmentation result
    arr = np.where(sitk.GetArrayFromImage(seg.get_result()) > 0, 1, 0).astype(np.int8)
    arr_gz = gzip.compress(arr.tobytes())
    print(f'arr_gz size: {len(arr_gz)}, first byte: {arr_gz[0]:d}, second byte: {arr_gz[1]:d}')
    arr_b64 = base64.b64encode(arr_gz) #.decode("utf-8")
    t2 = time.perf_counter()
    
    print(f'handle_scribble_interaction timing:')
    print(f'  t[nnInteractive] = {t1-t0:.6f}')
    print(f'  t[encode] = {t2-t1:.6f}')
    return { "status": "success", "result": arr_b64 }
    
    
    
@app.get("/reset_interactions/{session_id}")
def handle_reset_interactions(session_id: str):
    
    # Get the current segmentator session
    seg = session_manager.get_session(session_id)
    if seg is None:
       return {"error": "Invalid session"}
   
    # Handle the interaction
    seg.reset_interactions()
    
    # Base64 encode the segmentation result
    return { "status": "success" }
    
    
@app.get("/end_session/{session_id}")
def end_session(session_id: str):
    success = session_manager.delete_session(session_id)
    return {"message": "Session ended" if success else "Invalid session"}