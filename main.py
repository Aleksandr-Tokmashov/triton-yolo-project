from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import numpy as np
import tritonclient.http as httpclient
import cv2
from PIL import Image
import io

app = FastAPI()

triton_client = httpclient.InferenceServerClient(url="triton:8000")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    
    img_resized = cv2.resize(image_np, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch

def postprocess_detections(output: np.ndarray) -> List[dict]:
    detections = output[0] 
    
    results = []
    for det in detections:
        conf = float(det[4])
        if conf > 0.25:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            class_id = int(det[5])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class_id": class_id,
                "class_name": class_name
            })
    
    return results

@app.post("/infer_ensemble")
async def infer_ensemble(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        input_data = preprocess_image(contents)
        
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [
            httpclient.InferRequestedOutput("detections"),
            httpclient.InferRequestedOutput("segmentation_output0"),
            httpclient.InferRequestedOutput("segmentation_output1")
        ]
        
        response = triton_client.infer(
            model_name="yolo_ensemble",
            inputs=inputs,
            outputs=outputs
        )
        
        detections_output = response.as_numpy("detections")
        seg_output0 = response.as_numpy("segmentation_output0")
        seg_output1 = response.as_numpy("segmentation_output1")
        
        detections = postprocess_detections(detections_output)
        
        return {
            "detections": detections,
            "segmentation": {
                "output0_shape": seg_output0.shape,
                "output1_shape": seg_output1.shape
            },
            "model": "yolo_ensemble"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer_ensemble_batch")
async def infer_ensemble_batch(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    try:
        images_bytes = []
        for img in images:
            contents = await img.read()
            images_bytes.append(contents)
        
        batch_size = len(images_bytes)
        
        processed_images = []
        for img_bytes in images_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_np = np.array(image)
            
            img_resized = cv2.resize(image_np, (640, 640))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            processed_images.append(img_chw)
        
        batch_data = np.stack(processed_images, axis=0)
        
        print(f"Batch shape sent to ensemble: {batch_data.shape}")
        
        inputs = [httpclient.InferInput("images", batch_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_data)
        
        outputs = [
            httpclient.InferRequestedOutput("detections"),
            httpclient.InferRequestedOutput("segmentation_output0"),
            httpclient.InferRequestedOutput("segmentation_output1")
        ]
        
        response = triton_client.infer(
            model_name="yolo_ensemble",
            inputs=inputs,
            outputs=outputs
        )
        
        detections_output = response.as_numpy("detections")
        seg_output0 = response.as_numpy("segmentation_output0")
        seg_output1 = response.as_numpy("segmentation_output1")
        
        print(f"Ensemble detections output shape: {detections_output.shape}")
        print(f"Ensemble seg_output0 shape: {seg_output0.shape}")
        print(f"Ensemble seg_output1 shape: {seg_output1.shape}")
        
        batch_results = []
        for i in range(batch_size):
            detections = []
            dets = detections_output[i]
            
            for det in dets:
                conf = float(det[4])
                if conf > 0.25:
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    class_id = int(det[5])
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name
                    })
            
            batch_results.append({
                "image_index": i,
                "detections": detections,
                "segmentation_shapes": {
                    "output0": seg_output0[i].shape if i < seg_output0.shape[0] else None,
                    "output1": seg_output1[i].shape if i < seg_output1.shape[0] else None
                }
            })
        
        return {
            "results": batch_results,
            "batch_size": batch_size,
            "model": "yolo_ensemble",
            "output_shapes": {
                "detections": detections_output.shape,
                "segmentation_output0": seg_output0.shape,
                "segmentation_output1": seg_output1.shape
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        if triton_client.is_server_live() and triton_client.is_server_ready():
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy"}
    except:
        return {"status": "unhealthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)