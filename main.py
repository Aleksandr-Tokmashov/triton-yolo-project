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
    
    return img_chw

def create_batch(images: List[bytes]) -> np.ndarray:
    processed_images = []
    for img_bytes in images:
        img_processed = preprocess_image(img_bytes)
        processed_images.append(img_processed)
    
    batch = np.stack(processed_images, axis=0) 
    return batch

def postprocess_batch_yolo(output: np.ndarray, batch_size: int) -> List[List[dict]]:    
    batch_results = []
    for i in range(batch_size):
        detections = output[i] 
        image_results = []
        
        for det in detections:
            conf = float(det[4])
            if conf > 0.25:  
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                class_id = int(det[5])
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                
                image_results.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name
                })
        
        batch_results.append(image_results)
    
    return batch_results

@app.post("/infer_batch")
async def infer_batch(model_name: str, images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    try:
        images_bytes = []
        for img in images:
            contents = await img.read()
            images_bytes.append(contents)
        
        batch_size = len(images_bytes)
        
        batch_data = create_batch(images_bytes)
        
        print(f"Batch shape: {batch_data.shape}")
        
        inputs = [httpclient.InferInput("images", batch_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_data)
        
        if model_name == "yolo_det":
            outputs = [httpclient.InferRequestedOutput("output0")]
        elif model_name == "yolo_seg":
            outputs = [
                httpclient.InferRequestedOutput("output0"),
                httpclient.InferRequestedOutput("output1")
            ]
        else:
            raise HTTPException(status_code=400, detail="Unknown model name. Use 'yolo_det' or 'yolo_seg'")
        
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        if model_name == "yolo_det":
            output_data = response.as_numpy("output0")
            print(f"Output shape: {output_data.shape}")
            
            batch_results = postprocess_batch_yolo(output_data, batch_size)
            return {"results": batch_results, "batch_size": batch_size}
            
        elif model_name == "yolo_seg":
            output0 = response.as_numpy("output0")
            output1 = response.as_numpy("output1")
            
            print(f"Output0 shape: {output0.shape}")
            print(f"Output1 shape: {output1.shape}")
            
            batch_results = postprocess_batch_yolo(output0, batch_size)
            
            return {
                "results": batch_results,
                "masks_shapes": output1.shape[1:],
                "batch_size": batch_size
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer(model_name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_data = preprocess_image(contents)
        
        input_data = np.expand_dims(input_data, axis=0)
        
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        if model_name == "yolo_det":
            outputs = [httpclient.InferRequestedOutput("output0")]
        elif model_name == "yolo_seg":
            outputs = [
                httpclient.InferRequestedOutput("output0"),
                httpclient.InferRequestedOutput("output1")
            ]
        else:
            raise HTTPException(status_code=400, detail="Unknown model name. Use 'yolo_det' or 'yolo_seg'")
        
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        if model_name == "yolo_det":
            output_data = response.as_numpy("output0")[0]
            detections = []
            
            for det in output_data:
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
            
            return {"detections": detections}
            
        elif model_name == "yolo_seg":
            output0 = response.as_numpy("output0")[0]
            output1 = response.as_numpy("output1")[0]
            
            detections = []
            for det in output0:
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
            
            return {
                "detections": detections,
                "masks_shape": output1.shape
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