from fastapi import FastAPI, File, UploadFile, HTTPException
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

def postprocess_yolo(output: np.ndarray) -> list:
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

@app.post("/infer")
async def infer(model_name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        input_data = preprocess_image(contents)
        
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
            output_data = response.as_numpy("output0")
            detections = postprocess_yolo(output_data)
            return {"detections": detections}
            
        elif model_name == "yolo_seg":
            output0 = response.as_numpy("output0")
            output1 = response.as_numpy("output1")
            detections = postprocess_yolo(output0)
            
            return {
                "detections": detections,
                "masks_shape": output1.shape[1:]  
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