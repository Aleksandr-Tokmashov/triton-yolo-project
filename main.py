from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from typing import List, Dict, Any
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

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def nms(detections: List[dict], iou_threshold: float = 0.45) -> List[dict]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [det for det in detections if calculate_iou(best['bbox'], det['bbox']) < iou_threshold]
    return keep

def extract_masks_from_yolo(seg_output0: np.ndarray, seg_output1: np.ndarray) -> List[List[dict]]:
    batch_results = []
    conf_threshold = 0.25
    mask_threshold = 0.5
    
    for batch_idx in range(seg_output0.shape[0]):
        image_masks = []
        proto_masks = seg_output1[batch_idx]
        
        for det_idx in range(seg_output0.shape[1]):
            det = seg_output0[batch_idx, det_idx]
            obj_conf = float(det[4])
            if obj_conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            class_id = int(det[5])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            mask_coeffs = det[6:38]
            
            mask = np.zeros((160, 160), dtype=np.float32)
            for i in range(32):
                coeff = mask_coeffs[i]
                if abs(coeff) > 0.1:
                    mask += coeff * proto_masks[i]
            
            mask = 1 / (1 + np.exp(-mask))
            binary_mask = (mask > mask_threshold).astype(np.uint8)
            mask_area = binary_mask.sum()
            
            if mask_area < 100:
                continue
            
            mask_quality = mask_area / (160 * 160)
            mask_conf = obj_conf * mask_quality
            
            image_masks.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(obj_conf),
                "mask_confidence": float(mask_conf),
                "class_id": class_id,
                "class_name": class_name,
                "mask": binary_mask,
                "mask_area": int(mask_area)
            })
        
        if image_masks:
            image_masks = nms(image_masks, iou_threshold=0.5)
        
        batch_results.append(image_masks)
    
    return batch_results

def process_detections(detections_output: np.ndarray) -> List[dict]:
    detections = []
    for det in detections_output[0]:
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
    return nms(detections, iou_threshold=0.45)

def get_model_outputs(model_name: str):
    if model_name == "yolo_det":
        return [httpclient.InferRequestedOutput("output0")]
    elif model_name == "yolo_seg":
        return [
            httpclient.InferRequestedOutput("output0"),
            httpclient.InferRequestedOutput("output1")
        ]
    elif model_name == "yolo_ensemble":
        return [
            httpclient.InferRequestedOutput("detections"),
            httpclient.InferRequestedOutput("segmentation_output0"),
            httpclient.InferRequestedOutput("segmentation_output1")
        ]
    else:
        raise HTTPException(status_code=400, detail="Unknown model name")

@app.post("/infer")
async def infer(model_name: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_data = preprocess_image(contents)
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = get_model_outputs(model_name)
        response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
        if model_name == "yolo_det":
            detections_output = response.as_numpy("output0")
            detections = process_detections(detections_output)
            return {"detections": detections, "num_detections": len(detections), "model": model_name}
        
        elif model_name == "yolo_seg":
            seg_output0 = response.as_numpy("output0")
            seg_output1 = response.as_numpy("output1")
            masks_results = extract_masks_from_yolo(seg_output0, seg_output1)
            
            detections_with_masks = []
            for mask_info in masks_results[0]:
                detections_with_masks.append({
                    "bbox": mask_info["bbox"],
                    "confidence": mask_info["confidence"],
                    "mask_confidence": mask_info["mask_confidence"],
                    "class_id": mask_info["class_id"],
                    "class_name": mask_info["class_name"],
                    "mask_area": mask_info["mask_area"],
                    "has_mask": True
                })
            
            return {"detections_with_masks": detections_with_masks, "num_masks_processed": len(detections_with_masks), "model": model_name}
        
        elif model_name == "yolo_ensemble":
            detections_output = response.as_numpy("detections")
            seg_output0 = response.as_numpy("segmentation_output0")
            seg_output1 = response.as_numpy("segmentation_output1")
            
            detections = process_detections(detections_output)
            masks_results = extract_masks_from_yolo(seg_output0, seg_output1)
            
            detections_with_masks = []
            for mask_info in masks_results[0]:
                detections_with_masks.append({
                    "bbox": mask_info["bbox"],
                    "confidence": mask_info["confidence"],
                    "mask_confidence": mask_info["mask_confidence"],
                    "class_id": mask_info["class_id"],
                    "class_name": mask_info["class_name"],
                    "mask_area": mask_info["mask_area"],
                    "has_mask": True
                })
            
            return {
                "detections_with_masks": detections_with_masks,
                "detections_without_masks": detections,
                "num_masks_processed": len(detections_with_masks),
                "num_detections_processed": len(detections),
                "model": model_name
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        processed_images = []
        
        for img_bytes in images_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_np = np.array(image)
            img_resized = cv2.resize(image_np, (640, 640))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            processed_images.append(img_chw)
        
        batch_data = np.stack(processed_images, axis=0)
        inputs = [httpclient.InferInput("images", batch_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_data)
        
        outputs = get_model_outputs(model_name)
        response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
        batch_results = []
        
        if model_name == "yolo_det":
            detections_output = response.as_numpy("output0")
            for i in range(batch_size):
                detections = []
                if i < detections_output.shape[0]:
                    for det in detections_output[i]:
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
                detections = nms(detections, iou_threshold=0.45)
                batch_results.append({
                    "image_index": i,
                    "detections": detections,
                    "num_detections": len(detections)
                })
        
        elif model_name == "yolo_seg":
            seg_output0 = response.as_numpy("output0")
            seg_output1 = response.as_numpy("output1")
            masks_results = extract_masks_from_yolo(seg_output0, seg_output1)
            
            for i in range(batch_size):
                detections_with_masks = []
                if i < len(masks_results) and masks_results[i]:
                    for mask_info in masks_results[i]:
                        detections_with_masks.append({
                            "bbox": mask_info["bbox"],
                            "confidence": mask_info["confidence"],
                            "mask_confidence": mask_info["mask_confidence"],
                            "class_id": mask_info["class_id"],
                            "class_name": mask_info["class_name"],
                            "mask_area": mask_info["mask_area"],
                            "has_mask": True
                        })
                
                batch_results.append({
                    "image_index": i,
                    "detections_with_masks": detections_with_masks,
                    "num_masks_processed": len(detections_with_masks)
                })
        
        elif model_name == "yolo_ensemble":
            detections_output = response.as_numpy("detections")
            seg_output0 = response.as_numpy("segmentation_output0")
            seg_output1 = response.as_numpy("segmentation_output1")
            masks_results = extract_masks_from_yolo(seg_output0, seg_output1)
            
            for i in range(batch_size):
                detections = []
                if i < detections_output.shape[0]:
                    for det in detections_output[i]:
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
                detections = nms(detections, iou_threshold=0.45)
                
                detections_with_masks = []
                if i < len(masks_results) and masks_results[i]:
                    for mask_info in masks_results[i]:
                        detections_with_masks.append({
                            "bbox": mask_info["bbox"],
                            "confidence": mask_info["confidence"],
                            "mask_confidence": mask_info["mask_confidence"],
                            "class_id": mask_info["class_id"],
                            "class_name": mask_info["class_name"],
                            "mask_area": mask_info["mask_area"],
                            "has_mask": True
                        })
                
                batch_results.append({
                    "image_index": i,
                    "detections_with_masks": detections_with_masks,
                    "detections_without_masks": detections,
                    "num_masks_processed": len(detections_with_masks),
                    "num_detections_processed": len(detections)
                })
        
        return {"batch_size": batch_size, "results": batch_results, "model": model_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        original_h, original_w = image_np.shape[:2]
        
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
        
        seg_output0 = response.as_numpy("segmentation_output0")
        seg_output1 = response.as_numpy("segmentation_output1")
        
        masks_results = extract_masks_from_yolo(seg_output0, seg_output1)
        vis_image = image_np.copy()
        
        colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 0, 255),
            5: (255, 255, 0),
            7: (255, 0, 255)
        }
        
        for mask_info in masks_results[0]:
            class_id = mask_info["class_id"]
            color = colors.get(class_id, (128, 128, 128))
            mask = mask_info["mask"]
            mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h))
            
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x1, y1, w, h = cv2.boundingRect(largest_contour)
                x2, y2 = x1 + w, y1 + h
            else:
                x1, y1, x2, y2 = [int(coord) for coord in mask_info["bbox"]]
            
            colored_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
            colored_mask[mask_resized > 0] = color
            vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask, 0.3, 0)
            
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            vis_image = cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0)
            
            label = f"{mask_info['class_name']} {mask_info['confidence']:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        _, jpeg_bytes = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(content=jpeg_bytes.tobytes(), media_type="image/jpeg")
        
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