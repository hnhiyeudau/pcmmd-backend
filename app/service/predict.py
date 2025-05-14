import io
import numpy as np
import cv2
import torch
from PIL import Image
import json
from ultralytics import YOLO
import base64
import time

def load_model(model_path):
    """Load the YOLOv8 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

def process_image(model, image_bytes):
    """
    Process the uploaded image using YOLOv8 model and return cell analysis results
    """
    try:
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Run inference
        start_time = time.time()
        results = model(image_np)
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        
        if len(results) > 0:
            result = results[0]
            for i, det in enumerate(result.boxes.data):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(float, det[:4])
                
                # Extract confidence
                conf = float(det[4])
                
                # Extract class
                cls = int(det[5])
                cls_name = result.names[cls]
                
                # Calculate cell morphological metrics
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                area = width * height
                
                # Add to detections
                detections.append({
                    "id": i,
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    },
                    "metrics": {
                        "area": round(area, 2),
                        "aspect_ratio": round(aspect_ratio, 2),
                        # Additional metrics that would be calculated from instance segmentation
                        "perimeter": None,  # Would require mask/contour
                        "circularity": None,
                        "solidity": None
                    }
                })
        
        # Create visualization image with bounding boxes
        img_with_boxes = image_np.copy()
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            confidence = det["confidence"]
            label = f"{det['class']}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert visualization to base64 for frontend display
        _, buffer = cv2.imencode('.png', img_with_boxes)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        return {
            "success": True,
            "inference_time": round(inference_time, 3),
            "cell_count": len(detections),
            "detections": detections,
            "visualization": img_base64
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
