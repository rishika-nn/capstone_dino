"""
Object Detection Module using Grounding DINO
Performs open-vocabulary object detection for campus surveillance search system
"""

import torch
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Data structure for detected objects"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    cropped_image: Optional[Image.Image] = None

class GroundingDINODetector:
    """Object detector using Grounding DINO-Swin-T for open-vocabulary detection"""
    
    def __init__(self,
                 model_name: str = "IDEA-Research/grounding-dino-base",
                 confidence_threshold: float = 0.35,
                 use_gpu: bool = True,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.25):
        """
        Initialize Grounding DINO object detector
        
        Args:
            model_name: Hugging Face model identifier
            confidence_threshold: Minimum confidence for detections
            use_gpu: Whether to use GPU if available
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Enhanced detection categories for classroom and campus surveillance
        self.default_prompts = [
            # Bags and backpacks with detailed variations
            "backpack", "school backpack", "laptop backpack", "book bag", 
            "shoulder bag", "messenger bag", "tote bag", "gym bag",
            
            # Technology items with specific variations
            "laptop", "laptop computer", "notebook computer", "chromebook",
            "laptop charger", "power adapter", "laptop sleeve", "laptop stand",
            "tablet", "ipad", "surface tablet", "tablet with keyboard",
            
            # Study materials and accessories
            "textbook", "notebook", "binder", "folder", "planner",
            "pencil case", "pencil box", "calculator", "scientific calculator",
            
            # Electronics and gadgets
            "phone", "smartphone", "headphones", "earbuds", "charger",
            "power bank", "usb drive", "external hard drive",
            
            # People and interactions
            "person", "student", "teacher", "professor", "group of students",
            
            # Classroom furniture and equipment
            "desk", "chair", "table", "whiteboard", "projector", "screen",
            
            # Other campus items
            "water bottle", "lunch box", "coffee cup", "travel mug",
            "bicycle", "motorcycle", "car", "bus"
        ]
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load Grounding DINO model and processor from Hugging Face"""
        logger.info(f"Loading Grounding DINO model: {self.model_name}")
        
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            # Force float32 to avoid dtype mismatches
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Grounding DINO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO model: {e}")
            logger.info("Attempting to load alternative model...")
            
            # Fallback to base model if tiny fails
            try:
                self.model_name = "IDEA-Research/grounding-dino-base"
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                # Force float32 to avoid dtype mismatches
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("Loaded base model successfully")
            except Exception as e2:
                logger.error(f"Failed to load alternative model: {e2}")
                raise
    
    def detect_objects(self,
                      image: Image.Image,
                      text_prompts: Optional[List[str]] = None,
                      return_crops: bool = True) -> List[DetectedObject]:
        """
        Detect objects in an image using text prompts
        
        Args:
            image: PIL Image to detect objects in
            text_prompts: List of object categories to detect (uses defaults if None)
            return_crops: Whether to return cropped object images
            
        Returns:
            List of DetectedObject instances
        """
        # Use default prompts if none provided
        if text_prompts is None:
            text_prompts = self.default_prompts
        
        # Create prompt string (Grounding DINO expects period-separated)
        prompt_text = ". ".join(text_prompts) + "."
        
        try:
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt"
            )
            # Move inputs to device (model is float32)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results using the correct API
            # For Grounding DINO from transformers, use post_process_grounded_object_detection
            # but without box/text thresholds (they're applied during model forward pass)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]
            
            # Extract detections
            detected_objects = []
            
            if len(results["scores"]) > 0:
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]
                
                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.confidence_threshold:
                        # Convert box to integers (x1, y1, x2, y2)
                        bbox = tuple(map(int, box))
                        
                        # Optionally crop the detected region
                        cropped_img = None
                        if return_crops:
                            x1, y1, x2, y2 = bbox
                            # Ensure coordinates are within image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(image.width, x2)
                            y2 = min(image.height, y2)
                            cropped_img = image.crop((x1, y1, x2, y2))
                        
                        detected_obj = DetectedObject(
                            label=label,
                            confidence=float(score),
                            bbox=bbox,
                            cropped_image=cropped_img
                        )
                        detected_objects.append(detected_obj)
            
            logger.info(f"Detected {len(detected_objects)} objects in image")
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def detect_objects_batch(self,
                            images: List[Image.Image],
                            text_prompts: Optional[List[str]] = None,
                            return_crops: bool = True) -> List[List[DetectedObject]]:
        """
        Detect objects in a batch of images
        
        Args:
            images: List of PIL Images
            text_prompts: List of object categories to detect
            return_crops: Whether to return cropped object images
            
        Returns:
            List of lists of DetectedObject instances (one list per image)
        """
        results = []
        
        for image in images:
            detections = self.detect_objects(image, text_prompts, return_crops)
            results.append(detections)
        
        return results
    
    def filter_by_class(self,
                       detections: List[DetectedObject],
                       class_names: List[str]) -> List[DetectedObject]:
        """
        Filter detections by specific object classes
        
        Args:
            detections: List of detected objects
            class_names: List of class names to keep
            
        Returns:
            Filtered list of detections
        """
        class_names_lower = [c.lower() for c in class_names]
        filtered = [d for d in detections if d.label.lower() in class_names_lower]
        return filtered
    
    def get_top_detections(self,
                          detections: List[DetectedObject],
                          top_k: int = 5) -> List[DetectedObject]:
        """
        Get top K detections by confidence
        
        Args:
            detections: List of detected objects
            top_k: Number of top detections to return
            
        Returns:
            Top K detections sorted by confidence
        """
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        return sorted_detections[:top_k]
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")
    
    def unload_model(self):
        """Unload model from memory"""
        del self.model
        del self.processor
        self.clear_gpu_cache()
        logger.info("Model unloaded from memory")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = GroundingDINODetector(
        confidence_threshold=0.3,
        use_gpu=True
    )
    
    # Example: Detect objects in an image
    # from PIL import Image
    # image = Image.open("sample_frame.jpg")
    # 
    # # Detect with default prompts (campus surveillance objects)
    # detections = detector.detect_objects(image)
    # 
    # for det in detections:
    #     print(f"Detected: {det.label} (confidence: {det.confidence:.2f})")
    #     print(f"  Bounding box: {det.bbox}")
    # 
    # # Detect specific objects
    # custom_prompts = ["backpack", "laptop", "phone"]
    # detections = detector.detect_objects(image, text_prompts=custom_prompts)
