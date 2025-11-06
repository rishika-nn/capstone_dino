"""
Object-Focused Captioning Pipeline
Combines Grounding DINO object detection with BLIP attribute-based captioning
Designed for campus surveillance: focuses on objects and their attributes, not actions
"""

import torch
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from object_detector import GroundingDINODetector, DetectedObject
from caption_generator import BlipCaptionGenerator
from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ObjectCaption:
    """Data structure for object-focused captions"""
    frame_data: FrameData
    object_label: str  # From Grounding DINO
    object_bbox: Tuple[int, int, int, int]
    attribute_caption: str  # From BLIP (focused on attributes)
    confidence: float
    is_object_focused: bool = True  # Flag to distinguish from scene captions

class ObjectCaptionPipeline:
    """
    Pipeline that detects objects using Grounding DINO and generates 
    attribute-focused captions using BLIP
    """
    
    def __init__(self,
                 object_detector: Optional[GroundingDINODetector] = None,
                 caption_generator: Optional[BlipCaptionGenerator] = None,
                 use_gpu: bool = True,
                 min_object_size: int = 20,
                 max_objects_per_frame: int = 10,
                 include_scene_caption: bool = False):
        """
        Initialize the object-focused captioning pipeline
        
        Args:
            object_detector: Grounding DINO detector instance (creates new if None)
            caption_generator: BLIP caption generator (creates new if None)
            use_gpu: Whether to use GPU
            min_object_size: Minimum object size (pixels) to caption
            max_objects_per_frame: Maximum objects to caption per frame
            include_scene_caption: Whether to also generate full scene caption
        """
        self.use_gpu = use_gpu
        self.min_object_size = min_object_size
        self.max_objects_per_frame = max_objects_per_frame
        self.include_scene_caption = include_scene_caption
        
        # Initialize or use provided components
        if object_detector is None:
            logger.info("Initializing Grounding DINO detector...")
            self.object_detector = GroundingDINODetector(
                confidence_threshold=0.25,
                use_gpu=use_gpu
            )
        else:
            self.object_detector = object_detector
        
        if caption_generator is None:
            logger.info("Initializing BLIP caption generator...")
            self.caption_generator = BlipCaptionGenerator(
                batch_size=4,  # Smaller batch for cropped objects
                use_gpu=use_gpu,
                max_length=30,  # Shorter captions for objects
                num_beams=3
            )
        else:
            self.caption_generator = caption_generator
        
        # Attribute-focused prompts for BLIP
        # Tuned to elicit attributes (color, material, style) instead of actions
        self.attribute_prompts = [
            "a close-up photo of the {label}. Describe color, material, and notable attributes",
            "describe the {label} focusing on color, style, and accessories",
            "{label}. Provide a short attribute-focused description (color, size, pattern)"
        ]
        
        # Common color words to help score candidate captions
        self._color_words = set([
            'black','white','gray','grey','red','orange','yellow','green','blue','purple','pink',
            'brown','beige','tan','gold','silver','navy','maroon','teal'
        ])
        
        logger.info("Object-focused captioning pipeline initialized")
    
    def process_frame(self,
                     frame_data: FrameData,
                     object_prompts: Optional[List[str]] = None) -> List[ObjectCaption]:
        """
        Process a single frame: detect objects and caption them
        
        Args:
            frame_data: Frame to process
            object_prompts: Specific objects to detect (uses defaults if None)
            
        Returns:
            List of ObjectCaption instances
        """
        object_captions = []
        
        try:
            # Step 1: Detect objects using Grounding DINO
            detections = self.object_detector.detect_objects(
                image=frame_data.image,
                text_prompts=object_prompts,
                return_crops=True
            )
            
            # Filter small objects and limit to top detections
            detections = self._filter_detections(detections)
            
            if not detections:
                logger.debug(f"No objects detected in frame {frame_data.frame_id}")
                
                # Optionally generate scene caption as fallback
                if self.include_scene_caption:
                    scene_caption = self._generate_scene_caption(frame_data)
                    if scene_caption:
                        object_captions.append(scene_caption)
                
                return object_captions
            
            # Step 2: Generate attribute-focused captions for each detected object
            for detection in detections:
                if detection.cropped_image is None:
                    continue
                
                # Generate caption for cropped object
                caption_text = self._caption_object(
                    detection.cropped_image,
                    detection.label
                )
                
                if caption_text and len(caption_text.split()) >= 3:
                    obj_caption = ObjectCaption(
                        frame_data=frame_data,
                        object_label=detection.label,
                        object_bbox=detection.bbox,
                        attribute_caption=caption_text,
                        confidence=detection.confidence,
                        is_object_focused=True
                    )
                    object_captions.append(obj_caption)
            
            logger.debug(f"Generated {len(object_captions)} object captions for frame {frame_data.frame_id}")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_data.frame_id}: {e}")
        
        return object_captions
    
    def process_frames(self,
                      frames: List[FrameData],
                      object_prompts: Optional[List[str]] = None,
                      show_progress: bool = True) -> List[ObjectCaption]:
        """
        Process multiple frames
        
        Args:
            frames: List of frames to process
            object_prompts: Specific objects to detect
            show_progress: Whether to show progress bar
            
        Returns:
            List of all ObjectCaption instances from all frames
        """
        all_captions = []
        
        iterator = tqdm(frames, desc="Processing frames") if show_progress else frames
        
        for frame_data in iterator:
            frame_captions = self.process_frame(frame_data, object_prompts)
            all_captions.extend(frame_captions)
        
        logger.info(f"Generated {len(all_captions)} total object captions from {len(frames)} frames")
        return all_captions
    
    def _filter_detections(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filter detections by size and limit to top K"""
        # Filter by minimum size
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            
            if width >= self.min_object_size and height >= self.min_object_size:
                filtered.append(det)
        
        # Get top K by confidence
        filtered = self.object_detector.get_top_detections(
            filtered,
            top_k=self.max_objects_per_frame
        )
        
        return filtered
    
    def _score_attribute_caption(self, caption: str, object_label: str) -> float:
        """Heuristic score: prefer captions mentioning colors/attributes and the label."""
        c = (caption or "").lower()
        color_hits = sum(1 for w in self._color_words if w in c)
        label_hit = 2.0 if object_label.lower() in c else 0.0
        length_bonus = min(len(c.split()) / 10.0, 1.0)  # small bonus up to 1
        return color_hits * 1.5 + label_hit + length_bonus
    
    def _caption_object(self, cropped_image: Image.Image, object_label: str) -> str:
        """
        Generate attribute-focused caption for a detected object
        
        Args:
            cropped_image: Cropped object image
            object_label: Object label from Grounding DINO
            
        Returns:
            Attribute-focused caption string
        """
        try:
            # Generate single unconditional caption for the cropped object
            with torch.no_grad():
                inputs = self.caption_generator.processor(
                    images=cropped_image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.caption_generator.device) for k, v in inputs.items()}
                outputs = self.caption_generator.model.generate(
                    **inputs,
                    max_length=25,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True
                )
                raw_caption = self.caption_generator.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Format with object label and clean up
            formatted_caption = self._format_attribute_caption(raw_caption, object_label)
            return formatted_caption
            
        except Exception as e:
            logger.error(f"Error captioning object {object_label}: {e}")
            return f"{object_label}"
    
    def _format_attribute_caption(self, blip_caption: str, object_label: str) -> str:
        """
        Format BLIP caption to be attribute-focused
        
        Args:
            blip_caption: Raw caption from BLIP
            object_label: Object label from detector
            
        Returns:
            Formatted attribute caption
        """
        # Clean caption
        caption = (blip_caption or "").strip()
        
        # Extract color words from caption
        colors_found = [w for w in self._color_words if w in caption.lower()]
        
        # Remove action verbs to focus on attributes
        action_words = ['walking', 'running', 'sitting', 'standing', 'holding', 'carrying', 'talking', 'wearing']
        tokens = caption.lower().split()
        filtered_tokens = [t for t in tokens if t not in action_words]
        caption = ' '.join(filtered_tokens).strip().capitalize()
        
        # Build object-oriented caption
        obj_type = object_label.title()
        
        if colors_found and caption:
            # Example: "Backpack: black backpack with straps"
            color_str = colors_found[0]  # Use first detected color
            if color_str not in caption.lower():
                caption = f"{color_str} {caption}"
            formatted = f"{obj_type}: {caption}"
        elif caption:
            # Example: "Person: person in outdoor setting"
            formatted = f"{obj_type}: {caption}"
        else:
            # Fallback: just the object type
            formatted = obj_type
        
        # Normalize ending punctuation
        if formatted and formatted[-1] not in '.!?':
            formatted += '.'
        
        return formatted
    
    def _generate_scene_caption(self, frame_data: FrameData) -> Optional[ObjectCaption]:
        """
        Generate a full scene caption as fallback
        
        Args:
            frame_data: Frame to caption
            
        Returns:
            ObjectCaption with scene description
        """
        try:
            # Generate standard BLIP caption for full frame
            inputs = self.caption_generator.processor(
                images=frame_data.image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.caption_generator.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.caption_generator.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True
                )
            
            caption = self.caption_generator.processor.decode(outputs[0], skip_special_tokens=True)
            caption = self.caption_generator._clean_caption(caption)
            
            # Create scene caption object
            scene_caption = ObjectCaption(
                frame_data=frame_data,
                object_label="scene",
                object_bbox=(0, 0, frame_data.image.width, frame_data.image.height),
                attribute_caption=caption,
                confidence=1.0,
                is_object_focused=False
            )
            
            return scene_caption
            
        except Exception as e:
            logger.error(f"Error generating scene caption: {e}")
            return None
    
    def get_statistics(self, object_captions: List[ObjectCaption]) -> Dict:
        """Get statistics about generated object captions"""
        if not object_captions:
            return {"total": 0}
        
        # Count by object type
        object_counts = {}
        for oc in object_captions:
            label = oc.object_label
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(oc.confidence for oc in object_captions) / len(object_captions)
        
        stats = {
            "total_captions": len(object_captions),
            "unique_frames": len(set(oc.frame_data.frame_id for oc in object_captions)),
            "object_focused": sum(1 for oc in object_captions if oc.is_object_focused),
            "scene_captions": sum(1 for oc in object_captions if not oc.is_object_focused),
            "avg_confidence": avg_confidence,
            "objects_detected": object_counts,
            "most_common_object": max(object_counts, key=object_counts.get) if object_counts else None
        }
        
        return stats
    
    def clear_cache(self):
        """Clear GPU cache from both models"""
        self.object_detector.clear_gpu_cache()
        self.caption_generator.clear_gpu_cache()
        logger.info("Pipeline cache cleared")
    
    def unload_models(self):
        """Unload both models from memory"""
        self.object_detector.unload_model()
        self.caption_generator.unload_model()
        logger.info("Pipeline models unloaded")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ObjectCaptionPipeline(use_gpu=True)
    
    # Example: Process frames
    # from frame_extractor import VideoFrameExtractor
    # 
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # # Process with object-focused captioning
    # object_captions = pipeline.process_frames(frames)
    # 
    # # Print results
    # for oc in object_captions[:10]:
    #     print(f"Time: {oc.frame_data.timestamp:.2f}s")
    #     print(f"  Object: {oc.object_label} (conf: {oc.confidence:.2f})")
    #     print(f"  Caption: {oc.attribute_caption}")
    #     print()
    # 
    # # Get statistics
    # stats = pipeline.get_statistics(object_captions)
    # print(f"Statistics: {stats}")
