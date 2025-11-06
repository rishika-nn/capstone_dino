"""
BLIP Caption Generation Module
Generates semantic captions for video frames using the BLIP vision-language model
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import logging
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import gc
from dataclasses import dataclass
from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CaptionedFrame:
    """Data structure for frame with caption"""
    frame_data: FrameData
    caption: str
    confidence: Optional[float] = None

class BlipCaptionGenerator:
    """Generate captions for frames using BLIP model"""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-base",
                 batch_size: int = 8,
                 use_gpu: bool = True,
                 max_length: int = 50,
                 num_beams: int = 4,
                 generate_multiple_captions: bool = False,
                 captions_per_frame: int = 3):
        """
        Initialize the BLIP caption generator
        
        Args:
            model_name: Hugging Face model identifier
            batch_size: Batch size for processing
            use_gpu: Whether to use GPU if available
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.generate_multiple_captions = generate_multiple_captions
        self.captions_per_frame = captions_per_frame
        
        # Object-focused prompts for InstructBLIP - these will be used as instructions
        self.object_prompts = [
            "Describe the objects in this image, focusing on colors, sizes, and notable attributes like black backpack, red shirt",
            "List the objects visible in this image with their colors and key attributes",
            "Provide a detailed description of all objects, emphasizing their colors and distinguishing features"
        ]
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        # model_family will be set in _load_model: either 'blip2' or 'blip'
        self.model_family = None
        self._load_model()
        
    def _load_model(self):
        """Load BLIP/InstructBLIP model and processor"""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Try loading as BLIP-2 / InstructBLIP (Blip2Processor + Blip2ForConditionalGeneration)
            logger.info("Attempting to load model as BLIP-2 (Blip2Processor)")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_family = 'blip2'
            logger.info("Loaded BLIP-2 / InstructBLIP model successfully")
            
        except Exception as e:
            # If BLIP-2 loading fails, attempt to load classic BLIP (BlipProcessor)
            logger.warning(f"BLIP-2 load failed: {e}. Attempting to load classic BLIP model as fallback...")
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                self.model_family = 'blip'
                logger.info("Loaded classic BLIP model successfully (fallback)")
            except Exception as e2:
                logger.error(f"Failed to load BLIP fallback model: {e2}")
                # Re-raise the original exception to inform the caller
                raise
    
    def generate_captions(self, frames: List[FrameData], 
                         filter_empty: bool = True,
                         min_caption_length: int = 3) -> List[CaptionedFrame]:
        """
        Generate captions for a list of frames
        
        Args:
            frames: List of FrameData objects
            filter_empty: Whether to filter out empty/short captions
            min_caption_length: Minimum caption length (in words)
            
        Returns:
            List of CaptionedFrame objects
        """
        logger.info(f"Generating captions for {len(frames)} frames")
        
        captioned_frames = []
        
        # If multiple captions per frame is enabled, generate variants
        if self.generate_multiple_captions:
            logger.info(f"Generating {self.captions_per_frame} caption variants per frame")
            for frame_data in tqdm(frames, desc="Generating multi-captions"):
                variants = self.generate_object_focused_captions(frame_data, 
                                                                 num_variants=self.captions_per_frame)
                
                for caption in variants:
                    # Filter empty or short captions if requested
                    if filter_empty and len(caption.split()) < min_caption_length:
                        continue
                    
                    captioned_frame = CaptionedFrame(
                        frame_data=frame_data,
                        caption=caption
                    )
                    captioned_frames.append(captioned_frame)
        else:
            # Process frames in batches (original behavior)
            for batch_start in tqdm(range(0, len(frames), self.batch_size), 
                                    desc="Generating captions"):
                batch_end = min(batch_start + self.batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                
                # Generate captions for batch
                batch_captions = self._generate_batch_captions(batch_frames)
                
                # Create CaptionedFrame objects
                for frame_data, caption in zip(batch_frames, batch_captions):
                    # Filter empty or short captions if requested
                    if filter_empty and len(caption.split()) < min_caption_length:
                        logger.debug(f"Skipping short caption: '{caption}' for frame {frame_data.frame_id}")
                        continue
                    
                    captioned_frame = CaptionedFrame(
                        frame_data=frame_data,
                        caption=caption
                    )
                    captioned_frames.append(captioned_frame)
        
        logger.info(f"Generated {len(captioned_frames)} valid captions")
        return captioned_frames
    
    def _generate_batch_captions(self, batch_frames: List[FrameData], 
                                  text_prompt: Optional[str] = None) -> List[str]:
        """Generate captions for a batch of frames with optional text prompting"""
        try:
            # Extract PIL images from frame data
            images = [frame.image for frame in batch_frames]
            
            # Use object-focused instruction prompt if none provided (for InstructBLIP)
            if not text_prompt:
                text_prompt = "Describe the objects in this image, focusing on colors, sizes, and notable attributes like black backpack, red shirt"

            # Prepare inputs depending on loaded model family
            if self.model_family == 'blip2':
                # Preprocess images with instruction prompt (InstructBLIP uses text as instruction)
                inputs = self.processor(images=images, 
                                       text=[text_prompt] * len(images),
                                       return_tensors="pt", 
                                       padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate captions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        do_sample=False,  # Deterministic generation
                        early_stopping=True
                    )

                # Decode captions (batch)
                captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                # Classic BLIP (no instruction text support in the same way)
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate captions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        do_sample=False,
                        early_stopping=True
                    )

                # Decode captions (processor.decode supports single output, use batch list comprehension)
                try:
                    captions = [self.processor.decode(o, skip_special_tokens=True) for o in outputs]
                except Exception:
                    # Fallback to batch_decode if available
                    captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up captions
            captions = [self._clean_caption(caption) for caption in captions]
            
            return captions
            
        except Exception as e:
            logger.error(f"Error generating batch captions: {e}")
            # Return empty captions for failed batch
            return ["" for _ in batch_frames]
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and normalize generated caption"""
        # Remove extra whitespace
        caption = " ".join(caption.split())
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Ensure caption ends with period if it doesn't have punctuation
        if caption and caption[-1] not in '.!?':
            caption += '.'
        
        return caption
    
    def generate_object_focused_captions(self, frame: FrameData, 
                                         num_variants: int = 3) -> List[str]:
        """
        Generate multiple object-focused caption variants for a single frame
        Uses InstructBLIP instruction prompts to focus on objects, colors, and attributes
        
        Args:
            frame: Single FrameData object
            num_variants: Number of caption variants to generate
            
        Returns:
            List of diverse, object-focused caption variants
        """
        image = frame.image
        captions = []
        
        with torch.no_grad():
            if self.model_family == 'blip2':
                # Use instruction prompts for object-focused descriptions
                # InstructBLIP works best with explicit instructions
                for i, prompt in enumerate(self.object_prompts[:num_variants]):
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        do_sample=False,
                        early_stopping=True
                    )
                    caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                    captions.append(self._clean_caption(caption))
                
                # If we need more variants, generate with slightly different prompts
                if len(captions) < num_variants:
                    fallback_prompt = "Describe all visible objects with their colors and key attributes"
                    inputs = self.processor(images=image, text=fallback_prompt, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        do_sample=True,
                        temperature=0.7,
                        early_stopping=True
                    )
                    caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                    captions.append(self._clean_caption(caption))
            else:
                # Classic BLIP fallback: generate variants by sampling the image-only caption model
                for i in range(num_variants):
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams if i==0 else 1,
                        do_sample=True,
                        temperature=0.7 if i>0 else 0.0,
                        early_stopping=True
                    )
                    try:
                        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                    except Exception:
                        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    captions.append(self._clean_caption(caption))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_captions = []
        for caption in captions:
            caption_lower = caption.lower().strip()
            if caption_lower not in seen and len(caption.split()) >= 3:
                seen.add(caption_lower)
                unique_captions.append(caption)
        
        # If we don't have enough unique captions, keep at least one
        if not unique_captions and captions:
            unique_captions = [captions[0]]
        
        return unique_captions
    
    def generate_caption_variants(self, frame: FrameData, 
                                 num_variants: int = 3) -> List[str]:
        """
        Generate multiple caption variants for a single frame
        Useful for improving search diversity
        
        Args:
            frame: Single FrameData object
            num_variants: Number of caption variants to generate
            
        Returns:
            List of caption variants
        """
        # Delegate to the new object-focused method
        return self.generate_object_focused_captions(frame, num_variants)
    
    def filter_duplicate_captions(self, 
                                 captioned_frames: List[CaptionedFrame],
                                 time_window: float = 2.0) -> List[CaptionedFrame]:
        """
        Filter duplicate captions within a time window
        Note: When generating multiple captions per frame, this allows different captions
        for the same frame but prevents exact duplicate captions nearby in time
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            time_window: Time window in seconds to check for duplicates
            
        Returns:
            Filtered list of CaptionedFrame objects
        """
        filtered = []
        caption_timestamps = {}  # Track last timestamp for each caption
        
        for cf in captioned_frames:
            caption_lower = cf.caption.lower().strip()
            timestamp = cf.frame_data.timestamp
            
            # Check if we've seen this EXACT caption text recently
            if caption_lower in caption_timestamps:
                last_timestamp = caption_timestamps[caption_lower]
                # Only filter if it's the exact same caption within time window
                # Different captions for same frame are allowed
                if abs(timestamp - last_timestamp) < time_window:
                    # Skip exact duplicate within time window
                    logger.debug(f"Skipping duplicate caption at {timestamp:.2f}s: '{cf.caption}'")
                    continue
            
            # Keep this caption
            filtered.append(cf)
            caption_timestamps[caption_lower] = timestamp
        
        filtered_count = len(captioned_frames) - len(filtered)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} duplicate captions")
        return filtered
    
    def enhance_captions_with_context(self, 
                                     captioned_frames: List[CaptionedFrame],
                                     video_name: str = "video") -> List[CaptionedFrame]:
        """
        Enhance captions with contextual information
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            video_name: Name of the video for context
            
        Returns:
            Enhanced CaptionedFrame objects
        """
        for cf in captioned_frames:
            # Add timestamp context to caption
            timestamp_str = f"[{cf.frame_data.timestamp:.1f}s]"
            
            # You could add more context here based on your needs
            # For now, we'll keep the original caption but store metadata
            cf.frame_data.video_name = video_name
        
        return captioned_frames
    
    def get_caption_statistics(self, captioned_frames: List[CaptionedFrame]) -> Dict:
        """Get statistics about generated captions"""
        if not captioned_frames:
            return {"total": 0}
        
        captions = [cf.caption for cf in captioned_frames]
        caption_lengths = [len(c.split()) for c in captions]
        
        stats = {
            "total": len(captions),
            "unique": len(set(captions)),
            "avg_length": sum(caption_lengths) / len(caption_lengths),
            "min_length": min(caption_lengths),
            "max_length": max(caption_lengths),
            "duplicate_rate": 1 - (len(set(captions)) / len(captions))
        }
        
        return stats
    
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


# Example usage and testing
if __name__ == "__main__":
    # Test the caption generator
    generator = BlipCaptionGenerator(
        batch_size=8,
        use_gpu=True,
        max_length=50,
        num_beams=4
    )
    
    # Example: Generate captions for frames
    # from frame_extractor import VideoFrameExtractor
    # 
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # captioned_frames = generator.generate_captions(frames)
    # 
    # # Get statistics
    # stats = generator.get_caption_statistics(captioned_frames)
    # print(f"Caption statistics: {stats}")
    # 
    # # Print some examples
    # for cf in captioned_frames[:5]:
    #     print(f"Time: {cf.frame_data.timestamp:.2f}s - Caption: {cf.caption}")