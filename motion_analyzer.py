"""
Motion Analysis Module using Optical Flow
Computes motion intensity for adaptive temporal windows
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MotionData:
    """Data structure for motion information"""
    timestamp: float
    motion_intensity: float  # 0.0 (still) to 1.0 (fast motion)
    optical_flow_magnitude: float

class MotionAnalyzer:
    """
    Analyze motion in video using optical flow
    Computes motion intensity per frame to determine adaptive search windows
    """
    
    def __init__(self,
                 use_gpu: bool = False,
                 flow_method: str = "farneback"):
        """
        Initialize motion analyzer
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CUDA)
            flow_method: Optical flow method ("farneback" or "lucas_kanade")
        """
        self.use_gpu = use_gpu
        self.flow_method = flow_method
        self.motion_cache = {}  # Cache motion data by video
        
        logger.info(f"Motion analyzer initialized (method: {flow_method})")
    
    def analyze_video(self,
                     video_path: str,
                     frame_timestamps: List[float],
                     frames_per_second: float = 30.0) -> Dict[float, MotionData]:
        """
        Analyze motion for a video and return motion intensity per timestamp
        
        Args:
            video_path: Path to video file
            frame_timestamps: List of timestamps to analyze
            frames_per_second: Video FPS
            
        Returns:
            Dictionary mapping timestamps to MotionData
        """
        logger.info(f"Analyzing motion for video: {video_path}")
        
        # Check cache
        if video_path in self.motion_cache:
            logger.info("Using cached motion data")
            return self.motion_cache[video_path]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or frames_per_second
        motion_data = {}
        
        try:
            # Get frames at specified timestamps
            prev_frame = None
            prev_timestamp = None
            
            for timestamp in tqdm(sorted(frame_timestamps), desc="Computing motion"):
                # Seek to timestamp
                frame_num = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Compute optical flow
                    flow = self._compute_optical_flow(prev_frame, gray)
                    motion_intensity = self._compute_motion_intensity(flow)
                    
                    motion_data[timestamp] = MotionData(
                        timestamp=timestamp,
                        motion_intensity=motion_intensity,
                        optical_flow_magnitude=np.mean(np.abs(flow))
                    )
                else:
                    # First frame - no motion
                    motion_data[timestamp] = MotionData(
                        timestamp=timestamp,
                        motion_intensity=0.0,
                        optical_flow_magnitude=0.0
                    )
                
                prev_frame = gray.copy()
                prev_timestamp = timestamp
            
            # Cache results
            self.motion_cache[video_path] = motion_data
            
        finally:
            cap.release()
        
        logger.info(f"Computed motion data for {len(motion_data)} timestamps")
        return motion_data
    
    def analyze_frames(self,
                      frames: List[FrameData]) -> Dict[float, MotionData]:
        """
        Analyze motion directly from FrameData objects
        
        Args:
            frames: List of FrameData objects
            
        Returns:
            Dictionary mapping timestamps to MotionData
        """
        motion_data = {}
        
        if len(frames) < 2:
            # Not enough frames for motion analysis
            for frame in frames:
                motion_data[frame.timestamp] = MotionData(
                    timestamp=frame.timestamp,
                    motion_intensity=0.0,
                    optical_flow_magnitude=0.0
                )
            return motion_data
        
        # Sort frames by timestamp
        sorted_frames = sorted(frames, key=lambda f: f.timestamp)
        
        prev_gray = None
        
        for i, frame_data in enumerate(tqdm(sorted_frames, desc="Analyzing motion")):
            # Convert PIL to numpy array
            frame_np = np.array(frame_data.image)
            if len(frame_np.shape) == 3:
                gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame_np
            
            if prev_gray is not None:
                # Compute optical flow
                flow = self._compute_optical_flow(prev_gray, gray)
                motion_intensity = self._compute_motion_intensity(flow)
                
                motion_data[frame_data.timestamp] = MotionData(
                    timestamp=frame_data.timestamp,
                    motion_intensity=motion_intensity,
                    optical_flow_magnitude=np.mean(np.abs(flow))
                )
            else:
                # First frame
                motion_data[frame_data.timestamp] = MotionData(
                    timestamp=frame_data.timestamp,
                    motion_intensity=0.0,
                    optical_flow_magnitude=0.0
                )
            
            prev_gray = gray
        
        return motion_data
    
    def _compute_optical_flow(self,
                            prev_frame: np.ndarray,
                            curr_frame: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames
        
        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            
        Returns:
            Optical flow vector field
        """
        if self.flow_method == "farneback":
            # Farneback dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif self.flow_method == "lucas_kanade":
            # Lucas-Kanade sparse optical flow
            # First detect corners
            corners = cv2.goodFeaturesToTrack(
                prev_frame,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            if corners is not None:
                # Track points
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_frame, curr_frame,
                    corners, None
                )
                
                # Convert to dense flow (simplified)
                flow = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2))
                good_pts = next_pts[status == 1]
                good_corners = corners[status == 1]
                for pt, corner in zip(good_pts, good_corners):
                    dx = pt[0] - corner[0]
                    dy = pt[1] - corner[1]
                    # Simple interpolation (in production, use better method)
                    y, x = int(corner[1]), int(corner[0])
                    if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                        flow[y, x] = [dx, dy]
            else:
                flow = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2))
        else:
            raise ValueError(f"Unknown flow method: {self.flow_method}")
        
        return flow
    
    def _compute_motion_intensity(self, flow: np.ndarray) -> float:
        """
        Compute motion intensity from optical flow
        Returns value between 0.0 (still) and 1.0 (fast motion)
        
        Args:
            flow: Optical flow vector field
            
        Returns:
            Motion intensity (0.0 to 1.0)
        """
        # Compute magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Get statistics
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        
        # Normalize to 0-1 range using adaptive thresholding
        # Higher mean magnitude = more motion
        # We use a sigmoid-like normalization
        
        # Threshold for "significant motion" (pixels per frame)
        # This is tuned based on typical video motion
        motion_threshold = 5.0  # pixels
        
        # Normalize using sigmoid function
        normalized = 1.0 / (1.0 + np.exp(-(mean_magnitude - motion_threshold) / 2.0))
        
        # Clamp to [0, 1]
        motion_intensity = np.clip(normalized, 0.0, 1.0)
        
        return float(motion_intensity)
    
    def get_motion_at_timestamp(self,
                               timestamp: float,
                               motion_data: Dict[float, MotionData]) -> float:
        """
        Get motion intensity at a specific timestamp (with interpolation)
        
        Args:
            timestamp: Target timestamp
            motion_data: Dictionary of motion data
            
        Returns:
            Motion intensity (0.0 to 1.0)
        """
        if not motion_data:
            return 0.5  # Default medium motion
        
        # Find closest timestamp
        closest_ts = min(motion_data.keys(), key=lambda x: abs(x - timestamp))
        
        # If very close, return directly
        if abs(closest_ts - timestamp) < 0.5:  # Within 0.5 seconds
            return motion_data[closest_ts].motion_intensity
        
        # Otherwise interpolate between nearest neighbors
        sorted_ts = sorted(motion_data.keys())
        
        if timestamp < sorted_ts[0]:
            return motion_data[sorted_ts[0]].motion_intensity
        if timestamp > sorted_ts[-1]:
            return motion_data[sorted_ts[-1]].motion_intensity
        
        # Find bounding timestamps
        for i in range(len(sorted_ts) - 1):
            if sorted_ts[i] <= timestamp <= sorted_ts[i + 1]:
                # Linear interpolation
                t1, t2 = sorted_ts[i], sorted_ts[i + 1]
                m1 = motion_data[t1].motion_intensity
                m2 = motion_data[t2].motion_intensity
                
                alpha = (timestamp - t1) / (t2 - t1)
                return m1 * (1 - alpha) + m2 * alpha
        
        return 0.5  # Fallback
    
    def compute_adaptive_window_size(self,
                                    motion_intensity: float,
                                    base_window: float = 2.0,
                                    min_window: float = 1.0,
                                    max_window: float = 5.0) -> float:
        """
        Compute adaptive search window size based on motion intensity
        
        Fast motion (high intensity) -> larger window
        Slow motion (low intensity) -> smaller window
        
        Args:
            motion_intensity: Motion intensity (0.0 to 1.0)
            base_window: Base window size in seconds
            min_window: Minimum window size in seconds
            max_window: Maximum window size in seconds
            
        Returns:
            Adaptive window size in seconds
        """
        # Linear interpolation between min and max based on motion
        window_size = min_window + (max_window - min_window) * motion_intensity
        
        # Clamp to bounds
        window_size = np.clip(window_size, min_window, max_window)
        
        return float(window_size)


# Example usage
if __name__ == "__main__":
    analyzer = MotionAnalyzer()
    
    # Example: Analyze motion from frames
    # from frame_extractor import VideoFrameExtractor
    # 
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # motion_data = analyzer.analyze_frames(frames)
    # 
    # for timestamp, motion in list(motion_data.items())[:10]:
    #     print(f"Time: {timestamp:.2f}s - Motion: {motion.motion_intensity:.3f}")
    #     window = analyzer.compute_adaptive_window_size(motion.motion_intensity)
    #     print(f"  Adaptive window: {window:.2f}s")

