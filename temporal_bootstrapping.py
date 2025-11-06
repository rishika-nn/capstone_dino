"""
Temporal Bootstrapping Module
Implements two key features:
1. Temporal Bootstrapping: Find related objects at similar timestamps
2. Confidence-Aware: Weight boosts by detection confidence

Note: Uses fixed window size (suitable for stationary surveillance cameras)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from pinecone_manager import SearchResult, PineconeManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoostedResult:
    """Search result with temporal boost applied"""
    result: SearchResult
    original_score: float
    boost_amount: float
    boosted_score: float
    boost_reason: str  # Explanation of why boost was applied

class TemporalBootstrapper:
    """
    Implements temporal bootstrapping with fixed window and confidence-aware boosting
    Suitable for stationary surveillance cameras
    """
    
    def __init__(self,
                 base_boost_factor: float = 0.3,
                 confidence_weight: float = 1.0,
                 min_confidence_threshold: float = 0.5,
                 fixed_window_seconds: float = 2.0):
        """
        Initialize temporal bootstrapper
        
        Args:
            base_boost_factor: Base boost factor (0.0 to 1.0)
            confidence_weight: How much to weight confidence (0.0 to 2.0)
            min_confidence_threshold: Minimum confidence to apply boost
            fixed_window_seconds: Fixed temporal window size (for stationary cameras)
        """
        self.base_boost_factor = base_boost_factor
        self.confidence_weight = confidence_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.fixed_window_seconds = fixed_window_seconds
        
        logger.info("Temporal bootstrapper initialized (fixed window for stationary cameras)")
    
    def bootstrap_search(self,
                        primary_results: List[SearchResult],
                        related_queries: List[Tuple[str, np.ndarray]],
                        pinecone_manager: PineconeManager,
                        video_name: Optional[str] = None,
                        video_path: Optional[str] = None,
                        frame_timestamps: Optional[List[float]] = None,
                        top_k_per_query: int = 5) -> Dict[str, List[BoostedResult]]:
        """
        Perform temporal bootstrapping search
        
        Part 1: Temporal Bootstrapping
        - Find primary objects (e.g., "red shirt") in primary_results
        - Search for related objects (e.g., "blue bag") at similar timestamps using fixed window
        
        Part 2: Confidence-Aware
        - Weight boosts by primary detection confidence
        - High confidence detections boost related objects more
        
        Args:
            primary_results: Search results from primary query
            related_queries: List of (query_text, query_embedding) tuples for related objects
            pinecone_manager: Pinecone manager for searching
            video_name: Name of video (for filtering)
            video_path: Path to video (unused, kept for compatibility)
            frame_timestamps: Timestamps of frames (unused, kept for compatibility)
            top_k_per_query: Number of results per related query
            
        Returns:
            Dictionary mapping query texts to boosted results
        """
        logger.info(f"Starting temporal bootstrapping with {len(primary_results)} primary results "
                   f"and {len(related_queries)} related queries (fixed window: {self.fixed_window_seconds}s)")
        
        # Step 1: Extract primary object timestamps and confidences
        primary_timestamps = self._extract_timestamps_and_confidences(primary_results)
        
        # Step 2: For each related query, search with fixed window
        all_boosted_results = {}
        
        for query_text, query_embedding in related_queries:
            logger.info(f"Processing related query: '{query_text}'")
            
            # Search for related objects at each primary timestamp with fixed window
            query_results = []
            
            for primary_ts, primary_confidence in primary_timestamps:
                # Use fixed window size (suitable for stationary cameras)
                window_size = self.fixed_window_seconds
                
                # Define time window around primary timestamp
                time_window = (max(0, primary_ts - window_size),
                              primary_ts + window_size)
                
                # Search in Pinecone with time window
                search_results = pinecone_manager.semantic_search(
                    query_embedding=query_embedding,
                    top_k=top_k_per_query * 2,  # Get more for filtering
                    similarity_threshold=0.5,  # Lower threshold initially
                    video_filter=video_name,
                    time_window=time_window
                )
                
                # Apply confidence-aware boost
                boosted_results = self._apply_confidence_aware_boost(
                    search_results=search_results,
                    primary_timestamp=primary_ts,
                    primary_confidence=primary_confidence,
                    window_size=window_size
                )
                
                query_results.extend(boosted_results)
            
            # Deduplicate and merge results
            merged_results = self._merge_boosted_results(query_results)
            
            # Sort by boosted score
            merged_results.sort(key=lambda x: x.boosted_score, reverse=True)
            
            # Take top K
            all_boosted_results[query_text] = merged_results[:top_k_per_query]
            
            logger.info(f"Found {len(merged_results)} boosted results for '{query_text}' "
                       f"(returning top {top_k_per_query})")
        
        return all_boosted_results
    
    def _extract_timestamps_and_confidences(self,
                                           results: List[SearchResult]) -> List[Tuple[float, float]]:
        """
        Extract timestamps and confidences from search results
        
        Args:
            results: List of search results
            
        Returns:
            List of (timestamp, confidence/score) tuples
        """
        timestamps_confidences = []
        for result in results:
            timestamps_confidences.append((result.timestamp, result.score))
        return timestamps_confidences
    
    def _apply_confidence_aware_boost(self,
                                     search_results: List[SearchResult],
                                     primary_timestamp: float,
                                     primary_confidence: float,
                                     window_size: float) -> List[BoostedResult]:
        """
        Apply confidence-aware temporal boost to search results
        
        Args:
            search_results: Search results to boost
            primary_timestamp: Timestamp of primary object detection
            primary_confidence: Confidence of primary detection
            window_size: Window size used for search
            
        Returns:
            List of boosted results
        """
        boosted_results = []
        
        for result in search_results:
            # Calculate temporal proximity (how close in time)
            time_distance = abs(result.timestamp - primary_timestamp)
            
            # Normalize time distance (0.0 = same time, 1.0 = at window edge)
            # Use exponential decay: closer = bigger boost
            normalized_distance = time_distance / window_size if window_size > 0 else 1.0
            temporal_proximity = np.exp(-2.0 * normalized_distance)  # Decay factor
            
            # Confidence-aware boost factor
            # Higher primary confidence = bigger boost
            confidence_factor = np.power(primary_confidence, self.confidence_weight)
            
            # Only apply boost if primary confidence is above threshold
            if primary_confidence < self.min_confidence_threshold:
                confidence_factor = 0.0
            
            # Calculate boost amount
            boost_amount = self.base_boost_factor * temporal_proximity * confidence_factor
            
            # Apply boost to score
            # Boost is additive: new_score = old_score + boost * (1 - old_score)
            # This ensures scores don't exceed 1.0
            boosted_score = min(1.0, result.score + boost_amount * (1.0 - result.score))
            
            # Create boosted result
            boosted_result = BoostedResult(
                result=result,
                original_score=result.score,
                boost_amount=boost_amount,
                boosted_score=boosted_score,
                boost_reason=f"Temporal proximity: {time_distance:.2f}s, "
                           f"Primary confidence: {primary_confidence:.2f}"
            )
            
            boosted_results.append(boosted_result)
        
        return boosted_results
    
    def _merge_boosted_results(self,
                              results: List[BoostedResult],
                              time_threshold: float = 1.0) -> List[BoostedResult]:
        """
        Merge duplicate boosted results (same frame detected multiple times)
        Keep the result with highest boosted score
        
        Args:
            results: List of boosted results
            time_threshold: Time window to consider as duplicate
            
        Returns:
            Merged list of boosted results
        """
        if not results:
            return []
        
        # Group by frame_id and video_name
        grouped = defaultdict(list)
        for result in results:
            key = (result.result.frame_id, result.result.video_name)
            grouped[key].append(result)
        
        merged = []
        for key, group in grouped.items():
            # If multiple boosts for same frame, take maximum
            best_result = max(group, key=lambda x: x.boosted_score)
            merged.append(best_result)
        
        # Also merge by timestamp if same video (within time threshold)
        final_merged = []
        seen = {}  # (video_name, timestamp_bucket)
        
        for result in sorted(merged, key=lambda x: x.boosted_score, reverse=True):
            video_name = result.result.video_name
            timestamp = result.result.timestamp
            timestamp_bucket = int(timestamp / time_threshold)
            
            key = (video_name, timestamp_bucket)
            
            if key not in seen:
                seen[key] = result
                final_merged.append(result)
            else:
                # Compare with existing - keep better one
                existing = seen[key]
                if result.boosted_score > existing.boosted_score:
                    # Replace
                    final_merged.remove(existing)
                    final_merged.append(result)
                    seen[key] = result
        
        return final_merged
    
    def simple_bootstrap(self,
                        primary_results: List[SearchResult],
                        related_query: str,
                        related_embedding: np.ndarray,
                        pinecone_manager: PineconeManager,
                        video_name: Optional[str] = None,
                        video_path: Optional[str] = None,
                        frame_timestamps: Optional[List[float]] = None,
                        top_k: int = 10) -> List[BoostedResult]:
        """
        Simplified bootstrap for single related query
        
        Args:
            primary_results: Primary search results
            related_query: Related query text
            related_embedding: Related query embedding
            pinecone_manager: Pinecone manager
            video_name: Video name filter
            video_path: Video path for motion analysis
            frame_timestamps: Frame timestamps
            top_k: Number of results to return
            
        Returns:
            List of boosted results
        """
        return self.bootstrap_search(
            primary_results=primary_results,
            related_queries=[(related_query, related_embedding)],
            pinecone_manager=pinecone_manager,
            video_name=video_name,
            video_path=video_path,
            frame_timestamps=frame_timestamps,
            top_k_per_query=top_k
        ).get(related_query, [])


# Example usage
if __name__ == "__main__":
    # Example would be integrated into search engine
    pass

