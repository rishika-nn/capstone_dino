# Temporal Bootstrapping Implementation

## Overview

This implementation adds three novel features to the video search system:

1. **Temporal Bootstrapping**: Automatically finds related objects at similar timestamps
2. **Adaptive Window**: Adjusts search window size based on motion intensity
3. **Confidence-Aware Boosting**: Weights temporal boosts by detection confidence

## Implementation Details

### Part 1: Temporal Bootstrapping

**The Basic Idea**: "If I find the red shirt at 10 seconds, I should look extra carefully for the blue bag around 10 seconds too"

**Why It Works**: Objects that belong together appear at the same time. A person doesn't teleport away from their bag.

**Implementation**: 
- Located in `temporal_bootstrapping.py`
- Takes primary search results (e.g., "red shirt" detections)
- Automatically searches for related objects at similar timestamps
- Boosts scores for related objects found within temporal proximity

### Part 2: Adaptive Window

**The Basic Idea**: "In a running scene, search 5 seconds before/after. In a still scene, search 1 second before/after"

**The Problem It Solves**:
- **Fast motion scenes** (sports, chase): Objects move quickly → need BIGGER time window
- **Slow motion scenes** (office, library): Objects barely move → need SMALLER time window

**Examples**:
- Person jogging with bag: In 1 second, they've moved 10 meters → need to search wider
- Person sitting with bag: In 1 second, they've barely moved → narrow search is fine

**Implementation**:
- Located in `motion_analyzer.py`
- Uses optical flow to measure pixel movement between frames
- Computes motion intensity (0.0 = still, 1.0 = fast motion)
- Maps motion intensity to adaptive window size:
  - High motion (0.8) → Large window (5.0s)
  - Medium motion (0.5) → Medium window (3.0s)
  - Low motion (0.2) → Small window (1.0s)

### Part 3: Confidence-Aware Boosting

**The Basic Idea**: "Trust strong detections more than weak ones"

**The Problem It Solves**:
Current methods treat all detections equally:
- 95% confident "this is a red shirt"
- 60% confident "this might be a red shirt"

Both get the same boost for nearby bags. That's wrong!

**Solution**:
- High confidence detections (95%) → boost nearby objects a lot
- Medium confidence detections (70%) → boost nearby objects moderately
- Low confidence detections (<50%) → no boost applied

**Implementation**:
- Boost formula: `boost = base_boost * temporal_proximity * confidence_factor`
- Temporal proximity uses exponential decay: closer in time = bigger boost
- Confidence factor uses power function: `confidence^weight`

## File Structure

```
capstone-BLIP-main/
├── motion_analyzer.py              # Optical flow analysis
├── temporal_bootstrapping.py       # Core bootstrapping logic
├── video_search_engine.py          # Updated with search_with_bootstrapping()
├── video_search_config.py          # New configuration parameters
└── temporal_bootstrapping_example.py  # Usage examples
```

## Configuration

New parameters in `video_search_config.py`:

```python
# Temporal Bootstrapping Configuration
ENABLE_TEMPORAL_BOOTSTRAPPING = True  # Enable/disable feature
TEMPORAL_BOOST_FACTOR = 0.3            # Base boost factor (0.0 to 1.0)
CONFIDENCE_WEIGHT = 1.0                # How much to weight confidence
MIN_CONFIDENCE_THRESHOLD = 0.5         # Minimum confidence for boost
MIN_WINDOW_SECONDS = 1.0               # Minimum temporal window
MAX_WINDOW_SECONDS = 5.0               # Maximum temporal window
BASE_WINDOW_SECONDS = 2.0              # Base window size
```

## Usage

### Basic Usage

```python
from video_search_engine import VideoSearchEngine

engine = VideoSearchEngine()

# Process video first
engine.process_video(
    video_path="sample_video.mp4",
    video_name="sample_video",
    upload_to_pinecone=True,
    use_object_detection=True
)

# Search with temporal bootstrapping
results = engine.search_with_bootstrapping(
    primary_query="red shirt",
    auto_extract_related=True,  # Automatically find related objects
    top_k=5,
    video_filter="sample_video"
)

# Results contain both primary and boosted related queries
for query, query_results in results.items():
    print(f"\nQuery: '{query}'")
    for result in query_results:
        if 'boost_amount' in result:
            print(f"  Boosted from {result['original_score']:.3f} to {result['similarity_score']:.3f}")
            print(f"  Reason: {result['boost_reason']}")
        else:
            print(f"  Score: {result['similarity_score']:.3f}")
```

### Manual Related Queries

```python
results = engine.search_with_bootstrapping(
    primary_query="person walking",
    related_queries=["bag", "backpack", "bottle"],  # Manually specify
    top_k=5,
    video_filter="sample_video"
)
```

### Direct Motion Analysis

```python
from motion_analyzer import MotionAnalyzer
from frame_extractor import VideoFrameExtractor

extractor = VideoFrameExtractor()
analyzer = MotionAnalyzer()

frames = extractor.extract_frames("sample_video.mp4")
motion_data = analyzer.analyze_frames(frames)

# Get adaptive window for a timestamp
for timestamp, motion in list(motion_data.items())[:10]:
    window = analyzer.compute_adaptive_window_size(motion.motion_intensity)
    print(f"Time: {timestamp:.2f}s - Motion: {motion.motion_intensity:.3f} - Window: {window:.2f}s")
```

## Algorithm Details

### Boost Calculation

1. **Temporal Proximity**: `exp(-2.0 * (time_distance / window_size))`
   - Same timestamp: proximity = 1.0
   - At window edge: proximity ≈ 0.14

2. **Confidence Factor**: `primary_confidence^confidence_weight`
   - 95% confidence with weight 1.0 → factor = 0.95
   - 60% confidence with weight 1.0 → factor = 0.60
   - <50% confidence → factor = 0.0 (no boost)

3. **Final Boost**: `base_boost_factor * temporal_proximity * confidence_factor`

4. **Score Update**: `new_score = old_score + boost * (1 - old_score)`
   - Ensures scores don't exceed 1.0

### Motion Analysis

1. **Optical Flow**: Uses Farneback dense optical flow (default) or Lucas-Kanade sparse
2. **Motion Intensity**: Computes magnitude of flow vectors, normalized to 0-1 range
3. **Window Mapping**: Linear interpolation between min/max windows based on motion intensity

## Performance Considerations

- **Motion Analysis**: Computed once per video and cached
- **Optical Flow**: Can be slow for large videos; consider sampling frames
- **Bootstrapping**: Searches multiple related queries; may be slower than single query
- **GPU Acceleration**: Motion analysis can use GPU if available

## Future Enhancements

1. **Better Related Query Extraction**: Use NLP/LLM to extract related objects
2. **Learning-based Motion Thresholds**: Adapt thresholds based on video type
3. **Multi-level Bootstrapping**: Chain multiple levels of related objects
4. **Spatial Constraints**: Consider object positions, not just temporal proximity

## Testing

See `temporal_bootstrapping_example.py` for complete usage examples.

## Dependencies

All dependencies are already in `requirements.txt`:
- `opencv-python` - For optical flow computation
- Other dependencies - Already present in the codebase

