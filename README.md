# Video Frame Search System with BLIP & Pinecone

An industry-ready, production-grade video semantic search engine that enables natural language queries to find exact timestamps in videos. The system extracts frames, generates captions using BLIP, stores embeddings in Pinecone, and provides fast semantic search capabilities.

## 🎯 Features

- **Intelligent Frame Extraction**: Removes redundant frames using visual similarity analysis
- **Automatic Caption Generation**: Uses BLIP vision-language model for semantic understanding
- **Vector Search**: Leverages Pinecone for scalable, fast semantic search
- **Natural Language Queries**: Search videos using plain English descriptions
- **Timestamp Precision**: Returns exact timestamps of matching content
- **Production Ready**: Comprehensive error handling, logging, and configuration management
- **GPU Acceleration**: Optimized for NVIDIA GPUs (falls back to CPU if unavailable)

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for performance)
- Pinecone account (free tier available at https://www.pinecone.io)
- At least 8GB RAM (16GB recommended)
- 4GB+ GPU memory for optimal performance

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd video-search-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Pinecone API key
# Get your API key from: https://app.pinecone.io
```

### 3. Basic Usage

```python
from video_search_engine import VideoSearchEngine

# Initialize the engine
engine = VideoSearchEngine()

# Process a video
stats = engine.process_video(
    video_path="path/to/your/video.mp4",
    video_name="my_video",
    save_frames=False,  # Set to True to save extracted frames
    upload_to_pinecone=True
)

# Search for content
results = engine.search(
    query="person walking with a black bag",
    top_k=5
)

# Display results
for result in results:
    print(f"Time: {result['time_formatted']} - Score: {result['similarity_score']:.3f}")
    print(f"  Caption: {result['caption']}")
    print(f"  Video: {result['video_name']}")
```

## 📖 Detailed Documentation

### System Architecture

```
Video Input → Frame Extraction → Caption Generation → Embedding → Pinecone Storage → Query Interface
```

### Components

#### 1. **Frame Extractor** (`frame_extractor.py`)
- Extracts frames from video files
- Implements similarity-based redundancy filtering
- Maintains timestamp mappings
- Typical reduction: 50-70% fewer frames

#### 2. **Caption Generator** (`caption_generator.py`)
- Uses BLIP model for image captioning
- Batch processing for efficiency
- Quality filtering and duplicate removal
- GPU-accelerated inference

#### 3. **Embedding Generator** (`embedding_generator.py`)
- Converts captions to dense vectors
- Uses sentence-transformers models
- Normalized embeddings for cosine similarity
- Support for query embedding

#### 4. **Pinecone Manager** (`pinecone_manager.py`)
- Handles vector database operations
- Serverless or pod-based deployment
- Metadata filtering and similarity search
- Batch upload capabilities

#### 5. **Main Engine** (`video_search_engine.py`)
- Orchestrates the entire pipeline
- Provides unified API
- Comprehensive error handling
- Performance monitoring

### Configuration Options

Edit `video_search_config.py` or use environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SIMILARITY_THRESHOLD` | 0.85 | Similarity threshold for frame filtering (0-1) |
| `MAX_FRAMES_PER_VIDEO` | 1000 | Maximum frames to extract per video |
| `FRAME_RESIZE_WIDTH` | 640 | Frame resize width (None for original) |
| `BLIP_BATCH_SIZE` | 8 | Batch size for caption generation |
| `EMBEDDING_BATCH_SIZE` | 32 | Batch size for embedding generation |
| `QUERY_TOP_K` | 10 | Number of search results to return |
| `QUERY_SIMILARITY_THRESHOLD` | 0.6 | Minimum similarity score for results |

### Advanced Usage

#### Batch Search
```python
queries = ["black bag", "yellow bottle", "person walking", "car driving"]
batch_results = engine.batch_search(queries, top_k=3)

for query, results in batch_results.items():
    print(f"\nQuery: '{query}' - Found {len(results)} results")
    for result in results[:2]:
        print(f"  {result['time_formatted']} (score: {result['similarity_score']:.3f})")
```

#### Filtered Search
```python
# Search within specific time window
results = engine.search(
    query="person running",
    time_window=(10.0, 60.0),  # Search between 10-60 seconds
    video_filter="specific_video_name"
)
```

#### Index Management
```python
# Get index statistics
stats = engine.get_index_stats()
print(f"Total vectors: {stats['total_vectors']}")

# Clear index
engine.clear_index()

# Clean up resources
engine.cleanup()
```

## 🔧 Performance Optimization

### Memory Management
- Process videos in chunks for large files
- Adjust batch sizes based on available GPU memory
- Clear GPU cache between operations

### Speed Improvements
- Use smaller embedding models for faster inference
- Implement frame caching for repeated processing
- Adjust frame extraction threshold for fewer frames
- Use Pinecone serverless for faster queries

### Accuracy Enhancement
- Use larger BLIP models for better captions
- Generate multiple caption variants per frame
- Lower similarity threshold for frame extraction
- Use cross-encoder models for re-ranking

## 📊 Expected Performance

| Metric | Typical Value |
|--------|--------------|
| Frame extraction | 100-200 fps |
| Caption generation | 5-10 frames/sec (GPU) |
| Embedding generation | 100+ captions/sec |
| Query latency | <200ms |
| Frame reduction | 50-70% |
| Storage per hour of video | ~5-10MB in Pinecone |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `BLIP_BATCH_SIZE` in configuration
   - Set `FRAME_RESIZE_WIDTH` to smaller value (480 or 320)
   - Use CPU mode by setting `USE_GPU=False`

2. **Pinecone connection error**
   - Verify API key is correct
   - Check internet connection
   - Ensure index name doesn't contain invalid characters

3. **Poor search results**
   - Lower `QUERY_SIMILARITY_THRESHOLD`
   - Process video with lower `FRAME_SIMILARITY_THRESHOLD`
   - Try different query phrasings

4. **Slow processing**
   - Enable GPU acceleration
   - Increase batch sizes if memory allows
   - Use faster embedding model (all-MiniLM-L6-v2)

## 📈 Scalability

### Production Deployment

1. **Video Processing Pipeline**
   - Implement queue-based processing (Redis/RabbitMQ)
   - Use cloud storage for videos (S3/GCS)
   - Deploy workers on GPU instances

2. **Database Scaling**
   - Use Pinecone Pod for dedicated resources
   - Implement namespaces for multi-tenancy
   - Cache frequent queries

3. **API Service**
   - Deploy as REST API using FastAPI/Flask
   - Implement rate limiting
   - Add authentication/authorization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [BLIP](https://github.com/salesforce/BLIP) by Salesforce Research
- [Pinecone](https://www.pinecone.io) for vector database
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [OpenCV](https://opencv.org/) for video processing

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a research/educational project. Ensure you have appropriate rights to process and index video content.