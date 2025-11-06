"""
Text and Image Embedding Generation Module
Converts captions into dense vectors and images into CLIP vectors for multi-modal search
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import logging
from typing import List, Optional, Union, Tuple, Dict
from tqdm import tqdm
import gc
from dataclasses import dataclass
from caption_generator import CaptionedFrame
from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddedFrame:
    """Data structure for frame with caption and embedding"""
    captioned_frame: CaptionedFrame
    embedding: np.ndarray
    embedding_id: str

class TextEmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 normalize: bool = True):
        """
        Initialize the text embedding generator
        
        Args:
            model_name: Sentence transformer model name
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU if available
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load sentence transformer model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, 
                          captioned_frames: List[CaptionedFrame],
                          show_progress: bool = True) -> List[EmbeddedFrame]:
        """
        Generate embeddings for captioned frames
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            show_progress: Whether to show progress bar
            
        Returns:
            List of EmbeddedFrame objects
        """
        if not captioned_frames:
            return []
        
        logger.info(f"Generating embeddings for {len(captioned_frames)} captions")
        
        # Extract captions
        captions = [cf.caption for cf in captioned_frames]
        
        # Generate embeddings
        embeddings = self._encode_batch(captions, show_progress=show_progress)
        
        # Create EmbeddedFrame objects
        embedded_frames = []
        for cf, embedding in zip(captioned_frames, embeddings):
            # Generate unique embedding ID
            embedding_id = f"{cf.frame_data.frame_id}_emb"
            
            embedded_frame = EmbeddedFrame(
                captioned_frame=cf,
                embedding=embedding,
                embedding_id=embedding_id
            )
            embedded_frames.append(embedded_frame)
        
        logger.info(f"Generated {len(embedded_frames)} embeddings")
        return embedded_frames
    
    def _encode_batch(self, texts: List[str], 
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts into embeddings
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings [N, embedding_dim]
        """
        # Encode with sentence transformer
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query text
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        # Encode single query
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        # Ensure it's a 1D array
        if len(embedding.shape) > 1:
            embedding = embedding.squeeze()
        
        return embedding
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of query embeddings [N, embedding_dim]
        """
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray,
                          embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and embeddings
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embeddings to compare against
            
        Returns:
            Similarity scores
        """
        if self.normalize:
            # If normalized, use dot product (equivalent to cosine similarity)
            similarities = np.dot(embeddings, query_embedding)
        else:
            # Compute cosine similarity manually
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def find_similar(self,
                    query: str,
                    embedded_frames: List[EmbeddedFrame],
                    top_k: int = 10,
                    threshold: float = 0.5) -> List[Tuple[EmbeddedFrame, float]]:
        """
        Find similar frames to a query
        
        Args:
            query: Query string
            embedded_frames: List of EmbeddedFrame objects
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (EmbeddedFrame, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Extract embeddings from frames
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and create results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((embedded_frames[idx], score))
        
        return results
    
    def create_embedding_matrix(self, 
                               embedded_frames: List[EmbeddedFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Create embedding matrix and ID list for batch operations
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            
        Returns:
            Tuple of (embedding_matrix, embedding_ids)
        """
        embeddings = []
        ids = []
        
        for ef in embedded_frames:
            embeddings.append(ef.embedding)
            ids.append(ef.embedding_id)
        
        embedding_matrix = np.array(embeddings)
        
        return embedding_matrix, ids
    
    def prepare_for_pinecone(self, 
                           embedded_frames: List[EmbeddedFrame],
                           video_name: str = "video",
                           source_file_path: str = "") -> List[Tuple[str, List[float], Dict]]:
        """
        Prepare data for Pinecone upload
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            video_name: Name of the video for metadata
            
        Returns:
            List of (id, vector, metadata) tuples for Pinecone
        """
        pinecone_data = []
        
        for idx, ef in enumerate(embedded_frames):
            # Create unique ID that includes object index to avoid collisions
            # Format: frameID_objectIdx_emb
            unique_id = f"{ef.captioned_frame.frame_data.frame_id}_obj{idx}_emb"
            
            # Convert embedding to list
            vector = ef.embedding.tolist()
            
            # Prepare metadata
            metadata = {
                'timestamp': ef.captioned_frame.frame_data.timestamp,
                'caption': ef.captioned_frame.caption,
                'frame_id': ef.captioned_frame.frame_data.frame_id,
                'frame_index': ef.captioned_frame.frame_data.frame_index,
'video_name': video_name,
                'source_file_path': source_file_path
            }
            
            pinecone_data.append((unique_id, vector, metadata))
        
        return pinecone_data
    
    def augment_embeddings(self,
                          embedded_frames: List[EmbeddedFrame],
                          augmentation_factor: float = 0.1) -> List[EmbeddedFrame]:
        """
        Augment embeddings with small perturbations for better retrieval
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            augmentation_factor: Factor for perturbation (0.0 to 1.0)
            
        Returns:
            Augmented EmbeddedFrame objects
        """
        augmented = []
        
        for ef in embedded_frames:
            # Add small random perturbation
            noise = np.random.randn(*ef.embedding.shape) * augmentation_factor
            augmented_embedding = ef.embedding + noise
            
            # Re-normalize if needed
            if self.normalize:
                augmented_embedding = augmented_embedding / np.linalg.norm(augmented_embedding)
            
            # Create new EmbeddedFrame with augmented embedding
            augmented_ef = EmbeddedFrame(
                captioned_frame=ef.captioned_frame,
                embedding=augmented_embedding,
                embedding_id=f"{ef.embedding_id}_aug"
            )
            
            augmented.append(ef)  # Keep original
            augmented.append(augmented_ef)  # Add augmented
        
        return augmented
    
    def deduplicate_embeddings(self,
                              embedded_frames: List[EmbeddedFrame],
                              similarity_threshold: float = 0.95) -> List[EmbeddedFrame]:
        """
        Remove duplicate embeddings based on similarity threshold
        
        Args:
            embedded_frames: List of EmbeddedFrame objects
            similarity_threshold: Minimum similarity to consider as duplicate (0.0 to 1.0)
            
        Returns:
            List of unique EmbeddedFrame objects
        """
        if not embedded_frames:
            return []
        
        if len(embedded_frames) <= 1:
            return embedded_frames
        
        logger.info(f"Deduplicating {len(embedded_frames)} embeddings with threshold {similarity_threshold}")
        
        # Convert to numpy array for efficient computation
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        # Track which embeddings to keep
        keep_mask = np.ones(len(embedded_frames), dtype=bool)
        
        # Compare each embedding with subsequent ones
        for i in range(len(embeddings)):
            if not keep_mask[i]:
                continue
            
            # Compute similarity with all subsequent embeddings
            for j in range(i + 1, len(embeddings)):
                if not keep_mask[j]:
                    continue
                
                # Compute cosine similarity
                if self.normalize:
                    # If normalized, use dot product
                    similarity = np.dot(embeddings[i], embeddings[j])
                else:
                    # Compute cosine similarity manually
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                
                # Mark as duplicate if similarity exceeds threshold
                if similarity >= similarity_threshold:
                    keep_mask[j] = False
        
        # Filter embeddings based on keep mask
        unique_frames = [ef for ef, keep in zip(embedded_frames, keep_mask) if keep]
        
        removed_count = len(embedded_frames) - len(unique_frames)
        logger.info(f"Removed {removed_count} duplicate embeddings, kept {len(unique_frames)} unique")
        
        return unique_frames
    
    def get_embedding_statistics(self, embedded_frames: List[EmbeddedFrame]) -> Dict:
        """Get statistics about embeddings"""
        if not embedded_frames:
            return {"total": 0}
        
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        
        stats = {
            "total": len(embeddings),
            "dimension": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "mean_similarity": float(np.mean(np.dot(embeddings, embeddings.T)))
        }
        
        return stats
    
    def clear_cache(self):
        """Clear cache and free memory"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Cache cleared")
    
    def unload_model(self):
        """Unload model from memory"""
        del self.model
        self.clear_cache()
        logger.info("Embedding model unloaded")


class ClipMultiModalEncoder:
    """Encode images and texts into CLIP embedding space for multi-modal search."""
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 use_gpu: bool = True,
                 normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def encode_images(self, frames: List[FrameData], batch_size: int = 16) -> np.ndarray:
        if not frames:
            return np.zeros((0, self.get_dim()), dtype=np.float32)
        embeddings = []
        for i in tqdm(range(0, len(frames), batch_size), desc="CLIP img enc"):
            batch = frames[i:i+batch_size]
            images = [f.image for f in batch]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(img_feats.cpu().numpy())
        return np.vstack(embeddings)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dim()), dtype=np.float32)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP txt enc"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                txt_feats = self.model.get_text_features(**inputs)
                txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(txt_feats.cpu().numpy())
        return np.vstack(embeddings)

    def encode_query_text(self, query: str) -> np.ndarray:
        vec = self.encode_texts([query])
        return vec[0] if vec.shape[0] else np.zeros((self.get_dim(),), dtype=np.float32)

    def get_dim(self) -> int:
        # CLIP ViT-B/32 dimension is typically 512
        # Infer from model config if available
        try:
            return int(self.model.visual_projection.out_features)
        except Exception:
            return 512

    def unload(self):
        del self.model
        del self.processor
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


def extract_object_attribute_tags(caption: str) -> List[str]:
    """Simple heuristic tag extractor from caption: colors + nouns tokens."""
    if not caption:
        return []
    colors = {
        'black','white','gray','grey','red','orange','yellow','green','blue','purple','pink','brown','beige','tan','gold','silver','navy','maroon','teal'
    }
    tokens = [t.strip('.,;:!?').lower() for t in caption.split()]
    tags = [t for t in tokens if t in colors]
    # Add simple noun-like tokens (very naive fallback)
    for t in tokens:
        if t.isalpha() and len(t) > 2 and t not in tags and t not in colors:
            # Prefer a small set of common object words
            if t in {"person","man","woman","girl","boy","bag","backpack","bottle","phone","laptop","car","bike","bicycle","shirt","pants","shoes","jacket","hat","table"}:
                tags.append(t)
    # Deduplicate preserve order
    seen = set(); ordered = []
    for t in tags:
        if t not in seen:
            seen.add(t); ordered.append(t)
    return ordered


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding generator
    generator = TextEmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        use_gpu=True,
        normalize=True
    )
    
    # Example: Generate embeddings for captioned frames
    # from frame_extractor import VideoFrameExtractor
    # from caption_generator import BlipCaptionGenerator
    # 
    # # Extract frames
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # # Generate captions
    # caption_gen = BlipCaptionGenerator()
    # captioned_frames = caption_gen.generate_captions(frames)
    # 
    # # Generate embeddings
    # embedded_frames = generator.generate_embeddings(captioned_frames)
    # 
    # # Get statistics
    # stats = generator.get_embedding_statistics(embedded_frames)
    # print(f"Embedding statistics: {stats}")
    # 
    # # Test similarity search
    # query = "a person walking"
    # results = generator.find_similar(query, embedded_frames, top_k=5)
    # 
    # for ef, score in results:
    #     print(f"Score: {score:.3f} - Caption: {ef.captioned_frame.caption}")
    #     print(f"  Timestamp: {ef.captioned_frame.frame_data.timestamp:.2f}s")