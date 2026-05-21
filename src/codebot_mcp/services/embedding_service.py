from typing import List
import time
import voyageai
from codebot_mcp.config import settings
import logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Voyage AI API"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = None
        
        # Rate limiting for Voyage AI
        # Basic tier: 2000 RPM, 3M TPM
        self.max_rpm = 2000  # Requests per minute
        self.request_times = []  # Track request timestamps
        self.min_request_interval = 60.0 / self.max_rpm  # Minimum seconds between requests
    
    def _get_client(self):
        """Get or create Voyage AI client"""
        if self.client is None:
            if not settings.VOYAGE_API_KEY:
                raise ValueError("VOYAGE_API_KEY not set in environment variables")
            self.client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
            logger.info("Initialized Voyage AI client with model: %s", self.model_name)
        return self.client
    
    def _wait_for_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.
        
        Voyage AI limits:
        - Basic: 2000 RPM (requests per minute)
        - Basic: 3M TPM (tokens per minute)
        
        We enforce RPM limit with a sliding window.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60.0]
        
        # If we've made too many requests in the last minute, wait
        if len(self.request_times) >= self.max_rpm:
            # Wait until the oldest request is more than 60 seconds old
            oldest_request = self.request_times[0]
            wait_time = 60.0 - (current_time - oldest_request)
            if wait_time > 0:
                logger.debug("Rate limit: waiting %.2fs to respect %d RPM limit", wait_time, self.max_rpm)
                time.sleep(wait_time)
                current_time = time.time()
        
        # Also enforce minimum interval between requests (smoother distribution)
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                time.sleep(wait_time)
                current_time = time.time()
        
        # Record this request
        self.request_times.append(current_time)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using Voyage API"""
        self._wait_for_rate_limit()
        
        client = self._get_client()
        result = client.embed(
            texts=[text],
            model=self.model_name,
            input_type="document",
            output_dimension=self.dimension
        )
        return result.embeddings[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Voyage API"""
        if not texts:
            return []
        
        client = self._get_client()

        all_embeddings = []
        chunk_size = min(128, settings.BATCH_SIZE)

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            self._wait_for_rate_limit()
            all_embeddings.extend(self._embed_chunk(client, chunk))

        return all_embeddings

    def _embed_chunk(self, client, texts: List[str]) -> list:
        """Embed a chunk, splitting in half recursively if the token limit is hit."""
        try:
            result = client.embed(
                texts=texts,
                model=self.model_name,
                input_type="document",
                output_dimension=self.dimension,
            )
            return result.embeddings
        except Exception as e:
            if "max allowed tokens" in str(e) and len(texts) > 1:
                logger.warning(
                    "Token limit exceeded for batch of %d — splitting in half", len(texts)
                )
                mid = len(texts) // 2
                self._wait_for_rate_limit()
                left = self._embed_chunk(client, texts[:mid])
                self._wait_for_rate_limit()
                right = self._embed_chunk(client, texts[mid:])
                return left + right
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses input_type="query" for better search performance.
        """
        self._wait_for_rate_limit()
        
        client = self._get_client()
        result = client.embed(
            texts=[query],
            model=self.model_name,
            input_type="query",  # Optimized for queries
            output_dimension=self.dimension
        )
        return result.embeddings[0]
    
    RERANK_INSTRUCTION = (
        "Instruction: A developer is searching a codebase. The documents are Python function "
        "signatures and docstrings. The codebase has distinct parts listed below that represent "
        "separate concerns. Score each function by how well it matches the query, keeping in "
        "mind which part of the codebase it belongs to. If the query mentions a specific name, "
        "prefer functions whose file path or class name contains that name.\n"
    )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[dict]:
        """
        Rerank documents against a query using Voyage rerank-2.5.

        Prepends a code-search instruction to the query so the model
        prioritises functions that implement the described behavior.

        Args:
            query: The search query
            documents: List of document strings to rerank
            top_k: Optional; return only top N results
            instruction: Optional; custom instruction to prepend instead
                of the default RERANK_INSTRUCTION

        Returns:
            List of dicts with keys: index, relevance_score, document
            Sorted by relevance_score descending.
        """
        self._wait_for_rate_limit()
        client = self._get_client()
        prefix = instruction if instruction is not None else self.RERANK_INSTRUCTION
        instructed_query = f"{prefix}Query: {query}"
        result = client.rerank(
            query=instructed_query,
            documents=documents,
            model="rerank-2.5",
            top_k=top_k
        )
        return [
            {"index": r.index, "relevance_score": r.relevance_score, "document": r.document}
            for r in result.results
        ]

    def prepare_function_text(self, function_data: dict) -> str:
        """
        Prepare function text for embedding in natural Python format.
        Code field already contains: def line, docstring, body.
        """
        parts = []

        # File path as comment
        parts.append(f"# {function_data['file_path']}")
        if function_data.get('module_docstring'):
            parts.append(f'"""{function_data["module_docstring"]}"""')
        parts.append("")  # Empty line
        
        # Get the code
        code = function_data.get('code', '')
        
        # If it's a class method, wrap in class declaration
        if function_data.get('class_name'):
            parts.append(f"class {function_data['class_name']}:")
            if function_data.get('class_docstring'):
                parts.append(f'    """{function_data["class_docstring"]}"""')

            # Indent the entire function code (add 4 spaces to each line)
            indented_code = '\n'.join(
                f"    {line}" if line.strip() else ""  # Don't indent empty lines
                for line in code.split('\n')
            )
            parts.append(indented_code)
        else:
            # Standalone function - use code as-is
            parts.append(code)
        
        
        return "\n".join(parts)

# Singleton instance
embedding_service = EmbeddingService()