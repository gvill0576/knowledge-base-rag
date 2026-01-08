"""
AWS Bedrock embeddings configuration.

This module initializes the embedding model used to convert
text into numerical vectors for semantic search.
"""

import os
from langchain_aws import BedrockEmbeddings
import boto3
from dotenv import load_dotenv


def create_embeddings():
    """
    Create BedrockEmbeddings instance for converting text to vectors.
    
    Uses AWS Bedrock's Titan Embed Text v2 model which creates
    1536-dimensional vectors that capture semantic meaning.
    
    Returns:
        BedrockEmbeddings: Configured embeddings model
        
    Example:
        >>> embeddings = create_embeddings()
        >>> vector = embeddings.embed_query("Hello world")
        >>> len(vector)
        1536
    """
    # Load environment variables
    load_dotenv()
    
    # Set AWS profile if specified
    if os.getenv("AWS_PROFILE"):
        os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE")
    
    # Create Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    
    # Create embeddings instance
    embeddings = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0"
    )
    
    print("✅ Embeddings model initialized (Titan Embed Text v2)")
    return embeddings


if __name__ == "__main__":
    # Test embeddings creation
    print("Testing embeddings creation...")
    embeddings = create_embeddings()
    
    # Test embedding a sample text
    test_text = "This is a test sentence."
    vector = embeddings.embed_query(test_text)
    
    print(f"✅ Successfully created embedding")
    print(f"   Text: '{test_text}'")
    print(f"   Vector dimensions: {len(vector)}")
    print(f"   First 5 values: {vector[:5]}")
