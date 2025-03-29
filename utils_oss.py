"""
Utility functions for Aliyun OSS operations.
"""

import os
import requests
import dotenv
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from functools import lru_cache

# Load environment variables
dotenv.load_dotenv(".env")

@lru_cache
def get_oss_bucket():
    """Get OSS bucket with cached connection."""
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    endpoint = "https://oss-ap-southeast-1.aliyuncs.com"
    region = "ap-southeast-1"
    bucket_name = "midreal-ai"
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    return bucket

def upload_to_oss(content, task_id: str, prefix: str = "img", extension: str = "png") -> str:
    """Upload content to OSS.
    
    Args:
        content: Either a URL string to download from, or binary data to upload directly
        task_id (str): Task ID for the file
        prefix (str): OSS prefix/directory for the file
        extension (str): File extension (e.g., 'png', 'mp3')
        
    Returns:
        str: OSS URL of the uploaded file
        
    Raises:
        Exception: If download or upload fails
    """
    # If content is a URL, download it first
    if isinstance(content, str):
        response = requests.get(content)
        if response.status_code != 200:
            raise Exception(f"Failed to download from {content}")
        data = response.content
    else:
        # Assume content is binary data
        data = content
    
    # Upload to OSS
    bucket = get_oss_bucket()
    object_name = f"{prefix}/{task_id}.{extension}"
    result = bucket.put_object(object_name, data)
    
    if result.status != 200:
        raise Exception("Failed to upload to OSS")
        
    return f"https://midreal-ai.oss-ap-southeast-1.aliyuncs.com/{object_name}"
