import requests
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def download_and_encode_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    raise Exception(f"Failed to download image from {url}")

# API endpoint
API_BASE = "https://api.us1.bfl.ai"
API_KEY = os.getenv('BFL_API_KEY')

# Input parameters matching the original example
image_url = "https://replicate.delivery/pbxt/HtGQBfA5TrqFYZBf0UL18NTqHrzt8UiSIsAkUuMHtjvFDO6p/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://replicate.delivery/pbxt/HtGQBqO9MtVbPm0G0K43nsvvjBB0E0PaWOhuNRrRBBT4ttbf/mask.png"

# Download and encode images
print("Downloading and encoding images...")
image_base64 = download_and_encode_image(image_url)
mask_base64 = download_and_encode_image(mask_url)

# Request payload
payload = {
    "image": image_base64,
    "mask": mask_base64,
    "prompt": "small cute cat sat on a park bench",
    "steps": 30,
    "prompt_upsampling": False,
    "guidance": 7.0,  # Similar to guidance_scale in original
    "output_format": "png",
    "safety_tolerance": 2
}

headers = {
    "Content-Type": "application/json",
    "x-key": API_KEY
}

# Submit the generation task
response = requests.post(
    f"{API_BASE}/v1/flux-pro-1.0-fill",
    headers=headers,
    json=payload
)

if response.status_code == 200:
    task_data = response.json()
    task_id = task_data["id"]
    polling_url = task_data["polling_url"]
    
    # Poll for results
    while True:
        result_response = requests.get(
            f"{API_BASE}/v1/get_result",
            params={"id": task_id},
            headers=headers
        )
        
        if result_response.status_code == 200:
            result_data = result_response.json()
            status = result_data["status"]
            
            if status == "Ready":
                print("Generation completed!")
                result = result_data["result"]
                print("Result:", result)
                
                # Download the result image
                if result and 'sample' in result:
                    print("Downloading result image...")
                    img_response = requests.get(result['sample'])
                    if img_response.status_code == 200:
                        output_path = Path('output') / f'result_{int(time.time())}.png'
                        output_path.write_bytes(img_response.content)
                        print(f"Image saved to: {output_path}")
                    else:
                        print(f"Failed to download result image: {img_response.status_code}")
                break
            elif status in ["Error", "Request Moderated", "Content Moderated"]:
                print(f"Task failed with status: {status}")
                if result_data.get("details"):
                    print("Details:", result_data["details"])
                break
            else:
                print(f"Status: {status}, Progress: {result_data.get('progress', 'unknown')}")
                time.sleep(2)  # Wait before polling again
else:
    print(f"Error submitting task: {response.status_code}")
    print(response.text)
