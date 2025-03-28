import requests
import json
import base64
from datetime import datetime
import os

# Server URL
SERVER_URL = "http://34.143.175.19:7779/generate"

# Request payload using the same parameters as the Replicate example
payload = {
    "mask": "https://replicate.delivery/pbxt/HtGQBqO9MtVbPm0G0K43nsvvjBB0E0PaWOhuNRrRBBT4ttbf/mask.png",
    "image": "https://replicate.delivery/pbxt/HtGQBfA5TrqFYZBf0UL18NTqHrzt8UiSIsAkUuMHtjvFDO6p/overture-creations-5sI6fQgYIuo.png",
    "width": 1024,
    "height": 1024,
    "prompt": "small cute cat sat on a park bench",
    "num_inference_steps": 30,
    "guidance_scale": 7.0,
    "num_outputs": 1,
    "output_format": "png"
}

try:
    # Make POST request to the server
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()  # Raise exception for non-200 status codes
    
    # Parse response
    result = response.json()
    
    # Save base64-encoded images
    for i, image_base64 in enumerate(result["images"]):
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Generate unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/generated_image_{timestamp}_{i}.png"
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"Saved generated image to: {filename}")

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except json.JSONDecodeError as e:
    print(f"Error parsing response: {e}")
