import requests
from PIL import Image
import io
import base64
import dotenv
import os

dotenv.load_dotenv()

def upload_to_imgbb(image_path):
    # Open and read the image
    img = Image.open(image_path)
    
    # Convert to PNG and save to buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    
    # Upload to imgbb
    # Note: You need to replace YOUR_API_KEY with an actual imgbb API key
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": os.getenv("IMGBB_API_KEY"),
        "image": img_str
    }
    
    response = requests.post(url, payload)
    
    # Print response
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200 and response.json()['success']:
        return response.json()['data']['url']
    return None

if __name__ == "__main__":
    image_path = "examples/images/animation_lakeside_1.png"
    print("Note: You need to set your imgbb API key in the script first!")
    url = upload_to_imgbb(image_path)
    if url:
        print(f"\nSuccessfully uploaded image!")
        print(f"URL: {url}")
    else:
        print("\nFailed to upload image")
