from PIL import Image
import requests
import io
import base64
import numpy as np
import torch

class ModelCallsClient:
    API_BASE_URL = "http://localhost:8000"  # Change this to your API server URL
    
    @staticmethod
    def _tensor_to_base64(tensor):
        """Convert tensor to base64 string"""
        if tensor is None:
            return None
            
        # Convert tensor to PIL Image
        if tensor.dim() == 4:  # BCHW format
            tensor = tensor.squeeze(0)  # Remove batch dimension
        if tensor.dim() == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC
            
        # Scale to [0, 255] and convert to uint8
        tensor_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(tensor_np)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    @staticmethod
    def _base64_to_tensor(base64_str):
        """Convert base64 string to tensor"""
        if base64_str is None:
            return None
            
        # Decode base64 to image
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to tensor
        img_np = np.array(image)
        if len(img_np.shape) == 2:  # Single channel (depth map)
            tensor = torch.from_numpy(img_np).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W]
        else:  # RGB image
            tensor = torch.from_numpy(img_np).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
        return tensor
    
    @staticmethod
    def _base64_to_normal_tensor(base64_str):
        """Convert base64 string to normal map tensor (special case for normal maps)"""
        if base64_str is None:
            return None
            
        # Decode base64 to image
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to tensor (normal maps are in range [-1, 1])
        img_np = np.array(image)
        tensor = torch.from_numpy(img_np).float() / 127.5 - 1.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
        return tensor
    
    @classmethod
    def call_inpainting_pipeline(cls, prompt="", negative_prompt=None, image=None, mask_image=None, 
                               num_inference_steps=None, guidance_scale=7.5, height=None, width=None, 
                               self_guidance=None, inpaint_mask=None, rendered_image=None):
        """Call inpainting API endpoint"""
        # Convert tensors to base64
        image_base64 = cls._tensor_to_base64(image)
        mask_image_base64 = cls._tensor_to_base64(mask_image)
        
        # Prepare request data
        data = {
            "prompt": prompt,
            "image": image_base64,
            "mask_image": mask_image_base64,
            "guidance_scale": guidance_scale,
            "self_guidance": self_guidance if self_guidance is not None else False
        }
        
        # Add optional parameters
        if negative_prompt is not None:
            data["negative_prompt"] = negative_prompt
        if num_inference_steps is not None:
            data["num_inference_steps"] = num_inference_steps
        if height is not None:
            data["height"] = height
        if width is not None:
            data["width"] = width
        
        # Make API request
        response = requests.post(f"{cls.API_BASE_URL}/inpainting", data=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Convert response to tensor
        tensor = cls._base64_to_tensor(base64.b64encode(response.content).decode())
        
        # Return the tensor directly to match original interface
        return tensor

    @classmethod
    def call_depth_model(cls, image, denoising_steps=30, guidance_steps=8, depth_conditioning=None,
                        target_depth=None, mask_align=None, mask_farther=None, logger=None):
        """Call depth estimation API endpoint"""
        # Convert tensors to base64
        image_base64 = cls._tensor_to_base64(image)
        depth_conditioning_base64 = cls._tensor_to_base64(depth_conditioning)
        target_depth_base64 = cls._tensor_to_base64(target_depth)
        mask_align_base64 = cls._tensor_to_base64(mask_align)
        mask_farther_base64 = cls._tensor_to_base64(mask_farther)
        
        # Prepare request data
        data = {
            "image": image_base64,
            "denoising_steps": denoising_steps,
            "guidance_steps": guidance_steps
        }
        
        # Add optional parameters
        if depth_conditioning_base64 is not None:
            data["depth_conditioning"] = depth_conditioning_base64
        if target_depth_base64 is not None:
            data["target_depth"] = target_depth_base64
        if mask_align_base64 is not None:
            data["mask_align"] = mask_align_base64
        if mask_farther_base64 is not None:
            data["mask_farther"] = mask_farther_base64
        
        # Make API request
        response = requests.post(f"{cls.API_BASE_URL}/depth", data=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Convert response to tensor
        depth_tensor = cls._base64_to_tensor(base64.b64encode(response.content).decode())
        
        # Adjust tensor to match original format
        depth_tensor = depth_tensor * 200  # Scale back to original range
        
        return depth_tensor

    @classmethod
    def call_normal_estimator(cls, image):
        """Call normal map estimation API endpoint"""
        # Convert tensor to base64
        image_base64 = cls._tensor_to_base64(image)
        
        # Make API request
        response = requests.post(f"{cls.API_BASE_URL}/normal", data={"image": image_base64})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Convert response to tensor (normal maps are in range [-1, 1])
        normal_tensor = cls._base64_to_normal_tensor(base64.b64encode(response.content).decode())
        
        return normal_tensor  # [1, 3, H, W], [-1, 1]

    @classmethod
    def call_segmentation_model(cls, image):
        """Call segmentation API endpoint"""
        # Convert tensor to base64
        image_base64 = cls._tensor_to_base64(image)
        
        # Make API request
        response = requests.post(f"{cls.API_BASE_URL}/segment_model", data={"image": image_base64})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        result = response.json()
        
        # Create a simple object to match the original output format
        class SegmentationOutput:
            def __init__(self, predictions, labels):
                self.class_queries_logits = torch.tensor(predictions)
                self.labels = labels
        
        return SegmentationOutput(result["predictions"], result["labels"])

    @classmethod
    def call_segmentation_processor(cls, image):
        """Call segmentation processor API endpoint"""
        # Convert tensor to base64
        image_base64 = cls._tensor_to_base64(image)
        
        # Make API request
        response = requests.post(f"{cls.API_BASE_URL}/segment_processor", data={"image": image_base64})
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        result = response.json()
        
        # Extract processor outputs
        processor_outputs = result["processor_outputs"]
        
        # Convert lists back to tensors where appropriate
        processed_outputs = {}
        for key, value in processor_outputs.items():
            if isinstance(value, list):
                processed_outputs[key] = torch.tensor(value)
            else:
                processed_outputs[key] = value
        
        return processed_outputs

    @classmethod
    def call_segmentation_post_process(cls, segment_output, image):
        """Call segmentation post-processing API endpoint"""
        # Convert image to base64 if it's a tensor
        if isinstance(image, torch.Tensor):
            image_base64 = cls._tensor_to_base64(image)
        else:
            # If it's a PIL image, convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare the segment output data
        segment_data = {}
        if hasattr(segment_output, 'class_queries_logits'):
            segment_data["class_queries_logits"] = segment_output.class_queries_logits.detach().cpu().numpy().tolist()
        
        # Make API request
        response = requests.post(
            f"{cls.API_BASE_URL}/segment_post_process", 
            json={
                "image": image_base64,
                "segment_output": segment_data
            }
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response and convert to tensor
        result = response.json()
        pred_semantic_map = torch.tensor(result["pred_semantic_map"])
        
        return pred_semantic_map