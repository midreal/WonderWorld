from fastapi import FastAPI, Form
from fastapi.responses import Response
import io
from PIL import Image
import torch
import numpy as np
from model import ModelCalls, models
from typing import Optional, List
import json
import base64

app = FastAPI(title="WonderWorld API")

# 全局设备变量
device = None

# 在应用启动时初始化模型
@app.on_event("startup")
async def startup_event():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on device: {device}")
    models.initialize(device)
    print("Models initialized successfully")

def process_image(base64_str: str, is_mask: bool = False, target_size: Optional[tuple] = None) -> torch.Tensor:
    """Convert base64 string to tensor format"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image if target_size is provided
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    else:
        # Ensure dimensions are multiples of 64 for stable diffusion
        w, h = image.size
        w = ((w - 1) // 64 + 1) * 64
        h = ((h - 1) // 64 + 1) * 64
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert to tensor and normalize
    if is_mask:
        # For mask images, convert to binary mask
        if len(image_np.shape) == 3:
            image_np = image_np.mean(axis=2)  # Convert RGB to grayscale
        image_np = image_np > 127.5  # Convert to binary mask
        image_tensor = torch.from_numpy(image_np).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    else:
        # For regular images
        image_tensor = torch.from_numpy(image_np).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        image_tensor = image_tensor / 255.0  # Normalize to [0,1]
    
    # Always use CUDA if available, since models are on CUDA
    if torch.cuda.is_available():
        return image_tensor.cuda(non_blocking=True)
    return image_tensor

@app.post("/inpainting")
async def inpainting(
    prompt: str = Form(""),
    image: str = Form(...),
    mask_image: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: float = Form(7.5),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    self_guidance: Optional[bool] = Form(False),
):
    """Endpoint for image inpainting"""
    # Calculate target size
    if height and width:
        # Round to nearest multiple of 64
        height = ((height - 1) // 64 + 1) * 64
        width = ((width - 1) // 64 + 1) * 64
        target_size = (width, height)
    else:
        target_size = None
    
    # Process input image
    image_tensor = process_image(image, target_size=target_size)
    
    # Process mask image if provided
    if mask_image:
        # Use same size as input image
        mask_tensor = process_image(mask_image, is_mask=True, 
                                  target_size=(image_tensor.shape[3], image_tensor.shape[2]))
    else:
        mask_tensor = None
    
    result_tensor = ModelCalls.call_inpainting_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_tensor,
        mask_image=mask_tensor,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        self_guidance=self_guidance
    )
    
    # Convert tensor to PIL Image
    if isinstance(result_tensor, torch.Tensor):
        # Convert BFloat16 to Float32 before converting to numpy
        result_tensor = result_tensor.to(dtype=torch.float32)
        # Ensure correct dimension order: HWC for PIL Image
        if result_tensor.dim() == 4:  # BCHW format
            result_tensor = result_tensor.squeeze(0)  # Remove batch dimension
        if result_tensor.dim() == 3:  # CHW format
            result_tensor = result_tensor.permute(1, 2, 0)  # Convert to HWC
        # Convert to numpy and scale to [0, 255]
        result_np = (result_tensor.cpu().numpy() * 255).astype(np.uint8)
        result_image = Image.fromarray(result_np)
    else:
        # If it's already a PIL Image, use it directly
        result_image = result_tensor
    
    # Convert result to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/depth")
async def depth_estimation(
    image: str = Form(...),
    denoising_steps: int = Form(30),
    guidance_steps: int = Form(8),
    depth_conditioning: Optional[str] = Form(None),
    target_depth: Optional[str] = Form(None),
    mask_align: Optional[str] = Form(None),
    mask_farther: Optional[str] = Form(None)
):
    """Endpoint for depth estimation"""
    # Process input image
    image_tensor = process_image(image)
    
    # Process optional tensor inputs
    depth_cond = process_image(depth_conditioning) if depth_conditioning else None
    target = process_image(target_depth) if target_depth else None
    align_mask = process_image(mask_align, is_mask=True) if mask_align else None
    farther_mask = process_image(mask_farther, is_mask=True) if mask_farther else None
    
    depth_tensor = ModelCalls.call_depth_model(
        image=image_tensor,
        denoising_steps=denoising_steps,
        guidance_steps=guidance_steps,
        depth_conditioning=depth_cond,
        target_depth=target,
        mask_align=align_mask,
        mask_farther=farther_mask
    )
    
    # Convert depth map to image format (single channel)
    depth_tensor = depth_tensor.to(dtype=torch.float32)  # Convert from BFloat16 if needed
    if depth_tensor.dim() == 4:  # BCHW -> CHW
        depth_tensor = depth_tensor.squeeze(0)
    if depth_tensor.dim() == 3:  # CHW -> HW (for depth map)
        depth_tensor = depth_tensor.squeeze(0)
    depth_np = (depth_tensor.cpu().numpy() * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_np, mode='L')  # 'L' mode for single channel
    
    img_byte_arr = io.BytesIO()
    depth_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/normal")
async def normal_estimation(
    image: str = Form(...)
):
    """Endpoint for normal map estimation"""
    image_tensor = process_image(image)
    
    normal_tensor = ModelCalls.call_normal_estimator(image_tensor)
    
    # Convert normal map to image format
    # Normal map is in range [-1, 1], convert to [0, 255]
    normal_tensor = normal_tensor.to(dtype=torch.float32)  # Convert from BFloat16 if needed
    if normal_tensor.dim() == 4:  # BCHW -> CHW
        normal_tensor = normal_tensor.squeeze(0)
    normal_tensor = normal_tensor.permute(1, 2, 0)  # CHW -> HWC
    normal_np = ((normal_tensor.cpu().numpy() + 1) * 127.5).astype(np.uint8)
    normal_image = Image.fromarray(normal_np)
    
    img_byte_arr = io.BytesIO()
    normal_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.post("/segment_model")
async def segmentation(
    image: str = Form(...)
):
    """Endpoint for image segmentation"""
    # Process input image and ensure it's on the correct device
    image_tensor = process_image(image)
    
    # Get outputs from model
    outputs = ModelCalls.call_segmentation_model(image_tensor)
    
    # Move results to CPU for JSON serialization
    # OneFormer model returns class_queries_logits for semantic segmentation
    results = {
        "predictions": outputs.class_queries_logits.detach().cpu().numpy().tolist(),
        "labels": outputs.labels if hasattr(outputs, 'labels') else None
    }
    
    return results


@app.post("/segment_processor")
async def segmentation_processor(
    image: str = Form(...)
):
    """Endpoint for image segmentation preprocessing"""
    # Process input image and ensure it's on the correct device
    image_tensor = process_image(image)
    
    # Convert tensor to PIL Image for processor
    image_pil = Image.fromarray(
        (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    
    # Get processor outputs
    processor_outputs = ModelCalls.call_segmentation_processor(image_pil)
    
    # Convert tensors to serializable format
    serializable_outputs = {}
    for key, value in processor_outputs.items():
        if hasattr(value, 'detach'):
            # For tensor values, convert to list
            serializable_outputs[key] = value.detach().cpu().numpy().tolist()
        else:
            # For non-tensor values, keep as is
            serializable_outputs[key] = value
    
    return {"processor_outputs": serializable_outputs}

@app.post("/segment_post_process")
async def segmentation_post_process(request_data: dict):
    """Endpoint for post-processing semantic segmentation output"""
    # Extract image and segment output from request
    image_base64 = request_data.get("image")
    segment_output_data = request_data.get("segment_output", {})
    
    # Process the image to get PIL format
    image_tensor = process_image(image_base64)
    image_pil = Image.fromarray(
        (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    
    # Create a simple object to match the expected input format for post-processing
    class SegmentationOutput:
        def __init__(self, predictions):
            self.class_queries_logits = torch.tensor(predictions)
    
    # Create segment output object
    segment_output = SegmentationOutput(
        segment_output_data.get("class_queries_logits", [])
    )
    
    # Call the post-processing function
    pred_semantic_map = ModelCalls.call_segmentation_post_process(segment_output, image_pil)
    
    # Convert to serializable format
    return {"pred_semantic_map": pred_semantic_map.cpu().numpy().tolist()}