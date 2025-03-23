import torch
import requests
import io
import json
import base64
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor



import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from util.utils import prepare_scheduler
from PIL import Image

class Models:
    def __init__(self):
        self.segment_processor = None
        self.segment_model = None
        self.inpainting_pipeline = None
        self.depth_model = None
        self.normal_estimator = None
        self.device = None

    def initialize(self, device):
        self.device = device
        # self.segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        # self.segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
        
        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            safety_checker=None,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.inpainting_pipeline.scheduler = DDIMScheduler.from_config(self.inpainting_pipeline.scheduler.config)
        self.inpainting_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self.inpainting_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        
        # self.depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16).to(device)
        # self.depth_model.scheduler = EulerDiscreteScheduler.from_config(self.depth_model.scheduler.config)
        # self.depth_model.scheduler = prepare_scheduler(self.depth_model.scheduler)
        
        # self.normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        #     "prs-eth/marigold-normals-v0-1", 
        #     torch_dtype=torch.bfloat16
        # ).to(device)

# Global model instance
models = Models()
models.initialize('cuda')


SERVER_URL = 'http://localhost:5000'

def serialize_for_json(obj):
    """Convert any object to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.dim() == 4:
            obj = obj.squeeze(0)
        buffer = io.BytesIO()
        tensor_np = obj.detach().cpu().numpy()
        np.save(buffer, tensor_np)
        return {
            "__type__": "tensor",
            "data": base64.b64encode(buffer.getvalue()).decode('utf-8')
        }
    elif hasattr(obj, 'numpy'):  # Handle other array-like objects
        return serialize_for_json(torch.tensor(obj))
    elif hasattr(obj, 'item'):  # Handle scalar tensors
        return obj.item()
    elif str(type(obj)).startswith("<class 'PIL.Image."):  # Handle PIL Images
        buffer = io.BytesIO()
        obj.save(buffer, format='PNG')
        return {
            "__type__": "image",
            "data": base64.b64encode(buffer.getvalue()).decode('utf-8')
        }
    else:
        try:
            # Try default JSON serialization
            json.dumps(obj)
            return obj
        except:
            # Fall back to string representation
            return str(obj)

def deserialize_from_json(obj):
    """Convert JSON-serialized data back to appropriate Python/PyTorch objects"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        if "__type__" in obj:
            if obj["__type__"] == "tensor":
                buffer = io.BytesIO(base64.b64decode(obj["data"]))
                tensor_np = np.load(buffer)
                return torch.from_numpy(tensor_np)
            elif obj["__type__"] == "tensorbf16":
                buffer = io.BytesIO(base64.b64decode(obj["data"]))
                tensor_np = np.load(buffer)
                ret = torch.from_numpy(tensor_np)
                return ret.to(torch.bfloat16)
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(x) for x in obj]
    return obj

def pil_to_base64(image):
    """Convert PIL Image to base64 string"""
    return serialize_for_json(image)

def base64_to_pil(b64_data):
    """Convert base64 data back to PIL Image"""
    return deserialize_from_json(b64_data)

class ModelCalls:
    @staticmethod
    def call_inpainting_pipeline(prompt="", negative_prompt=None, image=None, mask_image=None, 
                               num_inference_steps=None, guidance_scale=7.5, height=None, width=None, 
                               self_guidance=None, inpaint_mask=None, rendered_image=None):
        try:
            data = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'image': serialize_for_json(image) if image is not None else None,
                'mask_image': serialize_for_json(mask_image) if mask_image is not None else None,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'height': height,
                'width': width,
                'self_guidance': self_guidance,
                'inpaint_mask': serialize_for_json(inpaint_mask) if inpaint_mask is not None else None,
                'rendered_image': serialize_for_json(rendered_image) if rendered_image is not None else None,
            }
            response = requests.post(f'{SERVER_URL}/inpainting', json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if 'error' in response_data:
                raise RuntimeError(f"Server error: {response_data['error']}")
            
            # Get device from first available tensor input
            device = None
            for tensor_input in [image, mask_image, inpaint_mask, rendered_image]:
                if isinstance(tensor_input, torch.Tensor):
                    device = tensor_input.device
                    break
            if device is None:
                device = torch.device('cuda')
                
            # Deserialize and ensure result is on correct device
            result = deserialize_from_json(response_data)
            if isinstance(result, torch.Tensor):
                result = result.to(device)
            return result
        except Exception as e:
            print(f"Error in call_inpainting_pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    @staticmethod
    def call_inpainting_pipeline(prompt="", negative_prompt=None, image=None, mask_image=None, 
                               num_inference_steps=None, guidance_scale=7.5, height=None, width=None, 
                               self_guidance=None, inpaint_mask=None, rendered_image=None):
        
        ret = models.inpainting_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            self_guidance=self_guidance,
            inpaint_mask=inpaint_mask,
            rendered_image=rendered_image,
        ).images[0]

        return ret.to(torch.float32).to(torch.bfloat16)

    @staticmethod
    def call_depth_model(image, denoising_steps=30, guidance_steps=8, depth_conditioning=None,
                        target_depth=None, mask_align=None, mask_farther=None, logger=None):
        try:
            # Normalize mask shapes to [1, 1, H, W]
            def normalize_mask(mask):
                if mask is None:
                    return None
                if not isinstance(mask, torch.Tensor):
                    return mask
                if mask.dim() == 3:  # [C, H, W]
                    mask = mask.unsqueeze(0)  # Add batch dim
                if mask.shape[1] != 1:  # If channel dim is not 1
                    mask = mask.unsqueeze(1)  # Add channel dim
                return mask
            
            # Normalize masks before serialization
            mask_align = normalize_mask(mask_align)
            mask_farther = normalize_mask(mask_farther)
            
            data = {
                'image': serialize_for_json(image),
                'denoising_steps': denoising_steps,
                'guidance_steps': guidance_steps,
                'depth_conditioning': serialize_for_json(depth_conditioning) if depth_conditioning is not None else None,
                'target_depth': serialize_for_json(target_depth) if target_depth is not None else None,
                'mask_align': serialize_for_json(mask_align) if mask_align is not None else None,
                'mask_farther': serialize_for_json(mask_farther) if mask_farther is not None else None,
            }
            response = requests.post(f'{SERVER_URL}/depth', json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if 'error' in response_data:
                raise RuntimeError(f"Server error: {response_data['error']}")
            
            # Get device from input tensor if it's a tensor, otherwise use CUDA
            device = image.device if isinstance(image, torch.Tensor) else torch.device('cuda')
            
            # Return the processed depth result on CUDA
            result = deserialize_from_json(response_data)
            depth = result['depth']
            if isinstance(depth, torch.Tensor):
                depth = depth.to(device)  # This should be CUDA
            return depth
        except Exception as e:
            print(f"Error in call_depth_model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def call_normal_estimator(image):
        try:
            # Get device from input tensor if it's a tensor, otherwise use CUDA
            device = image.device if isinstance(image, torch.Tensor) else torch.device('cuda')
            
            data = {
                'image': serialize_for_json(image),
            }
            response = requests.post(f'{SERVER_URL}/normal', json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if 'error' in response_data:
                raise RuntimeError(f"Server error: {response_data['error']}")
            
            result = deserialize_from_json(response_data)
            normal = result['normal']
            if isinstance(normal, torch.Tensor):
                normal = normal.to(device)
            return normal
        except Exception as e:
            print(f"Error in call_normal_estimator: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def call_segmentation_model(image, target_sizes=None):
        if target_sizes is None:
            target_sizes = [(512, 512)]
        try:
            # Convert tensor to base64
            if isinstance(image, torch.Tensor):
                image_data = image
            elif isinstance(image, Image.Image):
                image_data = ToTensor()(image)
            else:
                raise TypeError(f"Expected PIL Image or torch Tensor, got {type(image)}")
            
            data = {
                'image': serialize_for_json(image_data),
                'target_sizes': target_sizes
            }
            response = requests.post(f'{SERVER_URL}/segmentation', json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if 'error' in response_data:
                raise RuntimeError(f"Server error: {response_data['error']}")
            
            # Get device from input tensor if it's a tensor, otherwise use CUDA
            device = image.device if isinstance(image, torch.Tensor) else torch.device('cuda')
            
            # Return the processed segmentation result on CUDA
            result = deserialize_from_json(response_data['result'])
            if isinstance(result, torch.Tensor):
                result = result.to(device)  # This should be CUDA
            return result
        except Exception as e:
            print(f"Error in call_segmentation_model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# class ModelCalls:
    # @staticmethod
    # def call_inpainting_pipeline(prompt="", negative_prompt=None, image=None, mask_image=None, 
    #                            num_inference_steps=None, guidance_scale=7.5, height=None, width=None, 
    #                            self_guidance=None, inpaint_mask=None, rendered_image=None):
    #     return models.inpainting_pipeline(
    #         prompt=prompt,
    #         negative_prompt=negative_prompt,
    #         image=image,
    #         mask_image=mask_image,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         height=height,
    #         width=width,
    #         self_guidance=self_guidance,
    #         inpaint_mask=inpaint_mask,
    #         rendered_image=rendered_image,
    #     ).images[0]

#     @staticmethod
#     def call_depth_model(image, denoising_steps=30, guidance_steps=8, depth_conditioning=None,
#                         target_depth=None, mask_align=None, mask_farther=None, logger=None):
#         image_input = (image*255).byte().squeeze().permute(1, 2, 0)
#         image_input = Image.fromarray(image_input.cpu().numpy())
#         depth = models.depth_model(
#             image_input,
#             denoising_steps=denoising_steps,
#             ensemble_size=1,
#             processing_res=0,
#             match_input_res=True,
#             batch_size=0,
#             color_map=None,
#             show_progress_bar=True,
#             depth_conditioning=depth_conditioning,
#             target_depth=target_depth,
#             mask_align=mask_align,
#             mask_farther=mask_farther,
#             guidance_steps=guidance_steps,
#             logger=logger,
#         )
#         depth = depth[None, None, :].to(dtype=torch.float32)
#         depth /= 200
#         return depth

#     @staticmethod
#     def call_normal_estimator(image):
#         return models.normal_estimator(
#             image * 2 - 1,
#             num_inference_steps=10,
#             processing_res=768,
#             output_prediction_format='pt',
#         ).to(dtype=torch.float32)  # [1, 3, H, W], [-1, 1]

#     @staticmethod
#     def call_segmentation_model(image):
#         inputs = models.segment_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
#         outputs = models.segment_model(**inputs)
#         return outputs
