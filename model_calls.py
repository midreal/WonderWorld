from typing import List, Tuple, Union
import torch
import requests
import os
import tempfile
import dotenv
import base64
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from util.utils import prepare_scheduler
import io
from PIL import Image
import numpy as np
import time
import mask_utils
from utils_oss import upload_to_oss
import replicate

# Load environment variables
dotenv.load_dotenv()

class Models:
    """A container class that holds and initializes various AI models used in the application.
    
    This class manages the lifecycle and initialization of models for segmentation,
    inpainting, depth estimation, and normal map estimation.
    """
    
    def __init__(self):
        """Initialize the Models container with None values for all model attributes."""
        self.segment_processor = None  # OneFormer processor for segmentation
        self.segment_model = None      # OneFormer model for universal segmentation
        self.inpainting_pipeline = None # Stable Diffusion inpainting pipeline
        self.depth_model = None        # Marigold depth estimation model
        self.normal_estimator = None   # Marigold normal map estimation model
        self.device = None             # Device to run models on (cuda/cpu)

    def initialize(self, device: str) -> None:
        """Initialize all AI models and move them to the specified device.
        
        Args:
            device (str): The device to load models on ('cuda' or 'cpu')
            
        Note:
            - Loads OneFormer for ADE20K segmentation
            - Loads Stable Diffusion 2.0 for inpainting
            - Loads Marigold v1.0 for depth estimation
            - Loads Marigold v0.1 for normal map estimation
            - All models are loaded with bfloat16 precision where applicable
        """
        self.device = device
        self.segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
        
        # No need to store model ID for direct API access
        
        self.depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16).to(device)
        self.depth_model.scheduler = EulerDiscreteScheduler.from_config(self.depth_model.scheduler.config)
        self.depth_model.scheduler = prepare_scheduler(self.depth_model.scheduler)
        
        self.normal_estimator = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1", 
            torch_dtype=torch.bfloat16
        ).to(device)

# Global model instance
models = Models()

class ModelCalls:
    """Static interface for making inference calls to various AI models.
    
    This class provides a clean interface to interact with the models initialized in the Models class,
    handling all necessary pre-processing and post-processing of inputs and outputs.
    """
    
    @staticmethod
    def call_inpainting_pipeline(prompt: str = "", 
                               negative_prompt: str = None, 
                               image: Image.Image = None, 
                               mask_image: Image.Image = None,
                               num_inference_steps: int = None, 
                               guidance_scale: float = 7.5, 
                               height: int = None, 
                               width: int = None,
                               self_guidance: float = None, 
                               inpaint_mask: torch.Tensor = None, 
                               rendered_image: torch.Tensor = None) -> Image.Image:
        """Run the Stable Diffusion inpainting pipeline.
        
        Args:
            prompt (str): The text prompt to guide the image generation
            negative_prompt (str, optional): Text prompt for what not to generate
            image (PIL.Image): The source image to inpaint
            mask_image (PIL.Image): The mask indicating where to inpaint
            num_inference_steps (int, optional): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt (higher = closer)
            height (int, optional): Output image height
            width (int, optional): Output image width
            self_guidance (float, optional): Self-guidance scale factor
            inpaint_mask (torch.Tensor, optional): Alternative mask format
            rendered_image (torch.Tensor, optional): Pre-rendered image guide
            
        Returns:
            PIL.Image: The generated inpainted image
        """
        # Save input images to temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_image, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_feather_mask:
            try:
                print("Preparing image and mask...")
                # Save original images
                image.save(tmp_image.name, format='PNG')
                mask_image.save(tmp_mask.name, format='PNG')
                
                # Create feathered mask
                feather_mask = mask_utils.feather(tmp_mask.name)
                feather_mask.save(tmp_feather_mask.name, format='PNG')
                
                # Combine image with feathered mask
                img_w_mask = mask_utils.combine(tmp_image.name, tmp_feather_mask.name)
                
                # Upload to OSS
                print("Uploading image to OSS...")
                task_id = f"inpainting_{int(time.time())}"
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                img_w_mask.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                image_url = upload_to_oss(img_byte_arr, task_id, prefix="inpainting", extension="png")
                
                if not image_url:
                    raise RuntimeError("Failed to upload image to OSS")

                print("Image uploaded to OSS")
                    
            finally:
                # Cleanup temporary files
                for path in [tmp_image.name, tmp_mask.name, tmp_feather_mask.name]:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except OSError:
                        pass  # Ignore cleanup errors

        # Set default values if not provided
        if height is None:
            height = image.height if image else 512
        if width is None:
            width = image.width if image else 512
        if num_inference_steps is None:
            num_inference_steps = 30

        print("image_url", image_url)
        # write image_url to artifacts/inpainting_timestamp.png
        timestamp = int(time.time())
        inpaint_image_path = f"artifacts/inpainting_{timestamp}.png"
        response = requests.get(image_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download image from OSS, status code: {response.status_code}")
        with open(inpaint_image_path, "wb") as f:
            f.write(response.content)
        
        output = replicate.run(
            "black-forest-labs/flux-fill-pro",
            input={
                "image": image_url,
                "steps": num_inference_steps,
                "prompt": prompt,
                "guidance": guidance_scale,
                "outpaint": "None",
                "output_format": "png",
                "safety_tolerance": 2,
                "prompt_upsampling": False
            }
        )
        print("output", output)
        # Download the generated image
        response = requests.get(output)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download image from replicate, status code: {response.status_code}")
        
        # write image to artifacts/result_timestamp.png
        result_image_path = f"artifacts/result_{timestamp}.png"
        with open(result_image_path, "wb") as f:
            f.write(response.content)
        
        img = Image.open(io.BytesIO(response.content))
        return img

    @staticmethod
    def call_depth_model(image: torch.Tensor,
                        denoising_steps: int = 30,
                        guidance_steps: int = 8,
                        depth_conditioning: torch.Tensor = None,
                        target_depth: torch.Tensor = None,
                        mask_align: torch.Tensor = None,
                        mask_farther: torch.Tensor = None,
                        logger: object = None) -> torch.Tensor:
        """Estimate depth map from an input image using Marigold model.
        
        Args:
            image (torch.Tensor): Input image tensor [C, H, W] normalized to [0, 1]
            denoising_steps (int): Number of denoising steps
            guidance_steps (int): Number of guidance steps
            depth_conditioning (torch.Tensor, optional): Depth prior for conditioning
            target_depth (torch.Tensor, optional): Target depth values
            mask_align (torch.Tensor, optional): Mask for depth alignment
            mask_farther (torch.Tensor, optional): Mask for enforcing farther depth
            logger (object, optional): Logger object for progress tracking
            
        Returns:
            torch.Tensor: Estimated depth map normalized to [0, 1], shape [1, 1, H, W]
        """
        image_input = (image*255).byte().squeeze().permute(1, 2, 0)
        image_input = Image.fromarray(image_input.cpu().numpy())
        depth = models.depth_model(
            image_input,
            denoising_steps=denoising_steps,
            ensemble_size=1,
            processing_res=0,
            match_input_res=True,
            batch_size=0,
            color_map=None,
            show_progress_bar=True,
            depth_conditioning=depth_conditioning,
            target_depth=target_depth,
            mask_align=mask_align,
            mask_farther=mask_farther,
            guidance_steps=guidance_steps,
            logger=logger,
        )
        depth = depth[None, None, :].to(dtype=torch.float32)
        depth /= 200
        return depth

    @staticmethod
    def call_normal_estimator(image: torch.Tensor) -> torch.Tensor:
        """Estimate surface normal map from an input image using Marigold model.
        
        Args:
            image (torch.Tensor): Input image tensor normalized to [0, 1]
            
        Returns:
            torch.Tensor: Estimated normal map in [-1, 1] range, shape [1, 3, H, W]
                         where channels represent XYZ normal vector components
        """
        return models.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format='pt',
        ).to(dtype=torch.float32)  # [1, 3, H, W], [-1, 1]

    @staticmethod
    def call_segmentation_model(image: Union[Image.Image, torch.Tensor]):
        """Generate semantic segmentation using OneFormer model.
        
        Args:
            image (Union[PIL.Image, torch.Tensor]): Input image, either as PIL Image or tensor.
                If tensor, should be [B, C, H, W] or [C, H, W] format.
                
        Returns:
            ModelOutput: Transformer model outputs containing logits and auxiliary outputs
            
        Raises:
            TypeError: If input is neither PIL Image nor torch Tensor
        """
        # Handle both PIL Image and tensor inputs
        from PIL import Image
        from torchvision.transforms import ToPILImage
        import torch
        
        # Convert tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # [B, C, H, W]
                image = ToPILImage()(image.squeeze())
            elif image.dim() == 3:  # [C, H, W]
                image = ToPILImage()(image)
        
        # Ensure we have a PIL Image
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or torch Tensor, got {type(image)}")
            
        # Process the image
        inputs = models.segment_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
        outputs = models.segment_model(**inputs)
        return outputs

    @staticmethod
    def post_process_segmentation(segment_output,
                                target_sizes: List[Tuple[int, int]] = None) -> torch.Tensor:
        """Post-process segmentation model output into semantic map.
        
        Args:
            segment_output: Raw output from segmentation model
            target_sizes (List[Tuple[int, int]], optional): List of (height, width) tuples
                for resizing output. Defaults to [(512, 512)].
                
        Returns:
            torch.Tensor: Semantic segmentation map with class indices
        """
        """Post-process segmentation output to get semantic map"""
        if target_sizes is None:
            target_sizes = [(512, 512)]
        return models.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=target_sizes)[0]
