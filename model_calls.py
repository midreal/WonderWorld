import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from util.utils import prepare_scheduler
from PIL import Image
import os
import datetime

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
        self.segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
        
        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            safety_checker=None,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.inpainting_pipeline.scheduler = DDIMScheduler.from_config(self.inpainting_pipeline.scheduler.config)
        self.inpainting_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self.inpainting_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        
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
    @staticmethod
    def call_inpainting_pipeline(prompt="", negative_prompt=None, image=None, mask_image=None, 
                               num_inference_steps=None, guidance_scale=7.5, height=None, width=None, 
                               self_guidance=None, inpaint_mask=None, rendered_image=None):
        # Create directory for saving images if it doesn't exist
        save_dir = "/home/4dtc/WonderWorld/saved_images"
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate a timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the input image if it exists
        if image is not None:
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image if needed
                if image.dim() == 4:  # [B, C, H, W]
                    pil_image = Image.fromarray((image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                elif image.dim() == 3:  # [C, H, W]
                    pil_image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                else:
                    pil_image = image
            else:
                pil_image = image
                
            image_path = os.path.join(save_dir, f"input_image_{timestamp}.png")
            pil_image.save(image_path)
            print(f"Input image saved to: {image_path}")
        
        # Save the mask image if it exists
        if mask_image is not None:
            if isinstance(mask_image, torch.Tensor):
                # Convert tensor to PIL Image if needed
                if mask_image.dim() == 4:  # [B, C, H, W]
                    pil_mask = Image.fromarray((mask_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                elif mask_image.dim() == 3:  # [C, H, W]
                    if mask_image.shape[0] == 1:  # Single channel mask
                        pil_mask = Image.fromarray((mask_image[0].cpu().numpy() * 255).astype('uint8'))
                    else:
                        pil_mask = Image.fromarray((mask_image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                else:
                    pil_mask = mask_image
            else:
                pil_mask = mask_image
                
            mask_path = os.path.join(save_dir, f"mask_image_{timestamp}.png")
            pil_mask.save(mask_path)
            print(f"Mask image saved to: {mask_path}")
        
        return models.inpainting_pipeline(
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

    @staticmethod
    def call_depth_model(image, denoising_steps=30, guidance_steps=8, depth_conditioning=None,
                        target_depth=None, mask_align=None, mask_farther=None, logger=None):
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
    def call_normal_estimator(image):
        return models.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format='pt',
        ).to(dtype=torch.float32)  # [1, 3, H, W], [-1, 1]

    @staticmethod
    def call_segmentation_model(image):
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
    def post_process_segmentation(segment_output, target_sizes=None):
        """Post-process segmentation output to get semantic map"""
        if target_sizes is None:
            target_sizes = [(512, 512)]
        return models.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=target_sizes)[0]
