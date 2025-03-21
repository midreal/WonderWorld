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
        # Process the image and get inputs
        inputs = models.segment_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(models.segment_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Run model inference
        outputs = models.segment_model(**inputs)
        return outputs

    @staticmethod
    def call_segmentation_processor(image):

        # Process the image with the segment processor
        segmenter_input = models.segment_processor(image, ["semantic"], return_tensors="pt")
        
        # Move inputs to the model's device (CUDA if available)
        device = next(models.segment_model.parameters()).device
        segmenter_input = {name: tensor.to(device) if hasattr(tensor, 'to') else tensor 
                          for name, tensor in segmenter_input.items()}
        
        return segmenter_input

    @staticmethod
    def call_segmentation_post_process(segment_output, image):

        # Get the target size from the image (width, height)
        target_size = [image.size[::-1]]  # Reverse the size tuple to get (height, width)
        
        # Call the post-processing function from the segment processor
        pred_semantic_map = models.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=target_size)[0]
            
        return pred_semantic_map
