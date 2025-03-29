import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from util.utils import prepare_scheduler
from PIL import Image
import numpy as np
from PIL import Image
import os
import time
import cv2
import matplotlib

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
        
        # 保存深度图到本地
        os.makedirs('depth_outputs', exist_ok=True)
        depth_np = depth.to(dtype=torch.float32).cpu().numpy()  # 先转换为float32再转为numpy数组
        print("\nBefore multiplication depth_np stats:")
        print(f"Min: {depth_np.min()}, Max: {depth_np.max()}, Mean: {depth_np.mean()}")
        print(f"Shape: {depth_np.shape}, dtype: {depth_np.dtype}")
        
        depth_img = (depth_np * 255).astype(np.uint8)  # 归一化并转换为8位图像
        print("\nAfter multiplication depth_img stats:")
        print(f"Min: {depth_img.min()}, Max: {depth_img.max()}, Mean: {depth_img.mean()}")
        print(f"Shape: {depth_img.shape}, dtype: {depth_img.dtype}")
        
        depth_img = Image.fromarray(depth_img)
        save_path = os.path.join('depth_outputs', f'depth_{int(time.time())}.png')
        depth_img.save(save_path)
        
        depth = depth[None, None, :].to(dtype=torch.float32)
        print("\nBefore division depth stats:")
        print(f"Min: {depth.min().item()}, Max: {depth.max().item()}, Mean: {depth.mean().item()}")
        print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
        
        depth /= 200
        print("\nAfter division by 200 depth stats:")
        print(f"Min: {depth.min().item()}, Max: {depth.max().item()}, Mean: {depth.mean().item()}")
        print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
        
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

    @staticmethod
    def call_depth_anything_v2(image, mask=None, normalize_output=True):
        """
        使用Depth Anything V2 Large模型进行深度估计
        Args:
            image: torch.Tensor [B, C, H, W] 或 [C, H, W]，值范围[0,1]
            mask: 不使用
            normalize_output: 是否将输出标准化到[0,1]范围
        Returns:
            depth: torch.Tensor [1, 1, H, W]，深度图
        """
        from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
        import torch
        import numpy as np
        from PIL import Image

        # 固定使用vitl模型配置
        model_config = {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        }

        # 确保输入是正确的格式
        if image.dim() == 4:  # [B, C, H, W]
            image = image.squeeze(0)
        
        # 转换为PIL图像
        image_input = (image * 255).byte().permute(1, 2, 0)
        image_input = Image.fromarray(image_input.cpu().numpy())
        
        # 初始化模型
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DepthAnythingV2(**model_config)
        model.load_state_dict(torch.load('/home/4dtc/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()

        # 获取深度图
        with torch.no_grad():
            # 转换为numpy数组进行处理
            image_np = np.array(image_input)
            
            # 获取原始深度图
            depth = model.infer_image(image_np)  # 已经是numpy数组
            print("Original depth stats:")
            print(f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}")
            print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
            
            # 先归一化到0-1范围
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            # 然后反转，使近处为0（黑色），远处为1（白色）
            depth = 1 - depth
            # # 最后缩放到0-255范围
            # depth = depth * 255
            
            print("\nAfter normalization and inversion depth stats:")
            print(f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}")
            print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
            
            # 保存深度图到本地
            os.makedirs('depth_outputs_new', exist_ok=True)
            timestamp = int(time.time())
            
            # 保存原始深度值
            np.save(os.path.join('depth_outputs_new', f'depth_raw_{timestamp}.npy'), depth)
            
            # 归一化并保存可视化结果
            depth_normalized = depth.astype(np.uint8)
            
            # 转换灰度图为3通道用于显示
            depth_gray = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
            
            # 创建分隔区域
            split_region = np.ones((image_np.shape[0], 50, 3), dtype=np.uint8) * 255
            
            # 组合原图和深度图
            combined_result = cv2.hconcat([image_np[..., ::-1], split_region, depth_gray])
            
            # 保存组合图和单独的深度图
            basename = f'depth_{timestamp}'
            cv2.imwrite(os.path.join('depth_outputs_new', f'{basename}_combined.png'), combined_result)
            cv2.imwrite(os.path.join('depth_outputs_new', f'{basename}_depth.png'), depth_normalized)
            
            # 转换为tensor返回，保持原始深度值
            depth_tensor = torch.from_numpy(depth)[None, None, :].to(dtype=torch.float32)  # [1, 1, H, W]
            
            print("\nBefore division depth stats:")
            print(f"Min: {depth.min().item()}, Max: {depth.max().item()}, Mean: {depth.mean().item()}")
            print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
            depth /= 200  # 将深度值除以200进行归一化
            print("\nAfter division by 200 depth stats:")
            print(f"Min: {depth.min().item()}, Max: {depth.max().item()}, Mean: {depth.mean().item()}")
            print(f"Shape: {depth.shape}, dtype: {depth.dtype}")
            # 反转深度值，使得近处更暗
            # depth_tensor = 1.0 - depth_tensor
            return depth_tensor.to(DEVICE)
