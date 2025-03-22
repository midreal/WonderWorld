#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import Pyro5.api
import Pyro5.nameserver
import Pyro5.errors
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目相关模块
from models.models import KeyframeGen
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from util.utils import prepare_scheduler
from util.segment_utils import create_mask_generator_repvit
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldPipelineNormal, MarigoldNormalsPipeline

# 配置Pyro5
# 使用JSON序列化器，它更简单且更可靠
Pyro5.config.SERIALIZER = "json"
# 增加最大消息大小限制，以支持大型张量传输
Pyro5.config.MAX_MESSAGE_SIZE = 1024 * 1024 * 1024  # 1GB

# 为NumPy数组添加自定义序列化处理
def numpy_array_to_list(obj):
    """将NumPy数组转换为Python列表"""
    return obj.tolist()

def numpy_int_to_int(obj):
    """将NumPy整数转换为Python整数"""
    return int(obj)

def numpy_float_to_float(obj):
    """将NumPy浮点数转换为Python浮点数"""
    return float(obj)

def numpy_bool_to_bool(obj):
    """将NumPy布尔值转换为Python布尔值"""
    return bool(obj)

def tensor_to_list(obj):
    """将PyTorch张量转换为Python列表"""
    return obj.detach().cpu().numpy().tolist()

# 注册自定义类的序列化器
Pyro5.api.register_class_to_dict(np.ndarray, numpy_array_to_list)
Pyro5.api.register_class_to_dict(np.int64, numpy_int_to_int)
Pyro5.api.register_class_to_dict(np.int32, numpy_int_to_int)
Pyro5.api.register_class_to_dict(np.float64, numpy_float_to_float)
Pyro5.api.register_class_to_dict(np.float32, numpy_float_to_float)
Pyro5.api.register_class_to_dict(np.bool_, numpy_bool_to_bool)
Pyro5.api.register_class_to_dict(torch.Tensor, tensor_to_list)

@Pyro5.api.expose
class KeyframeGenService:
    """KeyframeGen服务类，提供远程访问KeyframeGen的所有功能"""
    
    def __init__(self):
        self.kf_gen = None
        self.config = None
        self.models = {}
        print("KeyframeGenService实例已创建")
    
    def initialize(self, config, rotation_path):
        """初始化KeyframeGen服务"""
        print("初始化KeyframeGen服务...")
        self.config = config

        # 打印配置信息
        print(f"发送配置: {list(config.keys())}")
        
        # 如果设备是字符串形式，转换为torch设备
        if "device" in config and isinstance(config["device"], str):
            config["device"] = torch.device(config["device"])
        
        try:
            # 初始化所有必要的模型
            print("正在加载分割模型...")
            self.models["segment_processor"] = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
            self.models["segment_model"] = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
            
            print("正在加载掩码生成器...")
            self.models["mask_generator"] = create_mask_generator_repvit()
            
            print("正在加载Stable Diffusion模型...")
            inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    config["stable_diffusion_checkpoint"],
                    safety_checker=None,
                    torch_dtype=torch.bfloat16,
                ).to(config["device"])
            inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
            inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
            inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
            self.models["inpainter_pipeline"] = inpainter_pipeline
            
            print("正在加载深度模型...")
            depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16).to(config["device"])
            depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
            depth_model.scheduler = prepare_scheduler(depth_model.scheduler)
            self.models["depth_model"] = depth_model

            print("正在加载法线估计器...")
            self.models["normal_estimator"] = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16).to(config["device"])
            
            print("正在创建KeyframeGen实例...")
            print(f"rotation_path类型: {type(rotation_path)}")
            print(f"rotation_path内容: {rotation_path}")
            try:
                self.kf_gen = KeyframeGen(
                    config=config, 
                    inpainter_pipeline=self.models["inpainter_pipeline"], 
                    mask_generator=self.models["mask_generator"], 
                    depth_model=self.models["depth_model"],
                    segment_model=self.models["segment_model"], 
                    segment_processor=self.models["segment_processor"], 
                    normal_estimator=self.models["normal_estimator"],
                    rotation_path=rotation_path, 
                    inpainting_resolution=config['inpainting_resolution_gen']
                ).to(config["device"])
            except Exception as e:
                print(f"初始化KeyframeGen时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            print("KeyframeGen初始化完成")
            return True
        except Exception as e:
            print(f"初始化KeyframeGen时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # ===== KeyframeGen属性访问方法 =====
    
    def get_image_latest(self):
        """获取最新图像"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.image_latest
    
    def set_image_latest(self, image):
        """设置最新图像"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        # 将numpy array转回tensor并移到正确的设备上
        if isinstance(image, (list, np.ndarray)):
            image = torch.tensor(image).to(self.config["device"])
        self.kf_gen.image_latest = image
        return True
    
    def get_run_dir(self):
        """获取运行目录"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.run_dir
    
    # ===== KeyframeGen方法代理 =====
    
    def generate_sky_mask(self, input_image=None):
        """生成天空掩码"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        # 在服务端就转换为float
        result = self.kf_gen.generate_sky_mask(input_image)
        return result.float()
    
    def generate_sky_pointcloud(self, model_config, image, mask, gen_sky, style):
        """生成天空点云"""
        # 如果需要，根据model_config创建SyncDiffusion模型
        syncdiffusion_model = None
        if gen_sky and model_config is not None:
            from syncdiffusion.syncdiffusion_model import SyncDiffusion
            sd_version = model_config.get("sd_version", "2.0-inpaint")
            syncdiffusion_model = SyncDiffusion(self.config["device"], sd_version=sd_version)
        
        # 调用KeyframeGen的方法
        if self.kf_gen:
            return self.kf_gen.generate_sky_pointcloud(syncdiffusion_model, image, mask, gen_sky, style)
        return False
    
    def recompose_image_latest_and_set_current_pc(self, scene_name):
        """重组最新图像并设置当前点云"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.recompose_image_latest_and_set_current_pc(scene_name)
    
    def increment_kf_idx(self):
        """增加关键帧索引"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.increment_kf_idx()
    
    def convert_to_3dgs_traindata(self, xyz_scale, remove_threshold, use_no_loss_mask):
        """转换为3DGS训练数据"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.convert_to_3dgs_traindata(xyz_scale, remove_threshold, use_no_loss_mask)
    
    def render(self, archive_output=False, camera=None, render_visible=False, render_sky=False, big_view=False, render_fg=False):
        """渲染场景"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.render(
            archive_output=archive_output,
            camera=camera,
            render_visible=render_visible,
            render_sky=render_sky,
            big_view=big_view,
            render_fg=render_fg
        )
    
    def get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0):
        """通过JS视图矩阵获取相机"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale)
    
    def get_gaussians(self):
        """获取高斯模型"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.gaussians
    
    def set_gaussians(self, gaussians):
        """设置高斯模型"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        self.kf_gen.gaussians = gaussians
        return True
    
    def set_scene_name(self, scene_name):
        """设置场景名称"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        self.kf_gen.scene_name = scene_name
        return True
    
    def get_scene_name(self):
        """获取场景名称"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.scene_name
    
    def inpaint_image_with_prompt(self, prompt, negative_prompt, mask, strength, num_inference_steps, guidance_scale):
        """使用提示进行图像修复"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.inpaint_image_with_prompt(prompt, negative_prompt, mask, strength, num_inference_steps, guidance_scale)
    
    def process_inpainted_image(self, inpainted_image, mask):
        """处理修复后的图像"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.process_inpainted_image(inpainted_image, mask)
    
    def update_point_cloud(self, inpainted_image, mask, scene_name):
        """更新点云"""
        if self.kf_gen is None:
            raise RuntimeError("KeyframeGen尚未初始化")
        return self.kf_gen.update_point_cloud(inpainted_image, mask, scene_name)
    
    # 添加其他所有KeyframeGen方法...

def start_keyframe_service(host="0.0.0.0", port=9090):
    """启动KeyframeGen服务"""
    # 创建服务实例
    keyframe_service = KeyframeGenService()
    
    # 创建守护进程
    daemon = Pyro5.api.Daemon(host=host)
    uri = daemon.register(keyframe_service)
    
    # 启动名称服务器（如果尚未运行）
    try:
        ns = Pyro5.api.locate_ns()
        ns.register("keyframe.service", uri)
        print(f"KeyframeGen服务已注册: {uri}")
    except Pyro5.errors.NamingError:
        print("名称服务器未运行，请先启动名称服务器：python -m Pyro5.nameserver")
        return
    
    print(f"KeyframeGen服务已启动在 {host}:{port}，等待连接...")
    daemon.requestLoop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='启动KeyframeGen服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=9090, help='服务端口')
    
    args = parser.parse_args()
    
    start_keyframe_service(host=args.host, port=args.port)
