#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import Pyro5.api
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import json

# 配置Pyro5使用JSON序列化器，它更简单且更可靠
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

class RemoteKeyframeGen:
    """KeyframeGen的远程代理类，提供与本地KeyframeGen相同的接口"""
    
    def __init__(self, service_uri=None, nameserver_host="localhost", nameserver_port=9090):
        """
        初始化RemoteKeyframeGen
        
        Args:
            service_uri: 服务URI，如果提供则直接连接
            nameserver_host: 名称服务器主机地址，默认为localhost
            nameserver_port: 名称服务器端口，默认为9090
        """
        if service_uri:
            self.service = Pyro5.api.Proxy(service_uri)
        else:
            # 连接到名称服务器
            nameserver = Pyro5.api.locate_ns(host=nameserver_host, port=nameserver_port)
            uri = nameserver.lookup("keyframe.service")
            self.service = Pyro5.api.Proxy(uri)
        
        self.initialized = False
        print(f"已连接到KeyframeGen服务: {self.service._pyroUri}")
    
    def initialize(self, config, rotation_path=None):
        """初始化远程KeyframeGen服务"""
        print("初始化远程KeyframeGen服务...")
        
        # 创建一个可序列化的配置字典
        serializable_config = {}
        
        # 处理配置参数，确保所有值都是可序列化的
        for key, value in config.items():
            if key == "device":
                # 设备需要转换为字符串
                serializable_config[key] = str(value)
            elif key == "rotation_path":
                # 跳过rotation_path，我们将单独处理它
                pass
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                # 将NumPy数组和PyTorch张量转换为列表
                if isinstance(value, torch.Tensor):
                    serializable_config[key] = value.detach().cpu().numpy().tolist()
                else:
                    serializable_config[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str, list, dict, type(None))):
                # 这些类型可以直接序列化
                serializable_config[key] = value
            else:
                # 其他类型尝试转换为字符串
                try:
                    serializable_config[key] = str(value)
                except:
                    print(f"警告: 无法序列化配置项 {key}，已跳过")
        
        # 准备rotation_path参数
        rotation_path_list = None
        if rotation_path is not None:
            # 如果外部提供了rotation_path，则使用外部提供的
            if isinstance(rotation_path, np.ndarray):
                rotation_path_list = rotation_path.tolist()
                print(f"从外部提供的numpy.ndarray转换rotation_path为list")
            elif isinstance(rotation_path, list):
                rotation_path_list = rotation_path
                print(f"使用外部提供的list类型rotation_path")
            else:
                try:
                    rotation_path_list = list(rotation_path)
                    print(f"从其他类型({type(rotation_path)})转换rotation_path为list")
                except Exception as e:
                    print(f"警告: 无法转换外部提供的rotation_path: {str(e)}")
        elif "rotation_path" in config:
            # 从config中获取rotation_path并限制长度
            try:
                rotation_path_value = config["rotation_path"]
                if "num_scenes" in config and hasattr(rotation_path_value, "__len__"):
                    rotation_path_value = rotation_path_value[:config["num_scenes"]]
                
                # 确保rotation_path是列表格式
                if isinstance(rotation_path_value, np.ndarray):
                    rotation_path_list = rotation_path_value.tolist()
                    print(f"从config中的numpy.ndarray转换rotation_path为list")
                elif isinstance(rotation_path_value, list):
                    rotation_path_list = rotation_path_value
                    print(f"使用config中的list类型rotation_path")
                else:
                    try:
                        rotation_path_list = list(rotation_path_value)
                        print(f"从config中的其他类型({type(rotation_path_value)})转换rotation_path为list")
                    except:
                        print(f"警告: 无法转换config中的rotation_path")
            except Exception as e:
                print(f"警告: 处理rotation_path时出错: {str(e)}")
        
        # 将rotation_path添加到config中
        if rotation_path_list is not None:
            serializable_config["rotation_path"] = rotation_path_list
        
        # 打印rotation_path信息用于调试
        if "rotation_path" in serializable_config:
            print(f"rotation_path类型: {type(serializable_config['rotation_path'])}")
            print(f"rotation_path长度: {len(serializable_config['rotation_path'])}")
            print(f"rotation_path前几个元素: {serializable_config['rotation_path'][:3]}")
        
        # 使用JSON序列化器，直接传递配置
        try:
            success = self.service.initialize(serializable_config, None)
            self.initialized = success
            return success
        except Exception as e:
            print(f"初始化服务失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.initialized = False
            return False
    
    # ===== 属性访问方法 =====
    
    @property
    def image_latest(self):
        """获取最新图像"""
        return self.service.get_image_latest()
    
    @image_latest.setter
    def image_latest(self, image):
        """设置最新图像"""
        # 存储本地引用
        self._image_latest = image
        
        # 处理 PyTorch 张量序列化
        if isinstance(image, torch.Tensor):
            # 将 PyTorch 张量转换为 NumPy 数组
            if image.is_cuda:
                image_np = image.detach().cpu().numpy()
            else:
                image_np = image.detach().numpy()
            
            # 将 NumPy 数组发送到服务端
            return self.service.set_image_latest(image_np)
        else:
            return self.service.set_image_latest(image)
    
    @property
    def run_dir(self):
        """获取运行目录"""
        if self._run_dir is None:
            self._run_dir = self.service.get_run_dir()
        return self._run_dir
    
    @property
    def gaussians(self):
        """获取高斯模型"""
        return self.service.get_gaussians()
    
    @gaussians.setter
    def gaussians(self, gaussians):
        """设置高斯模型"""
        self.service.set_gaussians(gaussians)
    
    @property
    def scene_name(self):
        """获取场景名称"""
        return self.service.get_scene_name()
    
    @scene_name.setter
    def scene_name(self, scene_name):
        """设置场景名称"""
        self.service.set_scene_name(scene_name)
    
    # ===== 方法代理 =====
    
    def generate_sky_mask(self, input_image=None):
        """生成天空掩码"""
        return self.service.generate_sky_mask(input_image)
    
    def generate_sky_pointcloud(self, syncdiffusion_model, image, mask, gen_sky, style):
        """生成天空点云"""
        # 不传递syncdiffusion_model对象，而是让服务端自己创建
        # 如果需要，可以传递模型的配置参数
        model_config = None
        if syncdiffusion_model is not None:
            model_config = {
                "sd_version": getattr(syncdiffusion_model, "sd_version", "2.0-inpaint"),
                # 添加其他必要的配置参数
            }
        
        return self.service.generate_sky_pointcloud(model_config, image, mask, gen_sky, style)
    
    def recompose_image_latest_and_set_current_pc(self, scene_name):
        """重组最新图像并设置当前点云"""
        return self.service.recompose_image_latest_and_set_current_pc(scene_name)
    
    def increment_kf_idx(self):
        """增加关键帧索引"""
        return self.service.increment_kf_idx()
    
    def convert_to_3dgs_traindata(self, xyz_scale, remove_threshold=None, use_no_loss_mask=False):
        """转换为3DGS训练数据"""
        return self.service.convert_to_3dgs_traindata(xyz_scale, remove_threshold, use_no_loss_mask)
    
    def render(self, archive_output=False, camera=None, render_visible=False, render_sky=False, big_view=False, render_fg=False):
        """渲染场景"""
        return self.service.render(
            archive_output=archive_output,
            camera=camera,
            render_visible=render_visible,
            render_sky=render_sky,
            big_view=big_view,
            render_fg=render_fg
        )
    
    def get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0):
        """通过JS视图矩阵获取相机"""
        return self.service.get_camera_by_js_view_matrix(view_matrix, xyz_scale)
    
    def inpaint_image_with_prompt(self, prompt, negative_prompt, mask, strength=0.75, num_inference_steps=50, guidance_scale=7.5):
        """使用提示进行图像修复"""
        return self.service.inpaint_image_with_prompt(prompt, negative_prompt, mask, strength, num_inference_steps, guidance_scale)
    
    def process_inpainted_image(self, inpainted_image, mask):
        """处理修复后的图像"""
        return self.service.process_inpainted_image(inpainted_image, mask)
    
    def update_point_cloud(self, inpainted_image, mask, scene_name):
        """更新点云"""
        return self.service.update_point_cloud(inpainted_image, mask, scene_name)
    
    # 添加其他所有KeyframeGen方法...
    
    # 添加to方法以模拟PyTorch的to方法
    def to(self, device):
        """模拟PyTorch的to方法，返回自身"""
        return self

if __name__ == "__main__":
    # 测试代码
    try:
        # 创建远程KeyframeGen客户端
        kf_gen = RemoteKeyframeGen()
        print("成功连接到KeyframeGen服务")
    except Exception as e:
        print(f"连接KeyframeGen服务失败: {str(e)}")
