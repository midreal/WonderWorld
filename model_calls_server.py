import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from util.utils import prepare_scheduler
from PIL import Image
import io
import base64, json
from flask import Flask, request, jsonify
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor

class Models:
    def __init__(self):
        self.segment_processor = None
        self.segment_model = None
        self.inpainting_pipeline = None
        self.depth_model = None
        self.normal_estimator = None
        self.device = None

    def initialize(self, device):
        print("Initializing models...")
        self.device = device
        self.segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')
        print("Loaded segmentation models")
        
        # self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-inpainting",
        #     safety_checker=None,
        #     torch_dtype=torch.bfloat16,
        # ).to(device)
        # self.inpainting_pipeline.scheduler = DDIMScheduler.from_config(self.inpainting_pipeline.scheduler.config)
        # self.inpainting_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        # self.inpainting_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        # print("Loaded inpainting pipeline")
        
        self.depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-v1-0", torch_dtype=torch.bfloat16).to(device)
        self.depth_model.scheduler = EulerDiscreteScheduler.from_config(self.depth_model.scheduler.config)
        self.depth_model.scheduler = prepare_scheduler(self.depth_model.scheduler)
        print("Loaded depth model")
        
        self.normal_estimator = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1", 
            torch_dtype=torch.bfloat16
        ).to(device)
        print("Loaded normal estimator")

# Global model instance
models = Models()
models.initialize('cuda')
app = Flask(__name__)

def serialize_for_json(obj):
    """Convert any object to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        buffer = io.BytesIO()
        # Convert bfloat16 to float32 before numpy conversion
        if obj.dtype == torch.bfloat16:
            obj = obj.cpu().to(torch.float32)
            tensor_np = obj.detach().numpy()
            np.save(buffer, tensor_np)
            return {
                "__type__": "tensorbf16",
                "data": base64.b64encode(buffer.getvalue()).decode('utf-8')
            }
        obj = obj.cpu()
        tensor_np = obj.detach().numpy()
        np.save(buffer, tensor_np)
        return {
            "__type__": "tensor",
            "data": base64.b64encode(buffer.getvalue()).decode('utf-8')
        }
    elif hasattr(obj, 'numpy'):  # Handle other array-like objects
        return serialize_for_json(torch.tensor(obj))
    elif hasattr(obj, 'item'):  # Handle scalar tensors
        return obj.item()
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
            elif obj["__type__"] == "image":
                buffer = io.BytesIO(base64.b64decode(obj["data"]))
                return Image.open(buffer)
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(x) for x in obj]
    return obj

def base64_to_tensor(data):
    """Convert base64 data back to PyTorch tensor"""
    return deserialize_from_json(data)

def pil_to_base64(image):
    """Convert PIL Image to base64 string"""
    return serialize_for_json(image)

def base64_to_pil(b64_data):
    """Convert base64 data back to PIL Image"""
    return deserialize_from_json(b64_data)

@app.route('/inpainting', methods=['POST'])
def inpainting():
    try:
        data = request.json
        
        # Handle optional inputs with proper deserialization
        def safe_deserialize_image(key):
            if key not in data:
                return None
            value = deserialize_from_json(data[key])
            if isinstance(value, torch.Tensor):
                return value.to(models.device)
            elif isinstance(value, Image.Image):
                return ToTensor()(value).unsqueeze(0).to(models.device)
            return None
        
        image = safe_deserialize_image('image')
        mask_image = safe_deserialize_image('mask_image')
        inpaint_mask = safe_deserialize_image('inpaint_mask')
        rendered_image = safe_deserialize_image('rendered_image')

        result = models.inpainting_pipeline(
            prompt=data.get('prompt', ''),
            negative_prompt=data.get('negative_prompt'),
            image=image,
            mask_image=mask_image,
            num_inference_steps=data.get('num_inference_steps'),
            guidance_scale=data.get('guidance_scale', 7.5),
            height=data.get('height'),
            width=data.get('width'),
            self_guidance=data.get('self_guidance'),
            inpaint_mask=inpaint_mask,
            rendered_image=rendered_image,
        ).images[0]
        
        return jsonify(serialize_for_json(result))
    except Exception as e:
        print(f"Error in inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/depth', methods=['POST'])
def depth():
    try:
        data = request.json
        # Load and ensure all tensors are on CUDA
        image = base64_to_tensor(data['image']).to('cuda')
        
        # Helper function to safely deserialize tensor-like values
        def safe_deserialize(key):
            if key not in data:
                return None
            value = data[key]
            if not isinstance(value, (dict, str)):
                return None
            try:
                tensor = base64_to_tensor(value).to('cuda')
                # For masks, ensure they have shape [1, 1, H, W]
                if key in ['mask_align', 'mask_farther']:
                    if tensor.dim() == 3:  # [C, H, W]
                        tensor = tensor.unsqueeze(0)  # Add batch dim
                    if tensor.shape[1] != 1:  # If channel dim is not 1
                        tensor = tensor.unsqueeze(1)  # Add channel dim
                return tensor
            except:
                return None
        
        # Handle optional tensors
        depth_conditioning = safe_deserialize('depth_conditioning')
        target_depth = safe_deserialize('target_depth')
        mask_align = safe_deserialize('mask_align')
        mask_farther = safe_deserialize('mask_farther')
        
        # Convert to PIL Image for depth model input
        image_input = (image*255).byte().squeeze().permute(1, 2, 0)
        image_input = Image.fromarray(image_input.cpu().numpy())
        
        depth = models.depth_model(
            image_input,
            denoising_steps=data.get('denoising_steps', 30),
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
            guidance_steps=data.get('guidance_steps', 8),
            logger=None,
        )
        
        # Ensure depth output is on CUDA with correct dtype
        depth = depth[None, None, :].to(device='cuda', dtype=torch.float32)
        depth /= 200
        return jsonify(serialize_for_json({'depth': depth}))
    except Exception as e:
        print(f"Error in depth: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/normal', methods=['POST'])
def normal():
    try:
        data = request.json
        image = base64_to_tensor(data['image']).to(models.device)
        
        result = models.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format='pt',
        ).to(device=models.device, dtype=torch.float32)
        
        return jsonify(serialize_for_json({'normal': result}))
    except Exception as e:
        print(f"Error in normal: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/segmentation', methods=['POST'])
def segmentation():
    try:
        data = request.json
        image_data = base64_to_tensor(data['image'])
        target_sizes = data.get('target_sizes', [(512, 512)])
        
        # Convert tensor to PIL Image if needed
        if isinstance(image_data, torch.Tensor):
            if image_data.dim() == 4:  # [B, C, H, W]
                image = ToPILImage()(image_data.squeeze())
            elif image_data.dim() == 3:  # [C, H, W]
                image = ToPILImage()(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise TypeError(f"Expected PIL Image or torch Tensor, got {type(image_data)}")
        
        # Process the image
        inputs = models.segment_processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {name: tensor.to(models.device) for name, tensor in inputs.items()}
        outputs = models.segment_model(**inputs)
        
        # Post-process the outputs
        result = models.segment_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes)[0]
            
        # Ensure result is on correct device
        if isinstance(result, torch.Tensor):
            result = result.to(models.device)
        
        # Convert result to serializable format
        serialized_result = serialize_for_json({'result': result})
        return jsonify(serialized_result)
    except Exception as e:
        print(f"Error in segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    models.initialize('cuda')
    app.run(host='0.0.0.0', port=5000)
