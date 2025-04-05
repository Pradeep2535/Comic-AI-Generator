import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initialize with memory-efficient settings"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.txt2img_pipe = None
        self.img2img_pipe = None
        
        # Memory optimization settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _load_pipeline(self, pipeline_class, **kwargs):
        """Helper to load pipeline with error handling"""
        try:
            pipe = pipeline_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use fp16 for memory savings
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False,
                **kwargs
            )
            
            # Progressive loading
            pipe.to(self.device)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
            
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except ImportError:
                    logger.warning("xformers not available, skipping optimization")
            
            return pipe
        except Exception as e:
            logger.error(f"Pipeline loading failed: {str(e)}")
            raise

    def initialize_pipelines(self):
        """Initialize pipelines with memory optimizations"""
        logger.info("Initializing pipelines...")
        
        # Load text-to-image pipeline
        self.txt2img_pipe = self._load_pipeline(StableDiffusionPipeline)
        
        # Load image-to-image pipeline (reuses components)
        self.img2img_pipe = self._load_pipeline(
            StableDiffusionImg2ImgPipeline,
            text_encoder=self.txt2img_pipe.text_encoder,
            vae=self.txt2img_pipe.vae,
            tokenizer=self.txt2img_pipe.tokenizer,
            scheduler=self.txt2img_pipe.scheduler,
        )
        
        logger.info("Pipelines initialized successfully")

    def generate_character(self, character_description: str, **kwargs) -> Image.Image:
        """Generate character with memory safeguards"""
        if not self.txt2img_pipe:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            # Clear cache before generation
            torch.cuda.empty_cache()
            
            prompt = f"""
            {character_description}
            high quality, detailed, cartoon style, vibrant colors
            """
            
            return self.txt2img_pipe(
                prompt=prompt,
                negative_prompt="blurry, deformed, ugly, low quality",
                num_inference_steps=kwargs.get('steps', 20),
                guidance_scale=kwargs.get('guidance', 7.5),
                height=512,
                width=384,  # Smaller size for memory savings
            ).images[0]
        finally:
            torch.cuda.empty_cache()

    def generate_scene(self, base_image: Image.Image, scene_description: str, **kwargs) -> Image.Image:
        """Generate scene with memory optimizations"""
        if not self.img2img_pipe:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            torch.cuda.empty_cache()
            
            return self.img2img_pipe(
                prompt=f"{scene_description}, consistent character, detailed background",
                image=base_image,
                strength=kwargs.get('strength', 0.6),
                num_inference_steps=kwargs.get('steps', 20),
                guidance_scale=kwargs.get('guidance', 7.0),
            ).images[0]
        finally:
            torch.cuda.empty_cache()

    def clear_cache(self):
        """Thorough memory cleanup"""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if self.txt2img_pipe:
            self.txt2img_pipe.to("cpu")
        if self.img2img_pipe:
            self.img2img_pipe.to("cpu")

    def __del__(self):
        """Destructor for cleanup"""
        self.clear_cache()