import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL as AKL, EulerAncestralDiscreteScheduler

# vae = AutoencoderKL.from_single_file(VAE_FILE_PATH, torch_dtype=torch.float16)

vae = AKL.from_single_file(
    "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL/blob/main/sdxl_vae.safetensors",
    torch_dtype=torch.float16
)

pipeline = StableDiffusionXLPipeline.from_single_file(
  "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL/blob/main/ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
  vae=vae,
  safety_checker=None,
  torch_dtype=torch.float16,
).to("cuda")

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
  prompt="score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_furry, beautiful female anthro shark portrait, dramatic lighting, dark background",
  negative_prompt="bad quality, score_3, score_2, score_1",
  height=1024,
  width=1024,
  num_inference_steps=25,
  guidance_scale=8.5,
).images[0]
