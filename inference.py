import os
import torch
import torchvision
from diffusers import DDIMScheduler
from rgb2x.load_image import load_exr_image, load_ldr_image
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
import cv2
import numpy as np
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def main():
    input_image_path = "hotpinkloratello.png"
    output_directory = "output"
    seed = 42
    inference_steps = 50
    max_image_size = 1000
    
    os.makedirs(output_directory, exist_ok=True)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print("Loading model...")
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=False)
    
    print(f"Loading image: {input_image_path}")
    if input_image_path.endswith(".exr"):
        photo = load_exr_image(input_image_path, tonemaping=True, clamp=True).to("cuda")
    elif input_image_path.endswith((".png", ".jpg", ".jpeg")):
        photo = load_ldr_image(input_image_path, from_srgb=True).to("cuda")
    else:
        raise ValueError("Unsupported image format. Use .exr, .png, .jpg, or .jpeg")
    
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    aspect_ratio = old_height / old_width
    
    if old_height > old_width:
        new_height = max_image_size
        new_width = int(new_height / aspect_ratio)
    else:
        new_width = max_image_size
        new_height = int(new_width * aspect_ratio)

    new_width = new_width // 8 * 8
    new_height = new_height // 8 * 8
    
    photo = torchvision.transforms.Resize((new_height, new_width))(photo)
    
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }
    
    print("Generating AOVs...")
    for aov_name in required_aovs:
        prompt = prompts[aov_name]
        print(f"  Generating {aov_name}...")
        
        generated_image = pipe(
            prompt=prompt,
            photo=photo,
            num_inference_steps=inference_steps,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=[aov_name],
            output_type="pil",
        ).images[0][0]
        
        generated_image = generated_image.resize((old_width, old_height), Image.LANCZOS)
        
        output_path = os.path.join(output_directory, f"{aov_name}.png")
        generated_image.save(output_path)
        
        print(f"  Saved to {output_path}")
    
    print("All AOVs generated successfully!")

if __name__ == "__main__":
    main()
