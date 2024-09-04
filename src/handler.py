""" Example handler file. """

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

import runpod
import base64
import io
import time

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    # pipe = AutoPipelineForText2Image.from_pretrained("hugger911/sdxlc_v10.safetensors")
    # pipe.to("cuda")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_single_file("https://huggingface.co/nDimensional/NatVis-Natural-Vision-SDXL/blob/main/NaturalVision-epoch68.fp16.safetensors", vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    refiner = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    refiner.to("cuda")
except RuntimeError:
    print('failed to initialize StableDiffusionXLPipeline')
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    num_inference_steps = job_input['num_inference_steps']
    refiner_inference_steps = job_input['refiner_inference_steps']
    width = job_input['width']
    height = job_input['height']
    guidance_scale = job_input['guidance_scale']
    negative_prompt = job_input['negative_prompt']
    # strength = job_input['strength']
    seed = job_input['seed']
    num_images = job_input['num_images']
    denoising_start = job_input['denoising_start']

    time_start = time.time()
    image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height, negative_prompt=negative_prompt).images
    image = refiner(prompt=prompt, num_inference_steps=refiner_inference_steps, denoising_start=denoising_start, image=image).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
