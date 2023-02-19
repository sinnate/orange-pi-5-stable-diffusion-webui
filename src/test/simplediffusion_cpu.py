from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


model_id = "prompthero/openjourney"
print("\n-step : 10 \n-size : 120x120 \nModel :" +model_id+"\n")
pipe = StableDiffusionPipeline.from_pretrained(model_id, low_cpu_mem_usage=True)
pipe = pipe.to("cpu")
input_text = input(" Write what you want to display: \n> ")


prompt = input_text
image = pipe(prompt, num_inference_steps=10, width=120, height=120).images[0]

image.save(input_text+"_output.png")