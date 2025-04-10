import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "AstraliteHeart/pony-diffusion"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)


block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

num_samples = 2

def infer(prompt):
    with autocast("cuda"):
        images = pipe([prompt] * num_samples, guidance_scale=7.5)["sample"]

    return images


with block as demo:
    gr.Markdown("<h1><center>Pony Diffusion</center></h1>")
    gr.Markdown(
        "pony-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality pony images through fine-tuning."
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
               
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[2], height="auto"
        )
        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)

    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by https://huggingface.co/hakurei
   <br/>
   </p>"""
    )


demo.launch(debug=True)
