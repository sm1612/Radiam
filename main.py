import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from nlp_test import classify_text
from flask import jsonify


app = Flask(__name__)


# Load the pretrained models and other components
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = 1

torch_device = "cpu"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# Define a function to generate the image
def generate_image(prompt):
    classification = classify_text(prompt)
    
    if classification == "medical":
        text_input = tokenizer(
            [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(num_inference_steps)  # Set the number of inference steps

        for t in scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        generated_image = pil_images[0]

        return generated_image
    return None



@app.route('/')
def home():
    return render_template('index.html')

# Flask route for generating the image
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_image = generate_image(prompt)

    if generated_image:
        # Save the generated image
        generated_image_path = "static/generated_image.jpg"
        generated_image.save(generated_image_path)
        return redirect(url_for('result', prompt=prompt))
    else:
        return jsonify({'error': 'Prompt does not belong to medical category.'})

# Define a route to display the result
@app.route('/result')
def result():
    prompt = request.args.get('prompt')
    generated_image_path = "static/generated_image.jpg"

    if os.path.exists(generated_image_path):
        return render_template('result.html', prompt=prompt, image_path=generated_image_path)
    else:
        return render_template('error.html', message='Prompt does not belong to medical category.')


if __name__ == '__main__':
    app.run()
