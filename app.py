from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import random
import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
from flask import jsonify
import nltk
nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)


# Download stopwords from NLTK
nltk.download('stopwords')

# Preprocess the text data


def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join the filtered tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


# Define the training data
medical_conditions = [
    "aneurysm", "arteriovenous malformation", "osteomyelitis", "pulmonary embolism", "hemothorax",
    "pneumothorax", "pulmonary nodule", "pulmonary infiltrate", "pleural effusion", "pulmonary edema",
    "atelectasis", "pneumonia", "bronchiectasis", "interstitial lung disease", "emphysema",
    "lung cancer", "mediastinal mass", "pleural thickening", "pulmonary fibrosis", "thoracic aortic aneurysm",
    "aortic dissection", "pulmonary hypertension", "pulmonary sequestration", "cavitary lung lesion",
    "pulmonary metastases", "pulmonary granuloma", "chest wall tumor", "diaphragmatic hernia", "rib fracture",
    "vertebral compression fracture", "spinal stenosis", "spinal disc herniation", "scoliosis",
    "spondylolisthesis", "sacroiliitis", "osteoporotic fracture", "hip fracture", "knee osteoarthritis",
    "meniscal tear", "ACL tear", "shoulder impingement", "rotator cuff tear", "biceps tendon rupture",
    "elbow fracture", "wrist fracture", "carpal tunnel syndrome", "finger dislocation", "hip osteoarthritis",
    "femoral neck fracture", "knee ligament tear", "ankle sprain", "Achilles tendon rupture", "plantar fasciitis",
    "calcaneal fracture", "lumbar disc herniation", "spinal cord compression", "sciatica", "spinal tumor"
]


non_medical_activities = [
    "I love pizza", "I enjoy playing soccer", "I went to the beach", "I watched a movie", "I read a book",
    "I went shopping", "I cooked dinner", "I played video games", "I went hiking", "I listened to music",
    "I attended a concert", "I went to a party", "I traveled to a new city", "I practiced yoga", "I painted a picture",
    "I spent time with friends", "I went for a bike ride", "I did gardening", "I watched a sports game", "I went for a walk",
    "I took a dance class", "I wrote a poem", "I did volunteer work", "I practiced meditation", "I visited a museum",
    "I went fishing", "I had a picnic", "I did a puzzle", "I played an instrument", "I went camping",
    "I tried a new recipe", "I did a workout", "I went to a comedy show", "I relaxed in a spa", "I went horseback riding",
    "I did a DIY project", "I visited a zoo", "I watched a theater performance", "I did bird-watching", "I went stargazing"
]

training_data = []

# Generate medical condition examples
medical_conditions = random.sample(
    medical_conditions, min(200, len(medical_conditions)))
for condition in medical_conditions:
    text = "I have " + condition
    label = "medical"
    training_data.append((text, label))

# Generate non-medical examples
non_medical_activities = random.sample(
    non_medical_activities, min(200, len(non_medical_activities)))
for activity in non_medical_activities:
    text = activity
    label = "non-medical"
    training_data.append((text, label))

# Preprocess the training data
preprocessed_training_data = [(preprocess_text(text), label) for text, label in training_data]

# Extract features from the training data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(
    [text for text, _ in preprocessed_training_data])
y_train = [label for _, label in preprocessed_training_data]

# Train the classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Function to classify new text


def classify_text(text):
    preprocessed_text = preprocess_text(text)
    X_test = vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(X_test)
    return prediction[0]


# Load the pretrained models and other components
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = UniPCMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler")
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 3.75  # Scale for classifier-free guidance
# Seed generator to create the initial latent noise
generator = torch.manual_seed(0)
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
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(
            text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    latents = latents * scheduler.init_noise_sigma

    # Set the number of inference steps
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(
            latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t,
                              encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

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


@app.route('/')
def home():
    return render_template('index.html')

# Flask Route for generating Image


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
