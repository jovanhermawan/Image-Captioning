import pandas
import numpy
import urllib.parse as parse
import os
from IPython.display import display
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import requests
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.debug import DebuggedApplication


app = Flask("image_captioner")


@app.route('/')
def show_image_captioner_form():
    
    return render_template('index.html')

@app.route('/blip')
def test_link():
    return render_template('blip.html')

@app.route('/results', methods=['POST', 'GET'])
def results():
    form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model() #you can use pickle to load the trained model
        image = request.files['img']
        capt = get_caption_git(image, processor_git, model_git)
        return render_template('results.html', capt=capt, )
    
@app.route('/resultsblip', methods=['POST', 'GET'])
def resultsblip():
    form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model() #you can use pickle to load the trained model
        image = request.files['img']
        capt = get_caption_blip(image, processor_blip, model_blip)
        return render_template('results.html', capt=capt, )
    
@app.route('/urlblip', methods=['POST', 'GET'])
def urlblip():
    form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model() #you can use pickle to load the trained model
        image = request.form['url']
        capt = get_caption_blip_url(image, processor_blip, model_blip)
        return render_template('urlresults.html', capt=capt, image=image)


@app.route('/url', methods=['POST', 'GET'])
def url():
    form = request.form
    if request.method == 'POST':
        # write your function that loads the model
        # model = get_model() #you can use pickle to load the trained model
        image = request.form['url']
        capt = get_caption_git_url(image, processor_git, model_git)
        return render_template('urlresults.html', capt=capt, image=image)

def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
def get_caption_blip(url, processor, model):
    image = Image.open(url)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    capt = processor.decode(out[0], skip_special_tokens=True)
    return(capt)

def get_caption_blip_url(url, processor, model):
    image = load_image(url)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    capt = processor.decode(out[0], skip_special_tokens=True)
    return(capt)

processor_git = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model_git = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def get_caption_git(url, processor, model):
    image = Image.open(url)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return(generated_caption)

def get_caption_git_url(url, processor, model):
    image = load_image(url)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return(generated_caption)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)