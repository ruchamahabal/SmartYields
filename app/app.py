from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dict
import requests
import config
import pickle
import io
import os
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

app = Flask(__name__)

'''Loading Models'''

# Loading Crop Disease Classification Model

disease_classes = [
	'Apple___Apple_scab',
	'Apple___Black_rot',
	'Apple___Cedar_apple_rust',
	'Apple___healthy',
	'Blueberry___healthy',
	'Cherry_(including_sour)___Powdery_mildew',
	'Cherry_(including_sour)___healthy',
	'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
	'Corn_(maize)___Common_rust_',
	'Corn_(maize)___Northern_Leaf_Blight',
	'Corn_(maize)___healthy',
	'Grape___Black_rot',
	'Grape___Esca_(Black_Measles)',
	'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
	'Grape___healthy',
	'Orange___Haunglongbing_(Citrus_greening)',
	'Peach___Bacterial_spot',
	'Peach___healthy',
	'Pepper,_bell___Bacterial_spot',
	'Pepper,_bell___healthy',
	'Potato___Early_blight',
	'Potato___Late_blight',
	'Potato___healthy',
	'Raspberry___healthy',
	'Soybean___healthy',
	'Squash___Powdery_mildew',
	'Strawberry___Leaf_scorch',
	'Strawberry___healthy',
	'Tomato___Bacterial_spot',
	'Tomato___Early_blight',
	'Tomato___Late_blight',
	'Tomato___Leaf_Mold',
	'Tomato___Septoria_leaf_spot',
	'Tomato___Spider_mites Two-spotted_spider_mite',
	'Tomato___Target_Spot',
	'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
	'Tomato___Tomato_mosaic_virus',
	'Tomato___healthy'
]

disease_model_path = 'models/plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path))
disease_model.eval()

# Loading Crop Recommendation Model

crop_recommendation_model_path = '../models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


''' Custom functions for calculations '''

def fetch_weather(city_name):
	"""
	Fetches and returns the temperature and humidity of a city
	:params: city_name
	:return: temperature, humidity
	"""
	api_key = config.weather_api_key
	base_url = "http://api.openweathermap.org/data/2.5/weather?"

	requests.packages.urllib3.disable_warnings()
	complete_url = base_url + "appid=" + api_key + "&q=" + city_name
	response = requests.get(complete_url, verify=False)
	x = response.json()

	if x["cod"] != "404":
		y = x["main"]
		# by default open weather map returns temperature in kelvin
		# convert it to degree celcius
		temperature = round((y["temp"] - 273.15), 2)
		humidity = y["humidity"]
		return temperature, humidity
	else:
		return None


def predict_disease_from_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

'''Flask app and render pages'''

# render home page
@ app.route('/')
def home():
	title = 'SmartYields'
	return render_template('index.html', title=title)

@ app.route('/recommend_crops')
def recommend_crops():
    title = 'SmartYields - Crop Recommendation'
    return render_template('crop.html', title=title)

# render crop recommendation result page
@ app.route('/crop_recommendation_result', methods=['POST'])
def crop_recommendation_result():
	title = 'SmartYields - Crop Recommendation'

	if request.method == 'POST':
		N = int(request.form['nitrogen'])
		P = int(request.form['phosphorous'])
		K = int(request.form['potassium'])
		ph = float(request.form['ph'])
		rainfall = float(request.form['rainfall'])
		city = request.form.get('city')

		if fetch_weather(city) != None:
			temperature, humidity = fetch_weather(city)
			data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
			prediction = crop_recommendation_model.predict(data)
			final_prediction = prediction[0]

			return render_template('crop_recommendation_result.html', prediction=final_prediction, title=title)
		else:
			return None

@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    title = 'SmartYields - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_disease_from_image(img)
            prediction = Markup(str(disease_dict[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


def before_request():
    app.jinja_env.cache = {}

if __name__ == '__main__':
	app.jinja_env.auto_reload = True
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.before_request(before_request)
	app.run(debug=True)