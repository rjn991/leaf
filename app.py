from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import g4f
import re

def add_newline(string):
    pattern = r'(\d+\.)' 
    replacement = r'\1\n'
    new_string = re.sub(pattern,'\n', string)
    return new_string

app = Flask(__name__)

# Load the TensorFlow model
leaf = tf.keras.models.load_model('model.h5')

# Define the class names
class_names = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))
    img = np.reshape(img, [1, 299, 299, 3])
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']

        # Save the file to the uploads folder
        file_path = "static/uploads/" + file.filename
        file.save(file_path)

        # Process the image
        img = process_image(file_path)


        # Make predictions
        preds = leaf.predict(img)
        max_idx = np.argmax(preds)
        prediction = class_names[max_idx]

        g4f.debug.logging = True  # Enable logging

        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"What are the benifits of using {prediction}. Give at most 5 benifits with numbers"}],
            # messages=[{"role": "user", "content": f"How to add newline to a string when number followed by a dot appears"}],
            provider=g4f.Provider.GptGo,
            # stream=True,
        )
        
        print(response)
        return render_template('index.html', prediction=prediction, file_path=file_path,gpt_response=response.replace("\n", "<br>"))

    return render_template('index.html', prediction=None, file_path=None)

if __name__ == '__main__':
    app.run(debug=True)
