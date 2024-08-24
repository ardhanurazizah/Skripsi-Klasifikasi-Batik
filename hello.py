import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import numpy as np
import base64
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'batik'
app.config['UPLOAD_FOLDER'] = 'uploads/'

mysql = MySQL(app)

PER_PAGE = 4  # Number of data to display per page

def calculate_intensity_frequency(image):
    intensity_freq = np.zeros((256, 3), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            intensity_freq[r, 0] += 1
            intensity_freq[g, 1] += 1
            intensity_freq[b, 2] += 1
    return intensity_freq

def contrast_enhancement(image, K, P):
    enhanced_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            new_r = np.round(K * (r - P) + P)
            new_g = np.round(K * (g - P) + P)
            new_b = np.round(K * (b - P) + P)
            new_r = max(0, min(new_r, 255))
            new_g = max(0, min(new_g, 255))
            new_b = max(0, min(new_b, 255))
            enhanced_image[i, j] = [new_r, new_g, new_b]
    return enhanced_image

def classify_motif(image_path, model, save_enhanced_path):
    input_image = cv2.imread(image_path)
    input_image = cv2.resize(input_image, (128, 128))

    intensity_freq = calculate_intensity_frequency(input_image)
    max_intensity_freq = np.max(intensity_freq)
    K = 255 / max_intensity_freq
    P = 0
    enhanced_image = contrast_enhancement(input_image, K, P)

    cv2.imwrite(save_enhanced_path, enhanced_image)

    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(enhanced_image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(pil_image)
    image = image.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    _, predicted = torch.max(output, 1)
    
    checkpoint = torch.load('models/model.pt', map_location=device)
    class_names = checkpoint['class_names']
    
    predicted_class = class_names[predicted.item()]
    
    return predicted_class, save_enhanced_path


@app.route("/")
def main():
    page = request.args.get('page', 1, type=int)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM data LIMIT %s, %s", (start, PER_PAGE))
    data = cur.fetchall()
    cur.close()
    
    images = []
    for row in data:
        image_blob = row[2]
        image_base64 = base64.b64encode(image_blob).decode('utf-8')
        images.append((row[0], row[1], image_base64, row[3], row[4]))
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM data")
    total_data = cur.fetchone()[0]
    cur.close()

    num_pages = total_data // PER_PAGE + (total_data % PER_PAGE > 0)
    
    return render_template('index.html', data=images, page=page, num_pages=num_pages)

@app.route("/search")
def search():
    keyword = request.args.get('search', '')
    page = request.args.get('page', 1, type=int)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM data WHERE namaBatik LIKE %s LIMIT %s, %s", ('%' + keyword + '%', start, PER_PAGE))
    data = cur.fetchall()
    cur.close()
    
    images = []
    for row in data:
        image_blob = row[2]
        image_base64 = base64.b64encode(image_blob).decode('utf-8')
        images.append((row[0], row[1], image_base64, row[3]))
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM data WHERE namaBatik LIKE %s", ('%' + keyword + '%',))
    total_data = cur.fetchone()[0]
    cur.close()

    num_pages = total_data // PER_PAGE + (total_data % PER_PAGE > 0)
    
    return render_template('index.html', data=images, page=page, num_pages=num_pages)

@app.route("/upload", methods=['POST'])
def upload():
   
    file = request.files['image']
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        enhanced_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{file.filename}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model from checkpoint
        model_path = 'models/model.pt'
        checkpoint = torch.load(model_path, map_location=device)
        
        # Ensure the 'model_state_dict' key exists in the checkpoint
        if 'model_state_dict' in checkpoint:
            # Create an instance of the VGG-19 model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False)
            
            # Modify the classifier part of the model
            old_conv = model.features[0]
            model.features[0] = nn.Sequential(
                 nn.Dropout(p=0.25),
                 old_conv)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 53)
            
            # Load the state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set the model to evaluation mode
            model.eval()
            
            # Move the model to the appropriate device
            model = model.to(device)

            prediction, enhanced_image_path = classify_motif(file_path, model, enhanced_image_path)

            original_image_path = file_path
            enhanced_image_path = enhanced_image_path

            page = request.args.get('page', 1, type=int)
            start = (page - 1) * PER_PAGE
            end = start + PER_PAGE

            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM data LIMIT %s, %s", (start, PER_PAGE))
            data = cur.fetchall()
            cur.close()

            images = []
            for row in data:
                image_blob = row[2]
                image_base64 = base64.b64encode(image_blob).decode('utf-8')
                images.append((row[0], row[1], image_base64, row[3]))

            cur = mysql.connection.cursor()
            cur.execute("SELECT COUNT(*) FROM data")
            total_data = cur.fetchone()[0]
            cur.close()

            num_pages = total_data // PER_PAGE + (total_data % PER_PAGE > 0)

            # Convert the enhanced image to base64 for display
            with open(enhanced_image_path, "rb") as image_file:
                enhanced_image_path = base64.b64encode(image_file.read()).decode('utf-8')

            # Render the template with the uploaded image, enhanced image, motif prediction, number of pages, and current page
            return render_template('index.html', data=images, uploaded_image=original_image_path, enhanced_image=enhanced_image_path, motif=prediction, num_pages=num_pages, page=page)
            
        else:
            return "Model state dictionary not found in the checkpoint."
        
    return redirect(url_for('main'))

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
