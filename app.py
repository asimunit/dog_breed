import json
from flask import Flask, render_template, request, url_for, redirect, Response
from flask_mysqldb import MySQL
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import glob
import random
import base64
from PIL import Image
from io import BytesIO
import json
import os

app = Flask(__name__)



@app.route('/imagedetection', methods=['POST'])
def imagedetection():
    print("asim")
    data = request.get_json()
    print(type(data))
    
    
    base = data["image"]
    print(data)
    resp = "done"

    im = Image.open(BytesIO(base64.b64decode(base)))
    im.save('img.jpg', 'JPEG')
  
    files=(glob.glob("img.jpg"))
    MODEL='dog_breed.pth'

    # Load the model for testing
    model = torch.load(MODEL)
    model.eval()

    # Class labels for prediction
    class_names=class_names=['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
    images=random.sample(list(files),1)

    # Preprocessing transformations
    preprocess=transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    # Enable gpu mode, if cuda available
    device = torch.device("cpu")

    # Perform prediction and plot results
    with torch.no_grad():
        for num,img in enumerate(images):
            img=Image.open(img).convert('RGB')
            inputs=preprocess(img).unsqueeze(0).to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)    
            output=class_names[preds]
    
    res = {"breed":output}
    resp = Response({"breed":output}, status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return res



if __name__ == '__main__':
    # it allow to run it on default port: 5000 , ip:127.0.0.1  ==> http://127.0.0.1:5000/
    app.run(debug=True,host='0.0.0.0', port=5004)
