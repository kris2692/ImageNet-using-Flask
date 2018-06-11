from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug import secure_filename

UPLOAD_FOLDER=r"D:\Flask-Image_net\uploads"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST','GET'])
def take_input():
  if request.method=='POST':
    model = ResNet50(weights='imagenet')
    image1 = request.files['picture']
    filename=secure_filename(image1.filename)
    image1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#    image_path=open(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename)),mode="rb")
    print(image1)
    img = image.load_img(image1, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
  return render_template('home.html')

if __name__=='__main__':
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'
  app.run()