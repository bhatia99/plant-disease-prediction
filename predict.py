from flask import Flask, request, render_template, send_from_directory
from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np


path = Path('data/plantdatafull')
np.random.seed(42)
data = ImageDataBunch.from_folder(path,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics = error_rate)
learn = load_learner(path)

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        #print(destination)
        file.save(destination)
        pred_class,pred_idx,outputs = learn.predict(destination)
        pred_class

    return render_template('report.html',pred_class=pred_class)


if __name__ == '__main__':
    app.run(debug=True)
