import flask
import werkzeug
import keras.models
import numpy
import scipy.misc
import cv2

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    print(flask.request.files)
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    img = cv2.imread(filename)#, mode="L")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow(img)
    img = img.reshape(784)
    loaded_model = keras.models.load_model('model.h5')
    predicted_label = loaded_model.predict_classes(numpy.array([img]))[0]
    print(predicted_label)

    return str(predicted_label)

app.run(host="0.0.0.0", port=8080, debug=True)
