from flask import *
from malaria import Cell
import os

app = Flask(__name__)


@app.route("/")
def upload():
    return render_template('upload.html')

@app.route('/results',methods=['POST'])
def success():
    f = request.files['file']
    file = f.filename
    if not os.path.isdir('static'):
        os.mkdir('static')
    path = 'static/'+file
    f.save(path)

    image = Cell(path)
    predicted,confidence = image.detect()
    return render_template('result.html',file=file,pred = predicted, con = confidence)


if __name__ == '__main__':
    app.run(debug=True)
