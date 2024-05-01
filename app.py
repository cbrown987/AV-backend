import secrets
from io import StringIO

from flask import Flask, jsonify, request, session, send_file

from model.create_image import DiffusionModel

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

session["model"] = DiffusionModel()


@app.route('/', methods=["GET", "POST"])
def generate_images():
    content = request.json
    if not ("model" in session):
        session["model"] = DiffusionModel()

    model = session["model"]
    model.set_model_variables(**content)

    model.create_models()
    pil_img = model.create_image()

    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
