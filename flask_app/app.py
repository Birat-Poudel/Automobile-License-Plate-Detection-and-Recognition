from flask import Flask, request, render_template
import os
from deeplearning import ocr

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, "static/uploads/")

@app.route("/", methods=["GET", "POST"])
def get_homepage():
    
    if request.method == "POST":
        upload_file = request.files["image_name"]
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text = ocr(path_save, filename)
        return render_template("index.html")    
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)