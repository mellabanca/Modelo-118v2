from flask import Flask, render_template, request, jsonify
from model_prediction import *

app = Flask(__name__)

text=""
predicted_emotion=""
predicted_emotion_img_url=""

#renderize a página HTML
@app.route("/")
def home():
    entries = show_entry()
    return render_template("index.html", entries=entries)
    
#preveja a emoção
@app.route("/predict-emotion", methods=["POST"])
def predict_emotion():
    input_text = request.json.get("text")
    if not input_text:
        return jsonify({
            "status": "error",
            "message": "Digite um texto para prever a emoção!"
        }), 400
    else:
        predicted_emotion, predicted_emotion_img_url = predict(input_text)                         
        return jsonify({
            "data": {
                "predicted_emotion": predicted_emotion,
                "predicted_emotion_img_url": predicted_emotion_img_url
            },
            "status": "success"
        }), 200
        
# Salve a entrada
@app.route("/save-entry", methods=["POST"])
def save_entry():

    # Obtenha a data, a emoção prevista e o texto digitado pelo usuário para salvar a entrada
    date = request.json.get("date")           
    emotion = request.json.get("emotion")
    save_text = request.json.get("text")

    save_text = save_text.replace("\n", " ")

    # Entrada CSV
    entry = f'{date},{save_text},{emotion}\n'  

    with open("./static/assets/data_files/data_entry.csv", "a") as f:
        f.write(entry)
    return jsonify("Success")
           
                
if __name__ == "__main__":
    app.run(debug=True)



