
import torch
import torchvision.transforms as transforms

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from flask_cors import CORS
from datetime import datetime

from werkzeug.utils import secure_filename



app = Flask(__name__)



# Cargar modelo (ya entrenado y guardado con torch.save)
model = torch.load("./resnet18_model1.pth", map_location="cpu", weights_only=False)
model.eval()

# Clases (asegúrate de que coincidan con tu entrenamiento)
classes = ["defective", "ok"]

# Transformaciones de imagen (mismo preprocesado que usaste en entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Habilita CORS para todas las rutas bajo /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Lista temporal en memoria
datos = []
file = None
last_prediction = None
last_certainty = None


# =======================
#       ENDPOINTS
# =======================

@app.route("/", methods=["GET"])
def index():
    """Ruta raíz: muestra estado y endpoints disponibles."""
    return jsonify({
        "status": "ok",
        "mensaje": "hola mundo",
        "endpoints": [
            "GET  /api/sensores",
            "POST /api/sensores",
            "GET  /api/sensores/ultimo",
            "GET /api/sensores/penultimo"
        ]
    })


@app.route("/api/sensores", methods=["POST"])
def recibir_datos():
    """Recibir un nuevo dato de sensor."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Cuerpo no es JSON válido"}), 400

    # Valida campos requeridos
    faltantes = [k for k in ("temperatura", "humedad","luz") if k not in data]
    if faltantes:
        return jsonify({"error": "Faltan campos", "faltantes": faltantes}), 400

    # Intenta convertir a float
    try:
        temperatura = float(data.get("temperatura"))
        humedad = float(data.get("humedad"))
        luz = float(data.get("luz"))
    except (TypeError, ValueError):
        return jsonify({"error": "temperatura, humedad y luz deben ser valores numéricos"}), 400

    registro = {
        "temperatura": temperatura,
        "humedad": humedad,
        "luz": luz,
        "fecha": datetime.now().isoformat(timespec="seconds")
    }
    datos.append(registro)
    return jsonify({"mensaje": "Dato guardado", "registro": registro}), 201


@app.route("/api/sensores", methods=["GET"])
def obtener_datos():
    """Obtener todos los datos almacenados."""
    return jsonify({"count": len(datos), "items": datos})


@app.route("/api/sensores/ultimo", methods=["GET"])
def ultimo():
    """Obtener el último dato registrado."""
    return jsonify(datos[-1] if datos else {}), (200 if datos else 204)

@app.route("/api/sensores/penultimo", methods=["GET"])
def penultimo():
    """Obtener el penúltimo dato registrado."""
    return jsonify(datos[-2] if len(datos) > 1 else {}), (200 if len(datos) > 1 else 204)

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    global file, last_prediction, last_certainty
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    image.save("uploaded_image.jpg")
    # Preprocesar
    img_tensor = transform(image).unsqueeze(0)

    # Pasar al modelo
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

    prediction = classes[predicted.item()]
    print(outputs)

    last_prediction = prediction
    last_certainty = confidence

    return jsonify({
        "prediction": prediction,
        "confidence": float(confidence)
    })

# Nueva ruta para obtener la última predicción y la imagen
@app.route("/api/predict/last", methods=["GET"])
def get_last_prediction():
    global file, last_prediction, last_certainty
    if file is None or last_prediction is None or last_certainty is None:
        return jsonify({"error": "No hay datos guardados"}), 404

    import base64
    from io import BytesIO
    image = Image.open("uploaded_image.jpg").convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "image_url": './uploaded_image.jpg',
        "prediction": last_prediction,
        "confidence": float(last_certainty)
    })

@app.route("/uploaded_image.jpg")
def uploaded_file():
    return send_from_directory('./', "uploaded_image.jpg")
# =======================
#       MAIN
# =======================
if __name__ == "__main__":
    # debug=True para recarga automática en desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
