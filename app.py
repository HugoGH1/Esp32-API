from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)

# Habilita CORS para todas las rutas bajo /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Lista temporal en memoria
datos = []


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
        luz = int(data.get("luz"))
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

# =======================
#       MAIN
# =======================
if __name__ == "__main__":
    # debug=True para recarga automática en desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
