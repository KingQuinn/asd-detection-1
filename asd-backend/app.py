import logging
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from asd_core import InputError, get_supabase, load_model, predict, save_prediction

# The React frontend (index.html) lives in the project root, one level up.
FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)
log = logging.getLogger(__name__)

bundle = load_model()

# Optional Supabase logging of predictions. Set SUPABASE_URL and SUPABASE_KEY
# in the environment to enable it; the API works without them.
supabase = get_supabase()


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "database": supabase is not None, "model": bundle["versions"]})


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        label, proba = predict(bundle, data)
    except InputError as e:
        return jsonify({"error": str(e)}), 400

    result = "ASD Positive" if label == 1 else "ASD Negative"

    # Store only when the user ticked the consent box.
    saved = False
    if supabase is not None and data.get("consent") is True:
        try:
            save_prediction(supabase, data, label)
            saved = True
        except Exception:
            # A database outage should not block the screening result.
            log.exception("Failed to store prediction in Supabase")

    return jsonify({"result": result, "probability": round(proba, 4), "saved": saved})


if __name__ == '__main__':
    # Port 5000 is taken by AirPlay Receiver on macOS, so default to 5001.
    app.run(port=int(os.environ.get("PORT", 5001)), debug=os.environ.get("FLASK_DEBUG") == "1")
