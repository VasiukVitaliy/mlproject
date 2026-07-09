from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline
from flask import Flask, request, jsonify
from flask.templating import render_template
from pathlib import Path
import threading

app = Flask(__name__)

train_pipe = TrainPipeline()

pipe = None
try:
    pipe = PredictPipeline(Path("./latest/model.pkl"), Path("./latest/preprocessor.pkl"))
except Exception as e:
    app.logger.warning(f"Модель ще не готова: {e}")

training_lock = threading.Lock()
is_training = False

def _run_training():
    global pipe, is_training
    try:
        train_pipe.train()
        pipe = PredictPipeline(Path("./latest/model.pkl"), Path("./latest/preprocessor.pkl"))
    except Exception as e:
        app.logger.error(f"Помилка тренування: {e}")
    finally:
        with training_lock:
            is_training = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/train", methods=["GET", "POST"])
def train():
    global is_training
    with training_lock:
        if is_training:
            return jsonify({"status": "already_training", "message": "Модель вже тренується"}), 202
        is_training = True

    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"status": "started", "message": "Модель тренується"}), 202

@app.route("/train/status", methods=["GET"])
def train_status():
    return jsonify({"is_training": is_training, "model_ready": pipe is not None})

@app.route("/predict", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')

    if is_training:
        return jsonify({"error": "Модель зараз тренується, спробуйте пізніше"}), 503
    if pipe is None:
        return jsonify({"error": "Модель ще не натренована"}), 503

    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('reading_score')),
        writing_score=float(request.form.get('writing_score'))
    )
    pred_df = data.get_data_as_data_frame()
    results = pipe.predict(pred_df)
    return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)