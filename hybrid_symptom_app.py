import json
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify, render_template_string, g
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -------------------- App Setup --------------------
app = Flask(__name__)
app.config["DATABASE"] = "symptom_logs.db"

# -------------------- Database Helpers --------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(app.config["DATABASE"])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    # Create table if not exists with new columns
    db.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        user_id TEXT,
        name TEXT,
        age INTEGER,
        days INTEGER,
        symptoms_json TEXT NOT NULL,
        predicted_condition TEXT,
        predicted_severity TEXT
    );
    """)
    db.commit()

with app.app_context():
    init_db()

# -------------------- Load Dataset and Train Model --------------------
# Make sure you have the CSV file generated: symptom_dataset_with_severity.csv
df = pd.read_csv("symptom_dataset_with_severity.csv")

# ML Pipeline: TF-IDF vectorizer on symptoms + Random Forest
model = Pipeline([
    ("tfidf", TfidfVectorizer(token_pattern=r"[^,]+")),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train model on symptoms -> condition
model.fit(df["symptoms"], df["condition"])

# Map condition -> severity
severity_map = dict(zip(df["condition"], df["severity"]))

# -------------------- HTML Template --------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Symptom Logger & ML Suggestion</title>
<style>
body {font-family:sans-serif; background:#111827; color:#eef2ff; padding:20px;}
input,button{padding:8px; margin:5px;}
.card{background:#1f2a44; padding:15px; margin-bottom:20px; border-radius:10px;}
.badge{padding:4px 8px;border-radius:8px;}
.b-self{background:#2e8b57;}
.b-tele{background:#e0a800;}
.b-urgent{background:#dc3545;}
.b-emer{background:#ff0000;}
table{border-collapse: collapse;}
th,td{padding:6px;}
</style>
</head>
<body>
<div class="card">
<h2>🩺 Symptom Logger & Next Steps (ML)</h2>
<form method="POST" action="/">
<label>Name:</label><br>
<input name="name" placeholder="Your name" value="{{ form_name }}"><br>
<label>Age:</label><br>
<input name="age" type="number" placeholder="Age" value="{{ form_age }}"><br>
<label>Days of Symptoms:</label><br>
<input name="days" type="number" placeholder="How many days" value="{{ form_days }}"><br>
<label>Symptoms (comma separated):</label><br>
<input name="symptoms" placeholder="e.g., fever, cough, sore throat" size="50" value="{{ form_symptoms }}"><br>
<button type="submit">Get Suggestion</button>
</form>
</div>

{% if result %}
<div class="card">
<h3>Predicted Condition: {{ result.predicted_condition }}</h3>
<span class="badge {% if result.predicted_severity=='self-care' %}b-self{% elif result.predicted_severity=='teleconsult-24h' %}b-tele{% elif result.predicted_severity=='urgent' %}b-urgent{% else %}b-emer{% endif %}">
Severity: {{ result.predicted_severity }}
</span>
<div>
<strong>Name:</strong> {{ result.name }} | <strong>Age:</strong> {{ result.age }} | <strong>Days:</strong> {{ result.days }}<br>
<strong>Input Symptoms:</strong> {{ result.input_symptoms }}
</div>
</div>
{% endif %}

<div class="card">
<h3>Recent Logs</h3>
<table border="1" style="color:#eef2ff;">
<tr><th>Time</th><th>Name</th><th>Age</th><th>Days</th><th>Symptoms</th><th>Predicted Condition</th><th>Severity</th></tr>
{% for row in logs %}
<tr>
<td>{{ row.ts }}</td>
<td>{{ row.name }}</td>
<td>{{ row.age }}</td>
<td>{{ row.days }}</td>
<td>{{ row.symptoms }}</td>
<td>{{ row.predicted_condition }}</td>
<td>{{ row.predicted_severity }}</td>
</tr>
{% endfor %}
</table>
</div>
</body>
</html>
"""

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    form_name = ""
    form_age = ""
    form_days = ""
    form_symptoms = ""
    result = None

    if request.method == "POST":
        form_name = request.form.get("name", "").strip()
        form_age = request.form.get("age", "").strip()
        form_days = request.form.get("days", "").strip()
        form_symptoms = request.form.get("symptoms", "").strip()

        if form_symptoms:
            pred_condition = model.predict([form_symptoms])[0]
            pred_severity = severity_map.get(pred_condition, "self-care")

            result = {
                "name": form_name,
                "age": form_age,
                "days": form_days,
                "input_symptoms": form_symptoms,
                "predicted_condition": pred_condition,
                "predicted_severity": pred_severity
            }

            # Save to DB
            db = get_db()
            db.execute(
                "INSERT INTO logs (ts, user_id, name, age, days, symptoms_json, predicted_condition, predicted_severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), "demo_user", form_name, form_age, form_days, json.dumps(form_symptoms),
                 pred_condition, pred_severity)
            )
            db.commit()

    # Fetch last 10 logs
    db = get_db()
    rows = db.execute(
        "SELECT ts, name, age, days, symptoms_json, predicted_condition, predicted_severity FROM logs ORDER BY id DESC LIMIT 10"
    ).fetchall()

    logs = []
    for r in rows:
        logs.append({
            "ts": r["ts"].replace("T", " ").split(".")[0],
            "name": r["name"],
            "age": r["age"],
            "days": r["days"],
            "symptoms": json.loads(r["symptoms_json"]),
            "predicted_condition": r["predicted_condition"],
            "predicted_severity": r["predicted_severity"]
        })

    return render_template_string(INDEX_HTML, form_name=form_name, form_age=form_age, form_days=form_days,
                                  form_symptoms=form_symptoms, result=result, logs=logs)

# -------------------- API Endpoint --------------------
@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    symptoms = data.get("symptoms", "")
    if not symptoms:
        return jsonify({"error": "Provide 'symptoms' field."}), 400

    pred_condition = model.predict([symptoms])[0]
    pred_severity = severity_map.get(pred_condition, "self-care")

    # Log
    db = get_db()
    db.execute(
        "INSERT INTO logs (ts, user_id, name, age, days, symptoms_json, predicted_condition, predicted_severity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), data.get("user_id", "api_user"), data.get("name", ""), data.get("age", ""),
         data.get("days", ""), json.dumps(symptoms), pred_condition, pred_severity)
    )
    db.commit()

    return jsonify({"predicted_condition": pred_condition, "predicted_severity": pred_severity})

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
