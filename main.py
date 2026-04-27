from flask import Flask, request, jsonify, render_template, redirect, session, Response
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
import joblib
import re
import emoji
import math
import unicodedata
import datetime
import base64
import sys
from datetime import datetime as dt
import csv
import io

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

app.config['MYSQL_HOST'] = os.getenv("MYSQL_HOST")
app.config['MYSQL_USER'] = os.getenv("MYSQL_USER")
app.config['MYSQL_PASSWORD'] = os.getenv("MYSQL_PASSWORD")
app.config['MYSQL_DB'] = os.getenv("MYSQL_DB")
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# ================= LOAD ML MODEL =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "static", "model", "hate_speech_text_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "static", "model", "hate_speech_tfidf.pkl")

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)


def _ensure_analysis_table():
    cur = mysql.connection.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comment_analyses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            comment_text TEXT NOT NULL,
            prediction VARCHAR(20) NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_id_created_at (user_id, created_at),
            INDEX idx_created_at (created_at)
        )
        """
    )
    mysql.connection.commit()
    cur.close()


def _log_analysis(comment_text: str, prediction: str):
    try:
        _ensure_analysis_table()
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO comment_analyses (user_id, comment_text, prediction) VALUES (%s, %s, %s)",
            (session.get('user_id'), comment_text, prediction)
        )
        mysql.connection.commit()
        cur.close()
    except Exception as e:
        print("ANALYSIS LOG ERROR:", e)


def _compute_initials(name: str) -> str:
    if not isinstance(name, str):
        return "U"
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "U"
    first = parts[0][0]
    second = parts[1][0] if len(parts) > 1 else ""
    return (first + second).upper()


@app.context_processor
def inject_user_context():
    name = session.get('user_name')
    first_name = session.get('user_first_name')
    last_name = session.get('user_last_name')
    email = session.get('user_email')
    return {
        'user_display_name': name or 'User',
        'user_first_name': first_name or '',
        'user_last_name': last_name or '',
        'user_email': email or '',
        'user_initials': _compute_initials(name or 'User')
    }

# ================= TEXT PREPROCESSING =================

def emoji_to_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"[:_]", " ", text)
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = emoji_to_text(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\\1", text)
    text = re.sub(r"(.)\\1{2,}", r"\\1\\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= EMOJI RULE ENGINE =================

HATE_EMOJI_KEYWORDS = [
    "angry",
    "symbols",
    "skull",
    "bomb",
    "knife",
    "gun",
    "explosion",
    "death",
    "pistol"
]

def predict_hate(comment):
    processed = clean_text(comment)

    if processed.strip() == "":
        return "NON-HATE", 0.0

    # Emoji rule check
    emoji_hit = False
    for word in HATE_EMOJI_KEYWORDS:
        if word in processed:
            emoji_hit = True
            break

    # ML prediction
    vec = tfidf.transform([processed])
    pred = model.predict(vec)[0]

    hate_threshold = float(os.environ.get("HATE_THRESHOLD", "0.55"))
    proba_hate = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(vec)[0]
            if len(probs) >= 2:
                proba_hate = float(probs[1])
        except Exception:
            proba_hate = None
    elif hasattr(model, "decision_function"):
        try:
            score = model.decision_function(vec)
            if hasattr(score, "__len__"):
                score = score[0]
            score = float(score)
            proba_hate = 1.0 / (1.0 + math.exp(-score))
        except Exception:
            proba_hate = None

    if proba_hate is not None:
        if emoji_hit and proba_hate >= min(hate_threshold, 0.45):
            return "HATE", round(proba_hate, 4)
        if proba_hate >= hate_threshold:
            return "HATE", round(proba_hate, 4)
        return "NON-HATE", round(1.0 - proba_hate, 4)

    label = "HATE" if pred == 1 else "NON-HATE"
    if emoji_hit and label != "HATE":
        label = "HATE"
    confidence = 1.0
    return label, confidence

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/help-tour')
def help_tour():
    return render_template('help_tour.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password required"}), 400

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if not user or not check_password_hash(user['password'], password):
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        session['user_id'] = user['id']
        session['user_first_name'] = user.get('first_name')
        session['user_last_name'] = user.get('last_name')
        session['user_name'] = f"{(user.get('first_name') or '').strip()} {(user.get('last_name') or '').strip()}".strip() or user.get('first_name') or 'User'
        session['user_email'] = user.get('email')

        return jsonify({"success": True, "redirect": "/comment_analysis"})

    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/user_dashboard')
def user_dashboard():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('user/user_dashboard.html')


@app.route('/comment_analysis')
def comment_analysis():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('user/comment_analysis.html')


# ================= PREDICTION ROUTE =================

@app.route('/predict_hate', methods=['POST'])
def predict_hate_route():
    try:
        data = request.get_json()
        comment = data.get("comment")

        if not comment:
            return jsonify({"success": False, "message": "No comment provided"}), 400

        result, confidence = predict_hate(comment)

        if session.get('user_id'):
            _log_analysis(comment, result)

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/api/dashboard-metrics')
def dashboard_metrics():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        _ensure_analysis_table()
        user_id = session.get('user_id')
        cur = mysql.connection.cursor()

        cur.execute(
            "SELECT COUNT(*) AS c FROM comment_analyses WHERE user_id = %s",
            (user_id,)
        )
        total = int(cur.fetchone().get('c') or 0)

        cur.execute(
            "SELECT COUNT(*) AS c FROM comment_analyses WHERE user_id = %s AND prediction = 'HATE'",
            (user_id,)
        )
        hate = int(cur.fetchone().get('c') or 0)

        non_hate = max(total - hate, 0)

        today = datetime.date.today()
        start = datetime.datetime.combine(today, datetime.time.min)
        end = datetime.datetime.combine(today, datetime.time.max)
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM comment_analyses
            WHERE user_id = %s AND created_at BETWEEN %s AND %s
            """,
            (user_id, start, end)
        )
        today_total = int(cur.fetchone().get('c') or 0)

        cur.close()

        non_hate_pct = round((non_hate / total) * 100, 1) if total else 0.0
        hate_pct = round((hate / total) * 100, 1) if total else 0.0

        return jsonify({
            "success": True,
            "total": total,
            "hate": hate,
            "non_hate": non_hate,
            "hate_pct": hate_pct,
            "non_hate_pct": non_hate_pct,
            "today_total": today_total,
        })

    except Exception as e:
        print("DASHBOARD METRICS ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/api/history')
def api_history():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        _ensure_analysis_table()
        user_id = session.get('user_id')
        limit = request.args.get('limit', '50')
        try:
            limit = max(1, min(int(limit), 200))
        except Exception:
            limit = 50

        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT comment_text, prediction, created_at
            FROM comment_analyses
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit)
        )
        rows = cur.fetchall() or []
        cur.close()

        items = []
        for r in rows:
            items.append({
                "comment_text": r.get('comment_text') or '',
                "prediction": r.get('prediction') or '',
                "created_at": (r.get('created_at').isoformat(sep=' ') if r.get('created_at') else '')
            })

        return jsonify({"success": True, "items": items})

    except Exception as e:
        print("HISTORY API ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/api/history/export')
def export_history():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        _ensure_analysis_table()
        user_id = session.get('user_id')

        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT id, comment_text, prediction, created_at
            FROM comment_analyses
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        rows = cur.fetchall() or []
        cur.close()

        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(["id", "comment_text", "prediction", "created_at"])
        for r in rows:
            created_at = r.get('created_at')
            writer.writerow([
                r.get('id') or "",
                r.get('comment_text') or "",
                r.get('prediction') or "",
                (created_at.isoformat(sep=' ') if created_at else "")
            ])

        csv_text = output.getvalue()
        output.close()

        ts = dt.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comment_history_{ts}.csv"
        return Response(
            csv_text,
            mimetype='text/csv; charset=utf-8',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Cache-Control': 'no-store'
            }
        )

    except Exception as e:
        print("HISTORY EXPORT ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/history')
def history():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('/user/history.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    try:
        data = request.get_json()
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        email = data.get('email')
        password = data.get('password')

        if not all([first_name, last_name, email, password]):
            return jsonify({"success": False, "message": "All fields required"}), 400

        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({"success": False, "message": "Email already exists"}), 409

        hashed_password = generate_password_hash(password)

        cur.execute(
            "INSERT INTO users (first_name, last_name, email, password) VALUES (%s, %s, %s, %s)",
            (first_name, last_name, email, hashed_password)
        )

        mysql.connection.commit()
        cur.close()

        return jsonify({"success": True, "redirect": "/login"})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/settings')
def settings():
    if not session.get('user_id'):
        return redirect('/login')
    return render_template('/user/settings.html')


@app.route('/api/profile', methods=['POST'])
def update_profile():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}
        first_name = (data.get('first_name') or '').strip()
        last_name = (data.get('last_name') or '').strip()

        if not first_name:
            return jsonify({"success": False, "message": "First name is required"}), 400

        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE users SET first_name = %s, last_name = %s WHERE id = %s",
            (first_name, last_name, session['user_id'])
        )
        mysql.connection.commit()
        cur.close()

        session['user_first_name'] = first_name
        session['user_last_name'] = last_name
        session['user_name'] = f"{first_name} {last_name}".strip()

        return jsonify({
            "success": True,
            "user_display_name": session['user_name'],
            "user_first_name": first_name,
            "user_last_name": last_name
        })

    except Exception as e:
        print("UPDATE PROFILE ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/api/change-password', methods=['POST'])
def change_password():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        if not old_password or not new_password:
            return jsonify({"success": False, "message": "Old password and new password required"}), 400

        if len(new_password) < 6:
            return jsonify({"success": False, "message": "New password must be at least 6 characters"}), 400

        cur = mysql.connection.cursor()
        cur.execute("SELECT password FROM users WHERE id = %s", (session['user_id'],))
        user = cur.fetchone()

        if not user or not check_password_hash(user['password'], old_password):
            cur.close()
            return jsonify({"success": False, "message": "Old password is incorrect"}), 400

        hashed_password = generate_password_hash(new_password)
        cur.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password, session['user_id']))
        mysql.connection.commit()
        cur.close()

        return jsonify({"success": True})

    except Exception as e:
        print("CHANGE PASSWORD ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500


@app.route('/api/delete-account', methods=['POST'])
def delete_account():
    if not session.get('user_id'):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        user_id = session.get('user_id')

        _ensure_analysis_table()
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM comment_analyses WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        mysql.connection.commit()
        cur.close()

        session.clear()
        return jsonify({"success": True, "redirect": "/login"})

    except Exception as e:
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        print("DELETE ACCOUNT ERROR:", e)
        return jsonify({"success": False, "message": "Server error"}), 500




@app.route('/get-started')
def get_started():
    if session.get('user_id'):
        return redirect('/user_dashboard')
    return redirect('/register')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
