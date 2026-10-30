import os, csv, io, sqlite3, json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, abort
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("STUDENT_DB_PATH", os.path.join(APP_DIR, "students.db"))
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-before-deployment")

SUBJECTS = ["Python", "Mathematics", "Computer Networks", "DBMS", "Web Development"]

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'teacher'
        );
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT DEFAULT '',
            course TEXT DEFAULT 'MCA',
            semester INTEGER DEFAULT 1,
            attendance REAL DEFAULT 0,
            assignment_score REAL DEFAULT 0,
            study_hours REAL DEFAULT 0,
            previous_score REAL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            subject TEXT NOT NULL,
            score REAL NOT NULL CHECK(score >= 0 AND score <= 100),
            exam_type TEXT DEFAULT 'Internal',
            exam_date TEXT DEFAULT '',
            UNIQUE(student_id, subject, exam_type)
        );
        """)
        if not db.execute("SELECT id FROM users WHERE username='admin'").fetchone():
            db.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                       ("admin", generate_password_hash("admin123"), "admin"))
        count = db.execute("SELECT COUNT(*) AS n FROM students").fetchone()["n"]
        if count == 0:
            demo = [
                ("MCA001","Aarav Sharma","aarav@example.com","MCA",1,94,88,3.5,78),
                ("MCA002","Diya Singh","diya@example.com","MCA",1,87,81,2.8,72),
                ("MCA003","Kabir Verma","kabir@example.com","MCA",1,68,62,1.2,55),
                ("MCA004","Meera Patel","meera@example.com","MCA",1,97,93,4.0,88),
                ("MCA005","Arjun Gupta","arjun@example.com","MCA",1,73,70,2.0,61),
                ("MCA006","Ananya Kaur","ananya@example.com","MCA",1,82,79,2.6,70),
                ("MCA007","Ishaan Das","ishaan@example.com","MCA",1,58,52,0.8,48),
                ("MCA008","Sara Khan","sara@example.com","MCA",1,90,86,3.2,82),
            ]
            for row in demo:
                cur = db.execute("""INSERT INTO students(roll_no,name,email,course,semester,attendance,assignment_score,study_hours,previous_score,created_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""", (*row, datetime.now().isoformat(timespec="seconds")))
                sid = cur.lastrowid
                scores = [min(100,max(20,row[8] + d)) for d in (-8, -3, 2, 5, 0)]
                for subj, score in zip(SUBJECTS, scores):
                    db.execute("INSERT INTO marks(student_id,subject,score,exam_type,exam_date) VALUES(?,?,?,?,?)",
                               (sid, subj, score, "Internal", datetime.now().date().isoformat()))

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def get_student(student_id):
    with connect() as db:
        row = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
        if not row: abort(404)
        marks = db.execute("SELECT * FROM marks WHERE student_id=? ORDER BY subject", (student_id,)).fetchall()
    scores = [m["score"] for m in marks]
    avg = round(sum(scores)/len(scores), 1) if scores else 0
    return dict(row), [dict(m) for m in marks], avg

def prediction_for(student, avg):
    # ML model is trained from the available class records; fallback is transparent and deterministic.
    with connect() as db:
        rows = db.execute("""SELECT s.attendance,s.assignment_score,s.study_hours,s.previous_score,
                             AVG(m.score) AS avg_score
                             FROM students s LEFT JOIN marks m ON s.id=m.student_id
                             GROUP BY s.id HAVING avg_score IS NOT NULL""").fetchall()
    predicted = None
    method = "weighted baseline"
    if ML_AVAILABLE and len(rows) >= 6:
        X = np.array([[r["attendance"], r["assignment_score"], r["study_hours"], r["previous_score"], r["avg_score"]] for r in rows], dtype=float)
        y = np.array([min(100, max(0, 0.25*r["attendance"] + 0.2*r["assignment_score"] + 0.15*min(r["study_hours"]*20,100) + 0.2*r["previous_score"] + 0.2*r["avg_score"])) for r in rows])
        model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1)
        model.fit(X, y)
        predicted = float(model.predict(np.array([[student["attendance"],student["assignment_score"],student["study_hours"],student["previous_score"],avg]], dtype=float))[0])
        method = "Random Forest model"
    else:
        predicted = .25*student["attendance"] + .2*student["assignment_score"] + .15*min(student["study_hours"]*20,100) + .2*student["previous_score"] + .2*avg
    predicted = round(min(100, max(0, predicted)), 1)
    risk = "Low"
    if predicted < 50 or student["attendance"] < 65:
        risk = "High"
    elif predicted < 70 or student["attendance"] < 75:
        risk = "Medium"
    if predicted >= 85: grade = "A"
    elif predicted >= 70: grade = "B"
    elif predicted >= 55: grade = "C"
    elif predicted >= 40: grade = "D"
    else: grade = "Needs support"
    suggestions = []
    if student["attendance"] < 75: suggestions.append("Improve attendance; aim for at least 75%.")
    if student["assignment_score"] < 70: suggestions.append("Complete pending assignments and ask for feedback.")
    if student["study_hours"] < 2: suggestions.append("Build a consistent daily study routine of 2–3 focused hours.")
    if avg < 60: suggestions.append("Revise weak subjects and practise questions every week.")
    if not suggestions: suggestions.append("Maintain the current routine and attempt advanced practice questions.")
    return {"score": predicted, "risk": risk, "grade": grade, "method": method, "suggestions": suggestions}

@app.route("/")
def home():
    if session.get("user_id"): return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        with connect() as db:
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Incorrect username or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    with connect() as db:
        students = db.execute("""SELECT s.*, COALESCE(AVG(m.score),0) AS avg_score
          FROM students s LEFT JOIN marks m ON s.id=m.student_id GROUP BY s.id ORDER BY s.name""").fetchall()
    total = len(students)
    avg = round(sum(s["avg_score"] for s in students)/total,1) if total else 0
    at_risk = sum(1 for s in students if s["avg_score"] < 50 or s["attendance"] < 65)
    top = sorted(students, key=lambda s: s["avg_score"], reverse=True)[:5]
    return render_template("dashboard.html", students=students, total=total, avg=avg, at_risk=at_risk, top=top)

@app.route("/students")
@login_required
def students_page():
    q = request.args.get("q","").strip()
    risk = request.args.get("risk","")
    with connect() as db:
        students = db.execute("""SELECT s.*, COALESCE(AVG(m.score),0) AS avg_score
            FROM students s LEFT JOIN marks m ON s.id=m.student_id
            WHERE s.name LIKE ? OR s.roll_no LIKE ? GROUP BY s.id ORDER BY s.name""",
            (f"%{q}%",f"%{q}%")).fetchall()
    result = []
    for s in students:
        row = dict(s)
        row["risk"] = "High" if row["avg_score"] < 50 or row["attendance"] < 65 else ("Medium" if row["avg_score"] < 70 or row["attendance"] < 75 else "Low")
        if not risk or row["risk"] == risk: result.append(row)
    return render_template("students.html", students=result, q=q, risk=risk)

@app.route("/students/add", methods=["GET","POST"])
@app.route("/students/<int:student_id>/edit", methods=["GET","POST"])
@login_required
def student_form(student_id=None):
    student = None
    if student_id:
        with connect() as db: student = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
        if not student: abort(404)
    if request.method == "POST":
        try:
            data = {
                "roll_no": request.form["roll_no"].strip(),
                "name": request.form["name"].strip(),
                "email": request.form.get("email","").strip(),
                "course": request.form.get("course","MCA").strip(),
                "semester": int(request.form.get("semester",1)),
                "attendance": float(request.form.get("attendance",0)),
                "assignment_score": float(request.form.get("assignment_score",0)),
                "study_hours": float(request.form.get("study_hours",0)),
                "previous_score": float(request.form.get("previous_score",0)),
            }
            if not data["roll_no"] or not data["name"]: raise ValueError("Roll number and name are required.")
            for key in ("attendance","assignment_score","previous_score"):
                if not 0 <= data[key] <= 100: raise ValueError(f"{key.replace('_',' ').title()} must be between 0 and 100.")
            if data["study_hours"] < 0 or data["study_hours"] > 24: raise ValueError("Study hours must be between 0 and 24.")
            with connect() as db:
                if student_id:
                    db.execute("""UPDATE students SET roll_no=?,name=?,email=?,course=?,semester=?,attendance=?,assignment_score=?,study_hours=?,previous_score=? WHERE id=?""",
                        (*data.values(), student_id))
                else:
                    db.execute("""INSERT INTO students(roll_no,name,email,course,semester,attendance,assignment_score,study_hours,previous_score,created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""", (*data.values(), datetime.now().isoformat(timespec="seconds")))
            flash("Student saved successfully.", "success")
            return redirect(url_for("students_page"))
        except sqlite3.IntegrityError:
            flash("That roll number already exists.", "error")
        except (ValueError, KeyError) as e:
            flash(str(e) or "Please check the entered values.", "error")
    return render_template("student_form.html", student=student)

@app.route("/students/<int:student_id>")
@login_required
def student_detail(student_id):
    student, marks, avg = get_student(student_id)
    prediction = prediction_for(student, avg)
    return render_template("student_detail.html", student=student, marks=marks, avg=avg, prediction=prediction, subjects=SUBJECTS)

@app.route("/students/<int:student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    with connect() as db: db.execute("DELETE FROM students WHERE id=?", (student_id,))
    flash("Student and associated marks deleted.", "success")
    return redirect(url_for("students_page"))

@app.route("/students/<int:student_id>/marks", methods=["POST"])
@login_required
def save_marks(student_id):
    try:
        with connect() as db:
            for subject in SUBJECTS:
                value = request.form.get(subject)
                if value is None or value == "": continue
                score = float(value)
                if not 0 <= score <= 100: raise ValueError(f"{subject}: score must be between 0 and 100.")
                db.execute("""INSERT INTO marks(student_id,subject,score,exam_type,exam_date) VALUES(?,?,?,?,?)
                  ON CONFLICT(student_id,subject,exam_type) DO UPDATE SET score=excluded.score, exam_date=excluded.exam_date""",
                  (student_id, subject, score, request.form.get("exam_type","Internal"), request.form.get("exam_date","")))
        flash("Marks saved successfully.", "success")
    except ValueError as e: flash(str(e), "error")
    return redirect(url_for("student_detail", student_id=student_id))

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction_page():
    result = None
    form = {
        "attendance": "85",
        "assignment_score": "80",
        "study_hours": "3",
        "previous_score": "70",
        "current_average": "72",
    }
    if request.method == "POST":
        try:
            for key in form:
                form[key] = request.form.get(key, "").strip()
                if form[key] == "":
                    raise ValueError("Please complete every field before running a prediction.")
            attendance = float(form["attendance"])
            assignment_score = float(form["assignment_score"])
            study_hours = float(form["study_hours"])
            previous_score = float(form["previous_score"])
            current_average = float(form["current_average"])
            for label, value in [
                ("Attendance", attendance),
                ("Assignment score", assignment_score),
                ("Previous exam score", previous_score),
                ("Current average", current_average),
            ]:
                if not 0 <= value <= 100:
                    raise ValueError(f"{label} must be between 0 and 100.")
            if not 0 <= study_hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            # Keep the feature set and inference method consistent with the student-profile predictor.
            synthetic_student = {
                "attendance": attendance,
                "assignment_score": assignment_score,
                "study_hours": study_hours,
                "previous_score": previous_score,
            }
            predicted = prediction_for(synthetic_student, current_average)
            # Show a transparent indicator breakdown for the explanation panel.
            contribution = {
                "Attendance": round(attendance * 0.25, 1),
                "Assignments": round(assignment_score * 0.20, 1),
                "Study routine": round(min(study_hours * 20, 100) * 0.15, 1),
                "Previous score": round(previous_score * 0.20, 1),
                "Current average": round(current_average * 0.20, 1),
            }
            result = {
                **predicted,
                "inputs": {
                    "attendance": attendance,
                    "assignment_score": assignment_score,
                    "study_hours": study_hours,
                    "previous_score": previous_score,
                    "current_average": current_average,
                },
                "contribution": contribution,
            }
        except ValueError as e:
            flash(str(e), "error")
    return render_template("prediction.html", result=result, form=form)

@app.route("/analytics")
@login_required
def analytics():
    with connect() as db:
        rows = db.execute("""SELECT s.name,s.roll_no,s.attendance,s.assignment_score,s.study_hours,s.previous_score,
          COALESCE(AVG(m.score),0) AS avg_score FROM students s LEFT JOIN marks m ON s.id=m.student_id
          GROUP BY s.id ORDER BY s.name""").fetchall()
    labels = [r["name"] for r in rows]
    averages = [round(r["avg_score"],1) for r in rows]
    attendance = [round(r["attendance"],1) for r in rows]
    subject_data = {}
    with connect() as db:
        for r in db.execute("SELECT subject, ROUND(AVG(score),1) AS avg FROM marks GROUP BY subject"):
            subject_data[r["subject"]] = r["avg"]
    return render_template("analytics.html", labels=json.dumps(labels), averages=json.dumps(averages),
                           attendance=json.dumps(attendance), subject_labels=json.dumps(list(subject_data.keys())),
                           subject_scores=json.dumps(list(subject_data.values())), rows=rows)

@app.route("/reports/export.csv")
@login_required
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Roll No","Name","Email","Course","Semester","Attendance %","Assignment Score","Study Hours/Day","Previous Score","Average Marks","Predicted Score","Risk"])
    with connect() as db:
        rows = db.execute("""SELECT s.*, COALESCE(AVG(m.score),0) AS avg_score FROM students s
          LEFT JOIN marks m ON s.id=m.student_id GROUP BY s.id ORDER BY s.roll_no""").fetchall()
    for r in rows:
        student = dict(r); prediction = prediction_for(student, student["avg_score"])
        writer.writerow([student["roll_no"],student["name"],student["email"],student["course"],student["semester"],student["attendance"],student["assignment_score"],student["study_hours"],student["previous_score"],round(student["avg_score"],1),prediction["score"],prediction["risk"]])
    return Response("\ufeff"+output.getvalue(), mimetype="text/csv", headers={"Content-Disposition":"attachment; filename=student_performance_report.csv"})

@app.route("/reports/import", methods=["GET","POST"])
@login_required
def import_csv():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please choose a .csv file.", "error")
            return redirect(url_for("import_csv"))
        try:
            stream = io.StringIO(file.stream.read().decode("utf-8-sig"))
            reader = csv.DictReader(stream)
            required = {"roll_no","name"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError("CSV must include at least roll_no and name columns.")
            added = 0
            with connect() as db:
                for row in reader:
                    roll = (row.get("roll_no") or "").strip()
                    name = (row.get("name") or "").strip()
                    if not roll or not name: continue
                    def num(k, default=0): return float(row.get(k) or default)
                    db.execute("""INSERT INTO students(roll_no,name,email,course,semester,attendance,assignment_score,study_hours,previous_score,created_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(roll_no) DO UPDATE SET name=excluded.name,email=excluded.email,course=excluded.course,semester=excluded.semester,attendance=excluded.attendance,assignment_score=excluded.assignment_score,study_hours=excluded.study_hours,previous_score=excluded.previous_score""",
                     (roll,name,row.get("email",""),row.get("course","MCA"),int(num("semester",1)),num("attendance"),num("assignment_score"),num("study_hours"),num("previous_score"),datetime.now().isoformat(timespec="seconds")))
                    added += 1
            flash(f"Imported/updated {added} student record(s). Add marks from each student profile.", "success")
        except Exception as e:
            flash(f"Import failed: {e}", "error")
        return redirect(url_for("students_page"))
    return render_template("import.html")

@app.route("/api/summary")
@login_required
def api_summary():
    with connect() as db:
        count = db.execute("SELECT COUNT(*) n FROM students").fetchone()["n"]
        avg = db.execute("SELECT AVG(score) n FROM marks").fetchone()["n"] or 0
        high = db.execute("""SELECT COUNT(*) n FROM students s LEFT JOIN marks m ON s.id=m.student_id
            GROUP BY s.id HAVING COALESCE(AVG(m.score),0)<50 OR s.attendance<65""").fetchall()
    return {"students":count,"class_average":round(avg,1),"at_risk":len(high)}

@app.errorhandler(404)
def not_found(_): return render_template("error.html", code=404, message="The requested page or student could not be found."), 404

if __name__ == "__main__":
    init_db()
    app.run(debug=os.environ.get("FLASK_DEBUG","0") == "1")
