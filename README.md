# EduTrack — Student Performance Analysis & Prediction System

A full-stack academic analytics web application with **Python + Flask** at its core, SQLite persistence, responsive HTML/CSS, JavaScript charts, and a scikit-learn prediction model.

## Features

- Login-protected teacher dashboard with demo data and class overview
- Student CRUD: create, view, edit, delete, search, and filter by risk
- Subject marks management for Python, Mathematics, Computer Networks, DBMS, and Web Development
- Dedicated, interactive Prediction Lab with score estimation, risk classification, indicator breakdown, tailored action plan, and print-ready summary
- Student-profile prediction estimate, risk category, indicative grade, and tailored support suggestions
- Random Forest regression using available class records when scikit-learn is installed; transparent weighted baseline fallback otherwise
- Analytics charts: student averages, subject averages, attendance vs marks
- CSV import for student profile data and CSV report export including prediction/risk
- SQLite database auto-created on first launch with demo students
- JSON summary endpoint at `/api/summary`
- Input validation and cascading deletion of marks when a student is deleted
- Responsive UI for desktop and mobile

## Tech stack

- **Backend:** Python 3, Flask
- **Database:** SQLite (built into Python)
- **Machine learning:** scikit-learn RandomForestRegressor + NumPy
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js (loaded from CDN)
- **Security helpers:** Werkzeug password hashing and Flask sessions

## Run locally (Windows)

1. Install Python 3.10 or newer.
2. Extract the project ZIP and open the folder in VS Code.
3. Open a terminal in this folder.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\\Scripts\\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the application:

   ```powershell
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in your browser.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Demo login

- **Username:** `admin`
- **Password:** `admin123`

The app creates demo records the first time it starts. Change the default credentials and `SECRET_KEY` before any real deployment. The database file `students.db` is created next to `app.py`.

## CSV import format

CSV import requires `roll_no` and `name`. Optional fields:

```csv
roll_no,name,email,course,semester,attendance,assignment_score,study_hours,previous_score
MCA009,Simran Kaur,simran@example.com,MCA,1,86,79,2.5,72
MCA010,Rahul Mehta,rahul@example.com,MCA,1,68,60,1.5,58
```

Import updates existing students when `roll_no` matches. Marks are entered separately in each student's profile.

## Prediction notes

The displayed score is a **learning-support estimate**, not a guaranteed final result. With scikit-learn installed and enough records, the app fits a Random Forest model on the current class dataset. The target used for training is a synthetic weighted academic indicator derived from the available inputs, so this is a functional demonstration of an ML workflow rather than a validated educational prediction model. For real use, train and evaluate against historical final outcomes, check bias and error across student groups, and keep teachers in the decision loop. Do not use predictions alone for high-stakes decisions.

## Project structure

```text
student_performance_system/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── students.html
│   ├── student_form.html
│   ├── student_detail.html
│   ├── analytics.html
│   ├── prediction.html
│   ├── import.html
│   └── error.html
└── static/
    └── style.css
```

## Common troubleshooting

- If `ModuleNotFoundError: No module named 'flask'` appears, activate the virtual environment and run `pip install -r requirements.txt`.
- If port 5000 is busy, stop the other process or set a different port in the `app.run()` line.
- Chart.js and Google Fonts use CDNs, so charts/fonts need an internet connection; the core Flask app and database run locally.
- This is a development server. Use a production WSGI server and set a strong secret key before deploying publicly.
