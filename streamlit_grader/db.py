# db.py

import sqlite3

DB_PATH = "grader.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Students ---
def add_student(name):
    conn = get_connection()
    with conn:
        conn.execute("INSERT OR IGNORE INTO Students (name) VALUES (?)", (name,))


def get_students():
    conn = get_connection()
    return conn.execute("SELECT * FROM Students").fetchall()


# --- Questions ---
def add_question(question, reference_answer):
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO Questions (question, reference_answer) VALUES (?, ?)",
            (question, reference_answer),
        )


def get_questions():
    conn = get_connection()
    return conn.execute("SELECT * FROM Questions").fetchall()


def get_question_by_id(q_id):
    conn = get_connection()
    return conn.execute("SELECT * FROM Questions WHERE id = ?", (q_id,)).fetchone()


# --- Submissions + Scores ---
def add_submission(student_id, question_id, answer_text):
    conn = get_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO Submissions (student_id, question_id, answer_text) VALUES (?, ?, ?)",
            (student_id, question_id, answer_text),
        )
        return cursor.lastrowid  # Submission ID


def add_score(submission_id, similarity_score, rubric_json, final_score):
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO Scores (submission_id, similarity_score, rubric_json, final_score) VALUES (?, ?, ?, ?)",
            (submission_id, similarity_score, rubric_json, final_score),
        )


def get_student_submissions(student_id):
    conn = get_connection()
    return conn.execute(
        """
        SELECT s.id as submission_id, s.answer_text, s.submitted_at,
               q.question, q.reference_answer,
               sc.similarity_score, sc.rubric_json, sc.final_score
        FROM Submissions s
        JOIN Questions q ON s.question_id = q.id
        LEFT JOIN Scores sc ON sc.submission_id = s.id
        WHERE s.student_id = ?
        ORDER BY s.submitted_at DESC
    """,
        (student_id,),
    ).fetchall()
