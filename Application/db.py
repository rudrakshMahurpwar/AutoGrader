# db.py

import sqlite3
import py_compile


from datetime import datetime

DB_PATH = "grader.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Students ---
def add_student(roll_number, name):
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO Students (roll_number, name) VALUES (?, ?)",
            (roll_number, name),
        )


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
    """
    Add a submission if not already exists for (student_id, question_id).
    Returns submission_id if inserted, None if duplicate exists.
    """
    conn = get_connection()
    with conn:
        # Check if already submitted
        existing = conn.execute(
            "SELECT id FROM Submissions WHERE student_id = ? AND question_id = ?",
            (student_id, question_id),
        ).fetchone()

        if existing:
            return None

        cursor = conn.execute(
            "INSERT INTO Submissions (student_id, question_id, answer_text, submitted_at) VALUES (?, ?, ?, ?)",
            (student_id, question_id, answer_text, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def get_student_submissions(student_id):
    conn = get_connection()
    return conn.execute(
        """
        SELECT s.id as submission_id, s.answer_text, s.submitted_at,
               q.id as question_id, q.question, q.reference_answer,
               sc.similarity_score, sc.rubric_json, sc.final_score,
               sc.feedback, sc.teacher_score
        FROM Submissions s
        JOIN Questions q ON s.question_id = q.id
        LEFT JOIN Scores sc ON sc.submission_id = s.id
        WHERE s.student_id = ?
        ORDER BY s.submitted_at DESC
    """,
        (student_id,),
    ).fetchall()


def get_student_submission(student_id, question_id):
    """Fetch a specific submission (used for pre-filling/updating)."""
    conn = get_connection()
    return conn.execute(
        """
        SELECT s.id as submission_id, s.answer_text, s.submitted_at,
               q.id as question_id, q.question, q.reference_answer,
               sc.similarity_score, sc.rubric_json, sc.final_score,
               sc.feedback, sc.teacher_score
        FROM Submissions s
        JOIN Questions q ON s.question_id = q.id
        LEFT JOIN Scores sc ON sc.submission_id = s.id
        WHERE s.student_id = ? AND s.question_id = ?
    """,
        (student_id, question_id),
    ).fetchone()


def update_submission(submission_id, new_answer):
    """Update a student's answer and reset timestamp (requires re-grading later)."""
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE Submissions SET answer_text = ?, submitted_at = ? WHERE id = ?",
            (new_answer, datetime.now().isoformat(), submission_id),
        )
        # Delete old score so grading can be re-run
        conn.execute("DELETE FROM Scores WHERE submission_id = ?", (submission_id,))


def add_score(submission_id, similarity_score, rubric_json, final_score, feedback):
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO Scores (submission_id, similarity_score, rubric_json, final_score, feedback)
               VALUES (?, ?, ?, ?, ?)""",
            (submission_id, similarity_score, rubric_json, final_score, feedback),
        )


def update_teacher_score(submission_id, teacher_score):
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE Scores SET teacher_score = ? WHERE submission_id = ?",
            (teacher_score, submission_id),
        )


def get_all_results():
    conn = get_connection()
    return conn.execute(
        """
        SELECT st.roll_number, st.name, q.question,
               s.answer_text, s.submitted_at,
               sc.similarity_score, sc.final_score, sc.teacher_score,
               sc.rubric_json, sc.feedback
        FROM Submissions s
        JOIN Students st ON s.student_id = st.id
        JOIN Questions q ON s.question_id = q.id
        LEFT JOIN Scores sc ON sc.submission_id = s.id
        ORDER BY st.roll_number, s.submitted_at
        """
    ).fetchall()
