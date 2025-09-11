CREATE TABLE IF NOT EXISTS Students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number TEXT UNIQUE NOT NULL,  -- New: Roll number is unique
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    reference_answer TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    answer_text TEXT NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- New: for tracking edits
    UNIQUE(student_id, question_id), -- New: ensures one answer per student per question
    FOREIGN KEY(student_id) REFERENCES Students(id),
    FOREIGN KEY(question_id) REFERENCES Questions(id)
);

CREATE TABLE IF NOT EXISTS Scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL UNIQUE, -- One score per submission
    similarity_score REAL,
    rubric_json TEXT,       -- JSON response from Mistral rubrics
    final_score REAL,       -- AI suggested score
    teacher_score REAL,     -- New: Editable by teacher
    feedback TEXT,          -- New: Store generated feedback
    FOREIGN KEY(submission_id) REFERENCES Submissions(id)
);
