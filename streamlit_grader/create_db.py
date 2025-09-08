# create_db.py

import sqlite3

with open("schema.sql", "r") as f:
    schema = f.read()

conn = sqlite3.connect("grader.db")
cursor = conn.cursor()
cursor.executescript(schema)
conn.commit()
conn.close()

print("✅ Database initialized successfully.")
