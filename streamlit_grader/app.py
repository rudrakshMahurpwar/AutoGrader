# app.py

import streamlit as st
from db import *
from grading import (
    compute_chunkwise_similarity,
    get_mistral_feedback_and_rubric,
    combine_scores,
)
import json

st.set_page_config(page_title="Auto Grader", layout="wide")

st.title("📘 AI-Powered Answer Grader")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["➕ Add Questions", "👤 Add Students", "📝 Submit Answers", "📊 View Results"],
)

# 1️⃣ Add Questions
if page == "➕ Add Questions":
    st.header("➕ Add a New Question")
    with st.form("add_question"):
        question = st.text_area("Question")
        reference_answer = st.text_area("Reference Answer (Ideal Answer)")
        submit = st.form_submit_button("Add Question")
        if submit:
            if question and reference_answer:
                add_question(question, reference_answer)
                st.success("✅ Question added successfully!")
            else:
                st.error("❌ Please fill both fields.")

    st.subheader("📚 All Questions")
    questions = get_questions()
    for q in questions:
        st.markdown(f"**Q{q['id']}**: {q['question']}")
        with st.expander("Reference Answer"):
            st.markdown(q["reference_answer"])

# 2️⃣ Add Students
elif page == "👤 Add Students":
    st.header("👤 Add New Student")
    with st.form("add_student"):
        roll_number = st.text_input("Roll Number")
        name = st.text_input("Student Name")
        submit = st.form_submit_button("Add Student")
        if submit:
            if roll_number and name:
                add_student(roll_number, name)
                st.success(f"✅ Student '{name}' (Roll: {roll_number}) added.")
            else:
                st.error("❌ Please enter both roll number and name.")

    st.subheader("📋 All Students")
    students = get_students()
    for s in students:
        st.markdown(f"- **{s['roll_number']}** – {s['name']}")

# 3️⃣ Submit Answers
elif page == "📝 Submit Answers":
    st.header("📝 Submit Student Answer")

    students = get_students()
    questions = get_questions()

    if not students or not questions:
        st.warning("⚠️ Add at least one student and one question first.")
    else:
        student_options = [f"{s['roll_number']} – {s['name']}" for s in students]
        student_selection = st.selectbox("Select Student", student_options)
        student = students[student_options.index(student_selection)]

        question_texts = [f"Q{q['id']}: {q['question']}" for q in questions]

        with st.form("submit_answer"):
            question_selection = st.selectbox("Select Question", question_texts)
            student_answer = st.text_area("Student's Answer")
            submit = st.form_submit_button("Grade Answer")

            if submit:
                q_idx = question_texts.index(question_selection)
                question = questions[q_idx]

                # Save submission
                submission_id = add_submission(
                    student["id"], question["id"], student_answer
                )

                # Compute similarity score
                similarity_score = compute_chunkwise_similarity(
                    question["reference_answer"], student_answer
                )

                # Get rubric + feedback
                llm_result = get_mistral_feedback_and_rubric(
                    question["question"], question["reference_answer"], student_answer
                )

                if "error" in llm_result:
                    st.error(f"❌ LLM error: {llm_result['error']}")
                else:
                    rubric = llm_result["rubric"]
                    feedback = llm_result["feedback"]
                    final_score = combine_scores(similarity_score, rubric)

                    # Save score
                    add_score(
                        submission_id,
                        similarity_score,
                        json.dumps(rubric),
                        final_score,
                        feedback,
                    )

                    # Display result
                    st.success("✅ Answer graded successfully!")
                    st.markdown(f"**Similarity Score:** `{similarity_score}`")
                    st.markdown(f"**Final Score:** `{final_score}`")

                    st.subheader("📊 Rubric")
                    for crit, score in rubric.items():
                        st.markdown(f"- **{crit}**: {score}/5")

                    st.subheader("🧠 Feedback")
                    st.markdown(feedback)

# 4️⃣ View Results
elif page == "📊 View Results":
    st.header("📊 View Student Results")

    students = get_students()
    if not students:
        st.warning("⚠️ Add some students first.")
    else:
        student_options = [f"{s['roll_number']} – {s['name']}" for s in students]
        student_selection = st.selectbox("Select Student", student_options)
        student = students[student_options.index(student_selection)]

        # 🔹 Show student info
        st.markdown(f"### 🧑 Student: **{student['name']}**")
        st.markdown(f"**Roll Number:** `{student['roll_number']}`")
        st.markdown("---")

        submissions = get_student_submissions(student["id"])
        if not submissions:
            st.info("ℹ️ No submissions found for this student.")
        else:
            for sub in submissions:
                st.subheader(f"📝 {sub['question']}")
                st.markdown(f"**Submitted on:** {sub['submitted_at']}")
                st.markdown(f"**Answer:** {sub['answer_text']}")

                if sub["similarity_score"] is not None:
                    st.markdown(f"**Similarity Score:** `{sub['similarity_score']}`")
                    st.markdown(f"**Final Score (AI):** `{sub['final_score']}`")

                    # 🔹 Rubric Scores
                    rubric = json.loads(sub["rubric_json"])
                    st.markdown("**Rubric Scores:**")
                    for crit, score in rubric.items():
                        st.markdown(f"- {crit}: {score}/5")

                    # 🔹 Feedback
                    if "feedback" in sub.keys() and sub["feedback"]:
                        st.markdown("**🧠 Feedback:**")
                        st.info(sub["feedback"])

                    # 🔹 Teacher override score
                    teacher_score = st.number_input(
                        "Teacher Override Score",
                        value=sub["final_score"] if sub["final_score"] else 0.0,
                        key=f"teacher_score_{sub['submission_id']}",
                    )
                    if st.button("Save Override", key=f"save_{sub['submission_id']}"):
                        update_teacher_score(sub["submission_id"], teacher_score)
                        st.success("✅ Teacher score saved.")
                else:
                    st.warning("⚠️ Grading pending.")
                st.markdown("---")
