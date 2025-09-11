# app.py

import streamlit as st
from db import *
from grading import (
    compute_chunkwise_similarity,
    get_sentence_similarity_details,
    get_mistral_feedback_and_rubric,
    combine_scores,
)
import json, io, csv

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
        question_selection = st.selectbox("Select Question", question_texts)
        q_idx = question_texts.index(question_selection)
        question = questions[q_idx]

        # 🔍 Check if student already submitted this question
        existing_submission = get_student_submission(student["id"], question["id"])

        # Pre-fill answer if exists
        existing_answer = (
            existing_submission["answer_text"] if existing_submission else ""
        )

        with st.form("submit_answer"):
            student_answer = st.text_area("Student's Answer", value=existing_answer)
            submit = st.form_submit_button("Save Answer")

            if submit:
                try:
                    if existing_submission:
                        # 🔄 Update existing submission
                        update_submission(
                            existing_submission["submission_id"], student_answer
                        )
                        submission_id = existing_submission["submission_id"]
                        st.info("✏️ Answer updated successfully!")
                    else:
                        # 🆕 Create new submission
                        submission_id = add_submission(
                            student["id"], question["id"], student_answer
                        )
                        st.success("✅ New answer submitted!")

                    # ✅ Always grade (for both update & new)
                    similarity_score = compute_chunkwise_similarity(
                        question["reference_answer"], student_answer
                    )

                    llm_result = get_mistral_feedback_and_rubric(
                        question["question"],
                        question["reference_answer"],
                        student_answer,
                    )

                    if "error" in llm_result:
                        st.error(f"❌ LLM error: {llm_result['error']}")
                    else:
                        rubric = llm_result["rubric"]
                        feedback = llm_result["feedback"]
                        final_score = combine_scores(similarity_score, rubric)

                        # Save score (overwrite if exists)
                        add_score(
                            submission_id,
                            similarity_score,
                            json.dumps(rubric),
                            final_score,
                            feedback,
                        )

                        st.success("✅ Answer graded successfully!")
                        st.markdown(f"**Similarity Score:** `{similarity_score}`")
                        st.markdown(f"**Final Score:** `{final_score}`")

                        st.subheader("📊 Rubric")
                        for crit, score in rubric.items():
                            st.markdown(f"- **{crit}**: {score}/5")

                        st.subheader("🧠 Feedback")
                        st.markdown(feedback)

                        # 🔍 Highlight most relevant sentences in student's answer
                        sentence_scores = get_sentence_similarity_details(
                            question["reference_answer"], student_answer
                        )

                        highlighted_answer = ""
                        for sent, score in sentence_scores:
                            if score > 0.9:  # threshold
                                highlighted_answer += f"**`{sent}`** "
                            else:
                                highlighted_answer += f"{sent} "

                        st.subheader("Student Answer with Highlights")
                        st.markdown(highlighted_answer, unsafe_allow_html=True)

                except Exception as e:
                    # 👨‍🏫 Friendly duplicate error message
                    if "UNIQUE constraint" in str(e):
                        st.warning(
                            "⚠️ You have already submitted an answer for this question. Please edit it instead."
                        )
                    else:
                        st.error(f"❌ Unexpected error: {e}")


# 4️⃣ View Results
elif page == "📊 View Results":
    st.header("📊 View Student Results")

    # 📥 Download CSV button (all students at once)
    results = get_all_results()
    if results:

        output = io.StringIO()
        writer = csv.writer(output)

        # CSV header
        writer.writerow(
            [
                "Roll Number",
                "Name",
                "Question",
                "Answer",
                "High-Similarity Sentences",
                "Similarity Score",
                "Final Score",
                "Teacher Score",
                "Factual Accuracy",
                "Completeness",
                "Clarity",
                "Relevance",
                "Feedback",
                "Submitted At",
            ]
        )

        for r in results:
            rubric = json.loads(r["rubric_json"]) if r["rubric_json"] else {}
            high_sim_sents = ""
            if (
                r["answer_text"] and r["question"]
            ):  # ensure answer + reference available
                sentence_scores = get_sentence_similarity_details(
                    r["question"], r["answer_text"]
                )
                high_sim_sents = "; ".join(
                    [sent for sent, score in sentence_scores if score > 0.9]
                )

            writer.writerow(
                [
                    r["roll_number"],
                    r["name"],
                    r["question"],
                    r["answer_text"],
                    high_sim_sents,
                    r["similarity_score"],
                    r["final_score"],
                    r["teacher_score"],
                    rubric.get("Factual Accuracy", ""),
                    rubric.get("Completeness", ""),
                    rubric.get("Clarity", ""),
                    rubric.get("Relevance", ""),
                    r["feedback"],
                    r["submitted_at"],
                ]
            )

        st.download_button(
            label="📥 Download All Results (CSV)",
            data=output.getvalue(),
            file_name="grading_results.csv",
            mime="text/csv",
        )

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

        # 📥 Download Individual Marksheet button
        student_submissions = get_student_submissions(student["id"])
        if student_submissions:
            output = io.StringIO()
            writer = csv.writer(output)

            # CSV header
            writer.writerow(
                [
                    "Roll Number",
                    "Name",
                    "Question",
                    "Answer",
                    "High-Similarity Sentences",
                    "Similarity Score",
                    "Final Score",
                    "Teacher Score",
                    "Factual Accuracy",
                    "Completeness",
                    "Clarity",
                    "Relevance",
                    "Feedback",
                    "Submitted At",
                ]
            )

            for sub in student_submissions:
                rubric = json.loads(sub["rubric_json"]) if sub["rubric_json"] else {}
                high_sim_sents = ""
                if sub["answer_text"] and sub["question"]:
                    sentence_scores = get_sentence_similarity_details(
                        sub["reference_answer"], sub["answer_text"]
                    )
                    high_sim_sents = "; ".join(
                        [sent for sent, score in sentence_scores if score > 0.9]
                    )

                writer.writerow(
                    [
                        student["roll_number"],
                        student["name"],
                        sub["question"],
                        sub["answer_text"],
                        high_sim_sents,
                        sub["similarity_score"],
                        sub["final_score"],
                        sub["teacher_score"],
                        rubric.get("Factual Accuracy", ""),
                        rubric.get("Completeness", ""),
                        rubric.get("Clarity", ""),
                        rubric.get("Relevance", ""),
                        sub["feedback"],
                        sub["submitted_at"],
                    ]
                )

            st.download_button(
                label=f"📥 Download {student['name']}'s Marksheet (CSV)",
                data=output.getvalue(),
                file_name=f"{student['roll_number']}_{student['name']}_marksheet.csv",
                mime="text/csv",
            )

        submissions = get_student_submissions(student["id"])
        if not submissions:
            st.info("ℹ️ No submissions found for this student.")
        else:
            for sub in submissions:
                st.subheader(f"📝 {sub['question']}")
                st.markdown(f"**Submitted on:** {sub['submitted_at']}")
                st.subheader("📌 Student Answer")
                sentence_scores = get_sentence_similarity_details(
                    sub["reference_answer"], sub["answer_text"]
                )

                highlighted_answer = ""
                for sent, score in sentence_scores:
                    if score > 0.9:
                        highlighted_answer += f"**`{sent}`** "
                    else:
                        highlighted_answer += f"{sent} "

                st.markdown(highlighted_answer, unsafe_allow_html=True)

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
