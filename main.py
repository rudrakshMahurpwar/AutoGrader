from utils import grade_long_answers


def get_ques_input():
    q_id: int = int(input("Enter Question id number: "))
    question: str = input("Enter Question: ")
    ref_answer: str = input("Enter Reference Answer: ")

    return {"q_id": q_id, "question": question, "ref_answer": ref_answer}


def get_stud_data():
    stud_id: int = int(input("Enter Student id: "))
    stud_name: str = input("Enter Student Name: ")

    return {"stud_id": stud_id, "stud_name": stud_name}


def get_stud_answer(stud_id):
    stud_answer: str = input("Enter Student Answer: ")

    return stud_answer


if __name__ == "__main__":

    enter_question: bool = True
    questions_list = []
    while enter_question:
        ques_data = get_ques_input()
        questions_list.append(ques_data)
        response: str = input("Enter one more Question? (yes/no): ").strip().lower()
        enter_question = response in ["yes", "y", "yeah", "sure"]
        print()
    print(questions_list)

    enter_student: bool = True
    students_list = []
    while enter_student:
        stud_data = get_stud_data()
        students_list.append(stud_data)
        response: str = input("Enter one more Student? (yes/no): ").strip().lower()
        enter_student = response in ["yes", "y", "yeah", "sure"]
        print()
    print(students_list)

    student_answers = []
    for s in students_list:
        print(s)
        print(f"Student Id: {s['stud_id']}, Student name: {s['stud_name']}")
        for q in questions_list:
            print(f"Question {q['q_id']}: {q['question']}")
            stud_answer = get_stud_answer(s["stud_id"])
            print(stud_answer)
    grade_long_answers(
        reference_answers=questions_list, student_answers=student_answers
    )
