from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_student_data():
    return {
        "name": "Rahul",
        "course": "Generative AI",
        "attendance": "82%",
        "assignment_status": "Pending"
    }

student_data = get_student_data()

question = "Create a message to remind the student about assignment submission."

context = f"""
Student Data:
Name: {student_data["name"]}
Course: {student_data["course"]}
Attendance: {student_data["attendance"]}
Assignment Status: {student_data["assignment_status"]}

Task:
{question}
"""

response = client.responses.create(
    model="gpt-5.5",
    input=context
)

print(response.output_text)

