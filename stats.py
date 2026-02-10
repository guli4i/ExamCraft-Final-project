import sqlite3
import matplotlib.pyplot as plt

DB_PATH = "exam.db"

# подключение к базе
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# запрос для анализа
cur.execute("""
SELECT exam_type, COUNT(*) 
FROM exams
GROUP BY exam_type
""")

data = cur.fetchall()
conn.close()

# если данных нет
if not data:
    print("No data available for visualization.")
    exit()

# подготовка данных
labels = [row[0].upper() for row in data]
values = [row[1] for row in data]

# построение графика
plt.bar(labels, values)
plt.title("Distribution of Generated Exam Types")
plt.xlabel("Exam Type")
plt.ylabel("Number of Exams")
plt.tight_layout()
plt.show()
