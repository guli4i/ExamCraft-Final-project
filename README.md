![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![Database](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)
![ML](https://img.shields.io/badge/Machine%20Learning-FLAN%20%2B%20GPT-green)
![Visualization](https://img.shields.io/badge/Visualization-Matplotlib-orange)

# ExamCraft

ExamCraft is an AI-powered web platform for automatic exam generation.

The system integrates Machine Learning, database storage, authentication, file export, and data visualization into a single full-stack application.

---

# Problem Statement

Manual exam creation is time-consuming and inconsistent.  
Existing tools often lack structured storage, analytics, and integration with AI.

ExamCraft solves this by:

• Automatically generating Open and MCQ questions  
• Storing exams per user  
• Providing export functionality (PDF & DOCX)  
• Visualizing generation statistics  

---

# Core Features

• User registration & authentication  
• AI-based Open question generation (FLAN-T5)  
• AI-based MCQ generation (GPT model)  
• Secure SQLite database storage  
• PDF & DOCX export  
• Data visualization using Matplotlib  
• Exam history per user  

---

# Technology Stack

Frontend:
- HTML
- CSS
- Bootstrap
- JavaScript

Backend:
- Python
- Flask

Machine Learning:
- FLAN-T5 (Open Questions)
- GPT API (MCQ Questions)

Database:
- SQLite

Data Analysis:
- NumPy
- Matplotlib

---

# Machine Learning Integration

Open Questions:
- Generated using FLAN-T5 (Seq2Seq Transformer model)
- Controlled sampling with temperature and top-p

MCQ Questions:
- Generated via GPT API
- Structured JSON output
- Answer shuffling for randomness

---

# Database Structure

## users table
- id
- username
- password_hash

## exams table
- id
- user_id (Foreign Key)
- exam_type
- questions (JSON)
- created_at

Each exam is linked to a specific user.

---

# API Endpoints

POST /generate_open  
POST /generate_mcq  
POST /upload_file  
POST /download_pdf  
POST /download_docx  

Authentication routes:
- /login
- /register
- /logout

---

# Data Visualization

The system generates a dynamic bar chart showing:

- Distribution of Open vs MCQ exams
- Statistical analysis of usage

Visualization built using:
- NumPy
- Matplotlib

---

# How to Run Locally

## 1. Clone repository

```bash
git clone <your_repo_link>
cd examcraft
2. Create virtual environment

Windows:

python -m venv venv
venv\Scripts\activate


Mac/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set environment variable (IMPORTANT)

Windows:

set OPENAI_API_KEY=your_api_key


Mac/Linux:

export OPENAI_API_KEY=your_api_key

5. Run server
python app.py


Open in browser:

http://127.0.0.1:5000

Deployment Notes

This project is configured for local deployment.

For production:

PostgreSQL

Gunicorn

Nginx

Environment variables

DEBUG=False

Proper security configuration

Project Structure
examcraft/
│
├── app.py
├── requirements.txt
├── exam.db
├── templates/
│   ├── intro.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── history.html
│   ├── exam.html
│   └── about.html

Limitations

• Uses SQLite (not production-grade)
• No asynchronous processing
• Limited ML fine-tuning
• Basic security level

Authors:

Gulnara Mukhametkarimova
Sultanseiit Zhurtbay
Final Project — 2026
