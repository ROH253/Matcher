from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import PyPDF2
import docx2txt
import re
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.secret_key = 'your_secret_key_here'

# Ensure the uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample skills
SKILL_KEYWORDS = [
    "python", "java", "flask", "django", "react", "javascript",
    "html", "css", "sql", "aws", "git", "docker", "kubernetes",
    "machine learning", "data analysis", "leadership", "communication",
    "project management"
]

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    """Extract text from PDF or DOCX files"""
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
    elif filepath.endswith('.docx'):
        text = docx2txt.process(filepath)
    else:
        text = ''
    return text

def preprocess_text(text):
    """Preprocess the text: remove special characters, convert to lowercase"""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

def extract_skills(text):
    """Extract skills from the resume based on keyword matching"""
    skills = []
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text:
            skills.append(skill)
    return list(set(skills))

def generate_improvement_suggestions(missing_skills, match_score):
    """Generate improvement suggestions based on missing skills and match score"""
    suggestions = []
    
    if match_score < 70:
        suggestions.append("<strong>Match Score is Low:</strong> Improve the resume content to better align with the job description.")
    
    if missing_skills:
        suggestions.append(f"<strong>Missing Skills:</strong> Add skills like {', '.join(missing_skills)} to enhance your match potential.")

    if not suggestions:
        suggestions.append("<strong>Great Match!</strong> The resume is well-aligned with the job description.")

    return "<br>".join(suggestions)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        files = request.files.getlist('resumes')

        if not job_description or not files:
            flash('Please provide a job description and upload at least one resume.', 'error')
            return redirect(url_for('index'))

        resume_filenames = []

        # Save resumes
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                resume_filenames.append(filename)

        resume_data = []

        for filename in resume_filenames:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Extract and preprocess text
            resume_text = extract_text_from_file(filepath)
            clean_resume = preprocess_text(resume_text)
            clean_jd = preprocess_text(job_description)

            # Vectorize and calculate similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([clean_jd, clean_resume])
            match_score = round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)

            # Extract skills and identify missing skills
            resume_skills = extract_skills(clean_resume)
            jd_skills = extract_skills(clean_jd)
            missing_skills = list(set(jd_skills) - set(resume_skills))

            # Generate improvement suggestions
            suggestions = generate_improvement_suggestions(missing_skills, match_score)

            resume_data.append({
                'filename': filename,
                'match_score': match_score,
                'skills': resume_skills,
                'suggestions': suggestions
            })

        # ✅ Sort resumes by match score (highest first)
        resume_data.sort(key=lambda x: x['match_score'], reverse=True)

        # Store resume data in session for displaying in results
        session['resume_data'] = resume_data
        session['job_description'] = job_description

        return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    """Display the results page with resume matches"""
    resume_data = session.get('resume_data', [])
    job_description = session.get('job_description', '')

    return render_template('results.html', resume_data=resume_data, job_description=job_description)

@app.route('/resume/<filename>')
def view_resume(filename):
    """Display resume details with skills, match score, and suggestions"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        flash('Resume not found.', 'error')
        return redirect(url_for('index'))

    # Extract and preprocess text
    resume_text = extract_text_from_file(filepath)
    clean_resume = preprocess_text(resume_text)

    # Extract skills
    skills = extract_skills(clean_resume)

    # Display the first 100 words as a brief summary
    brief_summary = ' '.join(clean_resume.split()[:100])

    # Display match score (dummy for individual view)
    match_score = 75

    # Generate suggestions
    jd_skills = ["python", "flask", "sql"]  # Example job skills for testing
    missing_skills = list(set(jd_skills) - set(skills))
    suggestions = generate_improvement_suggestions(missing_skills, match_score)

    return render_template('view_resume.html', 
                           filename=filename, 
                           resume_text=brief_summary, 
                           skills=skills, 
                           match_score=match_score,
                           suggestions=suggestions)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
