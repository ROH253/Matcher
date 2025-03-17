import os
import openai
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
import torch
from transformers import pipeline
from flask import Flask, request, render_template, redirect, url_for, flash, session
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get OpenAI API key from environment variable or use a default for development
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")

# Initialize the Flask app
app = Flask(__name__)

# Set secret key for session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Hugging Face NER pipeline
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(docx_path):
    try:
        return docx2txt.process(docx_path)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    return ""

def extract_skills(resume_text):
    """
    Extract skills from resume text using Hugging Face's NER model
    """
    # Apply the named entity recognition (NER) model
    entities = nlp(resume_text)
    
    # Filter out entities that are likely to be skills (e.g., tools, programming languages)
    skills = [entity['word'] for entity in entities if entity['entity'] == 'B-ORG' or entity['entity'] == 'B-MISC']
    
    return sorted(list(set(skills)))

def score_resume(resume_text, job_description):
    """
    Score resume against job description using OpenAI
    """
    scoring_prompt = f"""
    Rate how well the candidate's resume matches the job description on a scale of 0-100.
    Evaluate based on:
    1. Skills match (50%)
    2. Experience relevance (30%)
    3. Education fit (20%)
    
    Resume:
    {resume_text[:4000]}
    
    Job Description:
    {job_description[:2000]}
    
    Return ONLY a numeric score between 0-100, nothing else.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using GPT-4 for better analysis
            messages=[ 
                {"role": "system", "content": "You are a recruitment AI that analyzes resume fit."},
                {"role": "user", "content": scoring_prompt}
            ]
        )
        
        score_text = response["choices"][0]["message"]["content"].strip()
        # Extract just the number from the response
        import re
        score_match = re.search(r'\d+', score_text)
        if score_match:
            return float(score_match.group())
        return float(score_text)
    except Exception as e:
        print(f"Error scoring resume with AI: {e}")
        # Fallback to basic matching if AI fails
        common_words = set(resume_text.lower().split()) & set(job_description.lower().split())
        return min(100, len(common_words) * 2)

def get_improvement_suggestions(resume_text, job_description):
    """
    Get AI-powered suggestions for improving the resume
    """
    suggestion_prompt = f"""
    Analyze this resume against the job description and provide 3-5 specific suggestions 
    for improving the resume to better match the job requirements.
    
    Resume:
    {resume_text[:4000]}
    
    Job Description:
    {job_description[:2000]}
    
    Provide actionable suggestions in bullet point format.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[ 
                {"role": "system", "content": "You are a career coach specializing in resume improvement."},
                {"role": "user", "content": suggestion_prompt}
            ]
        )
        
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error getting suggestions with AI: {e}")
        return "Unable to generate suggestions at this time."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'job_description' not in request.form or not request.form['job_description'].strip():
        flash('No job description provided!', 'error')
        return redirect(request.url)
    
    job_description = request.form['job_description']
    session['job_description'] = job_description
    
    if 'resumes' not in request.files:
        flash('No resume files selected!', 'error')
        return redirect(request.url)
    
    files = request.files.getlist('resumes')
    
    if len(files) == 1 and files[0].filename == '':
        flash('No selected files!', 'error')
        return redirect(request.url)
    
    resume_data = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text
            resume_text = extract_text(file_path)
            if not resume_text:
                flash(f'Could not extract text from {filename}', 'warning')
                continue
            
            # Extract skills
            skills = extract_skills(resume_text)
            
            # Score resume against job description
            match_score = score_resume(resume_text, job_description)
            
            # Get improvement suggestions
            suggestions = get_improvement_suggestions(resume_text, job_description)
            
            resume_data.append({
                'filename': filename,
                'skills': skills,
                'match_score': match_score,
                'suggestions': suggestions
            })
    print("Resume Data:", resume_data)        
    
    if not resume_data:
        flash('No valid resumes were processed!', 'error')
        return redirect(request.url)
    
    # Sort by match score
    resume_data.sort(key=lambda x: x['match_score'], reverse=True)
    session['resume_data'] = resume_data
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'resume_data' not in session or 'job_description' not in session:
        flash('No data to display. Please upload resumes first.', 'error')
        return redirect(url_for('index'))
    
    resume_data = session['resume_data']
    job_description = session['job_description']
    
    # Define the threshold score (e.g., 50)
    threshold_score = 1
    
    # Filter resumes based on the match score
    filtered_resume_data = [resume for resume in resume_data if resume['match_score'] > threshold_score]
    
    # If no resumes passed the threshold, show a message
    if not filtered_resume_data:
        flash(f'No candidates scored above {threshold_score}.', 'info')
    filtered_resume_data.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Limit to top 4 resumes only
    top_resumes = filtered_resume_data[:4]
    
    return render_template('results.html', 
                          resume_data=top_resumes,
                          job_description=job_description)

@app.route('/view_resume/<filename>')
def view_resume(filename):
    if 'job_description' not in session:
        flash('Job description not found!', 'error')
        return redirect(url_for('index'))
        
    job_description = session['job_description']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash('File not found!', 'error')
        return redirect(url_for('results'))
    
    resume_text = extract_text(file_path)
    skills = extract_skills(resume_text)
    suggestions = get_improvement_suggestions(resume_text, job_description)
    
    return render_template('view_resume.html',
                          filename=filename,
                          resume_text=resume_text,
                          skills=skills,
                          suggestions=suggestions)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT environment variable is not set
    app.run(debug=True, host='0.0.0.0', port=port)
