import re
import pickle
import pandas as pd
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
import docx2txt

#  CONFIG 
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="",
    layout="wide"
)

st.title(" Resume Classification App")
st.caption("Upload resumes and classify them using ML")

#  SKILLS LIST 
skills_list = [
    # React Developer
    "react", "javascript", "html", "css", "redux", "typescript", "nodejs", "git",
    # SQL Developer
    "sql", "pl/sql", "oracle", "mysql", "postgresql", "etl", "ssrs", "ssis",
    # Workday
    "workday hcm", "workday integrations", "workday payroll", "reporting", "workday studio",
    # Peoplesoft
    "peoplesoft hcm", "peoplesoft financials", "peoplesoft payroll", "peoplesoft apps", "ps query",
    # Common skills
    "python", "java", "excel", "power bi", "tableau", "aws", "machine learning"
]

#  LOAD MODEL 
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("resume_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

#  TEXT EXTRACTION 
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.name.endswith(".docx"):
        return docx2txt.process(file)

    return ""

#  PREPROCESS 
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#  EXPERIENCE EXTRACTION 
def extract_experience(text):
    matches = re.findall(r'(\d+)\s*(years|year|yrs|yr)', text.lower())
    years = [int(m[0]) for m in matches]
    return max(years) if years else 0

#  SKILLS EXTRACTION 
def extract_skills(text):
    text_lower = text.lower()
    detected = [skill for skill in skills_list if skill.lower() in text_lower]
    return detected if detected else ["No skills detected"]

#  SIDEBAR 
st.sidebar.header(" Upload Resumes")
files = st.sidebar.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

#  PREDICTION 
results = []

if files:
    for file in files:
        raw_text = extract_text(file)
        clean_text = preprocess(raw_text)

        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]

        results.append({
            "Filename": file.name,
            "Predicted Profile": prediction,
            "Experience (Years)": extract_experience(raw_text),
            "Skills": ", ".join(extract_skills(raw_text))
        })

#  DISPLAY 
if results:
    st.subheader(" Classification Results")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Export to Excel
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label=" Download Results (Excel)",
        data=output,
        file_name="resume_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info(" Upload resumes from the sidebar to start classification")
