# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import nltk
import re

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
clf = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed


import PyPDF2

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(uploaded_file):
    try:
        return uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        return uploaded_file.read().decode('latin-1')


# Function to clean resume text
def clean_resume(resume_text):
    cleanText = re.sub('http\S+\s', ' ', resume_text)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText






# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        if uploaded_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_txt(uploaded_file)


        cleaned_resume= clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        input_features1 = input_features.toarray()
        prediction_id = clf.predict(input_features1)[0]
        st.write(prediction_id)
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }


        category_name = category_mapping.get(prediction_id, "Unknown")


        st.write("Predicted Category:", category_name)





# python main
if __name__ == "__main__":
    main()


