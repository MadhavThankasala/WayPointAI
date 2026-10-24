import streamlit as st

from resume_advisor import (
    DEFAULT_MODEL_ID,
    MAX_RESUME_CHARACTERS,
    ModelConfigurationError,
    ModelResponseError,
    ResumeInputError,
    analyze_resume,
    extract_resume_text,
)

st.set_page_config(page_title="WayPointAI")
st.title("WayPointAI")
st.write("Get practical career advice based on your resume and target roles!")
st.info(
    "The model is downloaded from Hugging Face on first use and runs locally. "
)

with st.form("resume_advice_form"):
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        help="Supported formats: PDF and UTF-8 text files.",
    )

    roles = st.multiselect(
        "What tech jobs are you interested in?",
        ["Software Engineer", "AI/ML Engineer", "Data Scientist",
         "Security Engineer", "Computer Engineer", "IT Support Specialist",
         "Hardware Engineer", "QA Engineer", "Product Manager"]
    )
    submitted = st.form_submit_button("Generate Roadmap")

if submitted:
    if uploaded_file is None:
        st.error("Please upload a PDF or text file.")
    elif not roles:
        st.error("Please enter the jobs you are interested in.")
    elif uploaded_file.size > 10 * 1024 * 1024:
        st.error("The uploaded file must be 10 MB or smaller.")
    else:
        try:
            resume_text = extract_resume_text(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            if len(resume_text) > MAX_RESUME_CHARACTERS:
                st.warning(
                    "The resume is long; only its first "
                    f"{MAX_RESUME_CHARACTERS:,} characters will be analyzed."
                )

            with st.spinner("Loading the local model and analyzing your resume..."):
                advice = analyze_resume(
                    resume_text=resume_text,
                    roles=roles,
                    model_id=DEFAULT_MODEL_ID,
                )
            st.subheader("Career advice")
            st.markdown(advice)
        except ResumeInputError as error:
            st.error(str(error))
        except ModelConfigurationError as error:
            st.error(str(error))
        except ModelResponseError as error:
            st.error(str(error))
