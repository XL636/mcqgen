import os 
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from scr.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from scr.mcqgenerator.MCQGenerator import generate_evaluation_chain
from scr.mcqgenerator.logger import logging

#load file, the file will be Dict type
with open(r'C:\Users\LEX\mcqgen\Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for websit/app
st.title("MCQs Creator Application with LangChain :)")

#Create a form using st.form
with st.form("user_inputs"):
    #File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    
    #Input Fields
    mcq_count = st.number_input("NO. of MCQs", min_value=3, max_value=50)

    #Subject
    subject = st.text_input("insert Subject", max_chars=20, placeholder='Math')

    # Quiz Tone
    tone = st.text_input("Complexity level of Question", max_chars=20, placeholder='Simple')

    # Add Button
    button = st.form_submit_button("Create MCQs")


# Check if the button is clicked and all files have input
if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("loading...."):
        try:
            text = read_file(uploaded_file)
            #count tokens and the cost of API call
            with get_openai_callback() as cb:
                response = generate_evaluation_chain(
                    {
                        'text':text,
                        'number':mcq_count,
                        'subject':subject,
                        'tone': tone,
                        'response_json':json.dumps(RESPONSE_JSON)
                    }
                )
            
        except Exception as e:
            traceback.print_exception(type(e),e,e.__traceback__)
            st.error("Error")

        else:
            print(f"Total Tokens:{cb.total_tokens}")
            print(f"prompt Tokens:{cb.prompt_tokens}")
            print(f"COmpletino Tokens:{cb.completion_tokens}")
            print(f"Total Cost:{cb.total_cost}")
            if isinstance(response, dict):
                # Extract the quiz data from the response
                quiz = response.get("quiz", None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index+1
                        st.table(df)
                        st.text_area(label="Review",value=response["review"])
                    else:
                        st.error("Error in the table data")
            else:
                st.write(response)