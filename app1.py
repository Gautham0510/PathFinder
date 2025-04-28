# --- Built-in Libraries
import math
from pathlib import Path

# --- Third Party Libraries
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
from PIL import Image
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Greens
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
import google.generativeai as genai

# --- Custom Local Modules
import counselor  # Make sure counselor.py exists in your repo
import imagify     # Make sure imagify.py exists too

# --- TensorFlow Model Loader Alias
load_model = tf.keras.models.load_model
## Configure Gemini AI API
genai.configure(api_key="AIzaSyDOE7eUJCOitdcn3wrCsVww5uHlJnBxQbA")
genai_model = genai.GenerativeModel('gemini-1.5-pro')

model = load_model(MODEL_PATH / "botmodel.h5")
tok = joblib.load(MODEL_PATH / "tokenizer_t.pkl")
words = joblib.load(MODEL_PATH / "words.pkl")
df2 = pd.read_csv('bot.csv')
flag=1

import string
import re
import json
import nltk
#run on the first time alone :
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


# THEN your main function starts 

def main():
    global flag
    lem = WordNetLemmatizer()
    n=1
    def tokenizer(x):
        tokens = x.split()
        rep = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [rep.sub('', i) for i in tokens]
        tokens = [i for i in tokens if i.isalpha()]
        tokens = [lem.lemmatize(i.lower()) for i in tokens]
        tokens = [i.lower() for i in tokens if len(i) > 1]
        return tokens

    def no_stop_inp(tokenizer,df,c):
        no_stop = []
        x = df[c][0]
        tokens = tokenizer(x)
        no_stop.append(' '.join(tokens))
        df[c] = no_stop
        return df

    def inpenc(tok,df,c):
        t = tok
        x = x = [df[c][0]]
        enc = t.texts_to_sequences(x)
        padded = pad_sequences(enc, maxlen=16, padding='post')
        return padded

    def predinp(model,x):
        pred = np.argmax(model.predict(x))
        return pred

    def botp(df3,pred):
        l = df3.user[0].split()
        if len([i for i in l if i in words])==0 :
            pred = 1
        return pred

    def botop(df2,pred):
        x2 = df2.groupby('labels').get_group(pred).shape[0]
        idx1 = np.random.randint(0,x2)
        op = list(df2.groupby('labels').get_group(pred).bot)
        return op[idx1]

    def botans(df3):
        tok = joblib.load('tokenizer_t.pkl')
        word = joblib.load('words.pkl')
        df3 = no_stop_inp(tokenizer, df3, 'user')
        inp = inpenc(tok, df3, 'user')
        pred = predinp(model, inp)
        pred = botp(df3, pred)
        ans = botop(df2, pred)
        return ans

    def get_text():
        x = st.text_input("You : ")
        x=x.lower()
        xx = x[:13]
        if(xx =="start my test"):
            global flag
            flag=0
        input_text  = [x]
        df_input = pd.DataFrame(input_text,columns=['user'])
        return df_input

    #flag=1
    qvals = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2,"Strongly Disagree": 1}
    st.markdown(
    """
    <div style='background-color: #1e1e1e; padding: 2rem; border-radius: 1rem; box-shadow: 0px 4px 20px rgba(255, 255, 255, 0.1); text-align: center;'>
        <h1 style='font-size: 60px; color: #6A5ACD; margin-bottom: 0;'>🧭 PathFinder</h1>
        <h4 style='color: #BBBBBB; margin-top: 0.5rem;'>Navigate Your Future with Confidence 🚀</h4>
    </div><br>
    """,
    unsafe_allow_html=True)
    # --- Banner Image
    banner = Image.open("img/21.png")
    st.image(banner, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Greeting Text
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <h4 style='text-align: center; color: #FFFFFF;'>Hi! I'm <b>PathFinder</b>, your personal career counseling assistant. 💬</h4>
        <h4 style='text-align: center; color: #BBBBBB;'>Select <b>'YES'</b> when you're ready for your Personality Test!</h4>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
        
    st.title("📝 PERSONALITY TEST")
    kr = st.selectbox("Would you like to begin with the test?", ["Select an Option", "Yes", "No"])
    if kr == "Yes":
            kr1 = st.selectbox("Select level of education",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"])

            #####################################  GRADE 10  ###########################################

            if(kr1=="Grade 10"):
                lis = []
                progress = st.progress(0)  # Initialize Progress Bar
                total_questions = 10       # Total Questions for Grade 10
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("I find writing programs for computer applications interesting")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",  "Strongly Disagree"], key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        progress.progress(1/total_questions)
                        st.header("Question 2")
                        st.write("I can understand mathematical problems with ease")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            progress.progress(2/total_questions)
                            st.header("Question 3")
                            st.write("Learning about the existence of individual chemical components is interesting")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                progress.progress(3/total_questions)
                                st.header("Question 4")
                                st.write("The way plants and animals thrive gets me curious")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    progress.progress(4/total_questions)
                                    st.header("Question 5")
                                    st.write("Studying about the way fundamental constituents of the universe interact with each other is fascinating")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        progress.progress(5/total_questions)
                                        st.header("Question 6")
                                        st.write(
                                            "Accounting and business management is my cup of tea")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            progress.progress(6/total_questions)
                                            st.header("Question 7")
                                            st.write(
                                                "I would like to know more about human behaviour, relations and patterns of thinking")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                progress.progress(7/total_questions)
                                                st.header("Question 8")
                                                st.write(
                                                    "I find the need to be aware of stories from the past.")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    progress.progress(8/total_questions)
                                                    st.header("Question 9")
                                                    st.write(
                                                        "I see myself as a sportsperson/professional trainer")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        progress.progress(9/total_questions)
                                                        st.header("Question 10")
                                                        st.write(
                                                            "I enjoy creating works of art")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed🎯")
                                                            #st.write(lis)
                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r"Subjects.csv")

                                                            input_list = lis

                                                            subjects = {1: "Computers",
                                                                        2: "Mathematics",
                                                                        3: "Chemistry",
                                                                        4: "Biology",
                                                                        5: "Physics",
                                                                        6: "Commerce",
                                                                        7: "Psychology",
                                                                        8: "History",
                                                                        9: "Physical Education",
                                                                        10: "Design"}

                                                            def output(listofanswers):
                                                                class my_dictionary(dict):
                                                                    def __init__(self):
                                                                        self = dict()

                                                                    def add(self, key, value):
                                                                        self[key] = value

                                                                ques = my_dictionary()

                                                                for i in range(0, 10):
                                                                    ques.add(i, input_list[i])

                                                                all_scores = []

                                                                for i in range(9):
                                                                    all_scores.append(ques[i] / 5)

                                                                li = []

                                                                for i in range(len(all_scores)):
                                                                    li.append([all_scores[i], i])
                                                                li.sort(reverse=True)
                                                                sort_index = []
                                                                for x in li:
                                                                    sort_index.append(x[1] + 1)
                                                                all_scores.sort(reverse=True)

                                                                a = sort_index[0:5]
                                                                b = all_scores[0:5]
                                                                s = sum(b)
                                                                d = list(map(lambda x: x * (100 / s), b))

                                                                return a, d

                                                            l, data = output(input_list)

                                                            # --- Create Top 5 Recommended Subjects
                                                            out = []
                                                            for i in range(5):
                                                                n = l[i]
                                                                c = subjects[n]
                                                                out.append(c)

                                                            # --- Pie Chart for Recommended Subjects
                                                            output_file("pie.html")
                                                            graph = figure(title="Recommended Subjects", height=500, width=500)

                                                            radians = [math.radians((percent / 100) * 360) for percent in data]

                                                            start_angle = [math.radians(0)]
                                                            prev = start_angle[0]
                                                            for angle in radians[:-1]:
                                                                start_angle.append(prev + angle)
                                                                prev = prev + angle

                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                            x, y = 0, 0
                                                            radius = 0.8
                                                            color = Greens[len(out)]

                                                            graph.xgrid.visible = False
                                                            graph.ygrid.visible = False
                                                            graph.xaxis.visible = False
                                                            graph.yaxis.visible = False

                                                            for i in range(len(out)):
                                                                graph.wedge(
                                                                    x, y, radius,
                                                                    start_angle=start_angle[i],
                                                                    end_angle=end_angle[i],
                                                                    color=color[i],
                                                                    legend_label=f"{out[i]} - {round(data[i])}%"
                                                                )

                                                            graph.add_layout(graph.legend[0], 'right')
                                                            st.bokeh_chart(graph, use_container_width=True)

                                                            # --- More Information on Recommended Subjects
                                                            st.markdown("---")
                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>📚 More Information on the Subjects</h2>", unsafe_allow_html=True)
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            for i in range(5):
                                                                st.subheader(out[i])
                                                                st.write(df['about'][int(l[i]) - 1])

                                                            # --- Choice of Degrees
                                                            st.markdown("---")
                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>🎓 Choice of Degrees</h2>", unsafe_allow_html=True)
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            for i in range(5):
                                                                st.subheader(out[i])
                                                                st.write(df['further career'][int(l[i]) - 1])

                                                            # --- Trends Over the Years
                                                            st.markdown("---")
                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>📈 Industry Trends Over the Years</h2>", unsafe_allow_html=True)
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            def Convert(string):
                                                                return list(map(float, string.split(",")))

                                                            x_years = ['2000', '2005', '2010', '2015', '2020']
                                                            y_trends = []

                                                            for i in range(5):
                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                y_trends.append(t)

                                                            output_file("line.html")
                                                            graph2 = figure(title="Trends", height=500, width=700)

                                                            colors_trend = ["Purple", "Blue", "Green", "Magenta", "Red"]

                                                            for i in range(5):
                                                                graph2.line(x_years, y_trends[i], line_color=colors_trend[i], legend_label=out[i])

                                                            graph2.xgrid.visible = True
                                                            graph2.ygrid.visible = True
                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                            # --- Divider for Personalized Career Recommendation
                                                            st.markdown("---")
                                                            st.markdown(
                                                                """
                                                                <h2 style='text-align: center; color: #FF6F61; font-size: 36px; padding-top: 20px;'>🎯 Personalized Career Recommendations</h2>
                                                                <hr style='border: 2px solid #FF6F61; width: 50%; margin: auto; box-shadow: 0 0 10px #FF6F61;'>
                                                                """,
                                                                unsafe_allow_html=True
                                                            )
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            # --- Session State Initialization
                                                            if 'learning_output' not in st.session_state:
                                                                st.session_state.learning_output = ""
                                                            if 'career_output' not in st.session_state:
                                                                st.session_state.career_output = ""
                                                            if 'expand_learning' not in st.session_state:
                                                                st.session_state.expand_learning = False
                                                            if 'expand_career' not in st.session_state:
                                                                st.session_state.expand_career = False

                                                            # --- Selectbox to choose Subject
                                                            selected_field = st.selectbox(
                                                                "Select a Subject for Personalized Guidance",
                                                                out,
                                                                key='guidance_selectbox_final_10'  # Key should be different from Grade 12!
                                                            )

                                                            # --- Buttons Side by Side
                                                            col1, col2 = st.columns(2)

                                                            with col1:
                                                                if st.button("🎓 Generate Learning Path", key='learning_path_button_10'):
                                                                    with st.spinner('Generating your personalized learning path... ⏳'):
                                                                        response = genai_model.generate_content(
                                                                            f"Generate a detailed learning path with courses, certifications, and skills needed to excel in {selected_field}."
                                                                        )
                                                                        st.session_state.learning_output = response.text
                                                                        st.session_state.expand_learning = True
                                                                        st.session_state.expand_career = False

                                                            with col2:
                                                                if st.button("🚀 Generate Career Growth Path", key='career_growth_button_10'):
                                                                    with st.spinner('Mapping your career growth... 📈'):
                                                                        response = genai_model.generate_content(
                                                                            f"Show how a career in {selected_field} evolves over the next 5-10 years including promotions and leadership positions."
                                                                        )
                                                                        st.session_state.career_output = response.text
                                                                        st.session_state.expand_career = True
                                                                        st.session_state.expand_learning = False

                                                            # --- Custom CSS for Dark Theme Cards
                                                            st.markdown("""
                                                                <style>
                                                                .card {
                                                                    background-color: #1e1e1e;
                                                                    padding: 25px;
                                                                    margin-top: 10px;
                                                                    margin-bottom: 20px;
                                                                    border-radius: 15px;
                                                                    box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.05);
                                                                    font-size: 16px;
                                                                    line-height: 1.6;
                                                                    color: #ffffff;
                                                                    transition: all 0.3s ease-in-out;
                                                                }
                                                                .card:hover {
                                                                    transform: translateY(-5px);
                                                                    box-shadow: 0px 8px 20px rgba(255, 255, 255, 0.15);
                                                                }
                                                                .learning-card {
                                                                    background-color: #292929;
                                                                }
                                                                .career-card {
                                                                    background-color: #232b2b;
                                                                }
                                                                </style>
                                                            """, unsafe_allow_html=True)

                                                            # --- Expander Cards for Outputs
                                                            if st.session_state.learning_output:
                                                                with st.expander("🎓 View Your Personalized Learning Path", expanded=st.session_state.expand_learning):
                                                                    st.markdown(f"<div class='card learning-card'>{st.session_state.learning_output}</div>", unsafe_allow_html=True)

                                                            if st.session_state.career_output:
                                                                with st.expander("🚀 View Your Career Growth Roadmap", expanded=st.session_state.expand_career):
                                                                    st.markdown(f"<div class='card career-card'>{st.session_state.career_output}</div>", unsafe_allow_html=True)
                                                            
                                                            st.markdown("---")
                                                            st.markdown(
                                                                """
                                                                <h4 style='text-align: center; color: #BBBBBB;'>Thanks for using <b>PathFinder</b> — Your Future Awaits! 🌟</h4>
                                                                """,
                                                                unsafe_allow_html=True
                                                            )



        ##########################################  GRADE 12  ########################################################

            elif (kr1 == "Grade 12"):
                lis = []
                st.header("Question 1")
                st.write("I enjoy debating and negotiating issues in public")
                n = imagify.imageify(n)
                inp = st.selectbox("",
                                   ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                    "Strongly Disagree"],
                                   key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("Studying the anatomy of the human body and giving first aid to people is something I'm always looking forward to")
                    n = imagify.imageify(n)
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                             "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("I can lead a team and easily manage projects")
                        n = imagify.imageify(n)
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("Working with tools, equipment, and machinery is enjoyable")
                            n = imagify.imageify(n)
                            inp4 = st.selectbox("",
                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write(
                                    "Budgeting, costing and estimating for a business isn't exhausting")
                                n = imagify.imageify(n)
                                inp5 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                     "Disagree",
                                                     "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                        "I can see myself taking part in competitive sporting events to become a professional")
                                    n = imagify.imageify(n)
                                    inp6 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                            "I don't burn out while doing translations, reading and correcting language")
                                        n = imagify.imageify(n)
                                        inp7 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                "I would love to act in or direct a play or film")
                                            n = imagify.imageify(n)
                                            inp8 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree",
                                                                 "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.header("Question 9")
                                                st.write(
                                                    "Making sketches of people or landscapes is a hobby I see as a career")
                                                n = imagify.imageify(n)
                                                inp9 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='9')
                                                if (inp9 != "Select an Option"):
                                                    lis.append(qvals[inp9])
                                                    st.header("Question 10")
                                                    st.write(
                                                        "I can easily work with numbers and calculations most of the time")
                                                    n = imagify.imageify(n)
                                                    inp10 = st.selectbox("",
                                                                         ["Select an Option", "Strongly Agree", "Agree",
                                                                          "Neutral",
                                                                          "Disagree",
                                                                          "Strongly Disagree"], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(qvals[inp10])
                                                        st.header("Question 11")
                                                        st.write(
                                                            "I enjoy doing clerical work i.e. filing, counting stock and issuing receipts")
                                                        n = imagify.imageify(n)
                                                        inp11 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree",
                                                                              "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='11')
                                                        if (inp11 != "Select an Option"):
                                                            lis.append(qvals[inp11])
                                                            st.header("Question 12")
                                                            st.write(
                                                                "I love studying the culture and life style of human societies")
                                                            n = imagify.imageify(n)
                                                            inp12 = st.selectbox("",
                                                                                 ["Select an Option", "Strongly Agree",
                                                                                  "Agree",
                                                                                  "Neutral",
                                                                                  "Disagree",
                                                                                  "Strongly Disagree"], key='12')
                                                            if (inp12 != "Select an Option"):
                                                                lis.append(qvals[inp12])
                                                                st.header("Question 13")
                                                                st.write(
                                                                    "Teaching children and young people is something I see myself doing on a daily basis")
                                                                n = imagify.imageify(n)
                                                                inp13 = st.selectbox("",
                                                                                     ["Select an Option",
                                                                                      "Strongly Agree", "Agree",
                                                                                      "Neutral",
                                                                                      "Disagree",
                                                                                      "Strongly Disagree"], key='13')
                                                                if (inp13 != "Select an Option"):
                                                                    lis.append(qvals[inp13])
                                                                    st.header("Question 14")
                                                                    st.write(
                                                                        "I won't have a problem persevering in the army or police force")
                                                                    n = imagify.imageify(n)
                                                                    inp14 = st.selectbox("",
                                                                                         ["Select an Option",
                                                                                          "Strongly Agree", "Agree",
                                                                                          "Neutral",
                                                                                          "Disagree",
                                                                                          "Strongly Disagree"],
                                                                                         key='14')
                                                                    if (inp14 != "Select an Option"):
                                                                        lis.append(qvals[inp14])
                                                                        st.header("Question 15")
                                                                        st.write(
                                                                            "Introducing consumers to new products and convincing them to buy the same is something that comes with ease")
                                                                        n = imagify.imageify(n)
                                                                        inp15 = st.selectbox("",
                                                                                             ["Select an Option",
                                                                                              "Strongly Agree", "Agree",
                                                                                              "Neutral",
                                                                                              "Disagree",
                                                                                              "Strongly Disagree"],
                                                                                             key='15')
                                                                        if (inp15 != "Select an Option"):
                                                                            lis.append(qvals[inp10])
                                                                            st.success("Test Completed")
                                                                            #st.write(lis)
                                                                            st.title("RESULTS:")
                                                                            df = pd.read_csv(r"Graduate.csv")

                                                                            input_list = lis

                                                                            streams = {1: "Law",
                                                                                       2: "Healthcare",
                                                                                       3: "Management",
                                                                                       4: "Engineering",
                                                                                       5: "Finance",
                                                                                       6: "Sports",
                                                                                       7: "Language and communication",
                                                                                       8: "Performing Arts",
                                                                                       9: "Applied and Visual arts",
                                                                                       10: "Science and math",
                                                                                       11: "Clerical and secretarial",
                                                                                       12: "Social Science",
                                                                                       13: "Education and Social Support",
                                                                                       14: "Armed Forces",
                                                                                       15: "Marketing and sales"}

                                                                            def output(listofanswers):
                                                                                class my_dictionary(dict):
                                                                                    def __init__(self):
                                                                                        self = dict()

                                                                                    def add(self, key, value):
                                                                                        self[key] = value

                                                                                ques = my_dictionary()

                                                                                for i in range(0, 15):
                                                                                    ques.add(i, input_list[i])

                                                                                all_scores = []

                                                                                for i in range(14):
                                                                                    all_scores.append(ques[i] / 5)

                                                                                li = []

                                                                                for i in range(len(all_scores)):
                                                                                    li.append([all_scores[i], i])
                                                                                li.sort(reverse=True)
                                                                                sort_index = []
                                                                                for x in li:
                                                                                    sort_index.append(x[1] + 1)
                                                                                all_scores.sort(reverse=True)

                                                                                a = sort_index[0:5]
                                                                                b = all_scores[0:5]
                                                                                s = sum(b)
                                                                                d = list(
                                                                                    map(lambda x: x * (100 / s), b))

                                                                                return a, d

                                                                            l, data = output(input_list)

                                                                            # --- Create Top 5 Recommended Fields
                                                                            out = []
                                                                            for i in range(5):
                                                                                n = l[i]
                                                                                c = streams[n]
                                                                                out.append(c)

                                                                            # --- Pie Chart for Recommended Fields
                                                                            output_file("pie.html")
                                                                            graph = figure(title="Recommended Fields", height=500, width=500)

                                                                            radians = [math.radians((percent / 100) * 360) for percent in data]

                                                                            start_angle = [math.radians(0)]
                                                                            prev = start_angle[0]
                                                                            for angle in radians[:-1]:
                                                                                start_angle.append(prev + angle)
                                                                                prev = prev + angle

                                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                                            x, y = 0, 0
                                                                            radius = 0.8
                                                                            color = Greens[len(out)]

                                                                            graph.xgrid.visible = False
                                                                            graph.ygrid.visible = False
                                                                            graph.xaxis.visible = False
                                                                            graph.yaxis.visible = False

                                                                            for i in range(len(out)):
                                                                                graph.wedge(
                                                                                    x, y, radius,
                                                                                    start_angle=start_angle[i],
                                                                                    end_angle=end_angle[i],
                                                                                    color=color[i],
                                                                                    legend_label=f"{out[i]} - {round(data[i])}%"
                                                                                )

                                                                            graph.add_layout(graph.legend[0], 'right')
                                                                            st.bokeh_chart(graph, use_container_width=True)

                                                                            # --- More Information on Recommended Fields
                                                                            st.markdown("---")
                                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>📚 More Information on the Fields</h2>", unsafe_allow_html=True)
                                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                                            for i in range(5):
                                                                                st.subheader(out[i])
                                                                                st.write(df['About'][int(l[i]) - 1])

                                                                            # --- Average Annual Salary
                                                                            st.markdown("---")
                                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>💸 Average Annual Salary</h2>", unsafe_allow_html=True)
                                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                                            for i in range(5):
                                                                                st.subheader(out[i])
                                                                                st.write(f"Rs. {df['avgsal'][int(l[i]) - 1]}")

                                                                            # --- Trends Over the Years
                                                                            st.markdown("---")
                                                                            st.markdown("<h2 style='text-align: center; color:#6A5ACD;'>📈 Industry Trends Over the Years</h2>", unsafe_allow_html=True)
                                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                                            # Convert trends
                                                                            def Convert(string):
                                                                                li = list(map(float, string.split(",")))
                                                                                return li

                                                                            x_years = ['2000', '2005', '2010', '2015', '2020']
                                                                            y_trends = []

                                                                            for i in range(5):
                                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                                y_trends.append(t)

                                                                            output_file("line.html")
                                                                            graph2 = figure(title="Trends", height=500, width=700)

                                                                            colors_trend = ["Purple", "Blue", "Green", "Magenta", "Red"]

                                                                            for i in range(5):
                                                                                graph2.line(x_years, y_trends[i], line_color=colors_trend[i], legend_label=out[i])

                                                                            graph2.xgrid.visible = True
                                                                            graph2.ygrid.visible = True
                                                                            graph2.add_layout(graph2.legend[0], 'right')

                                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                                            #✅ Start Personalized Recommendation Divider
                                                                            st.markdown("---")
                                                                            st.markdown(
                                                                                """
                                                                                <h2 style='text-align: center; color: #FF6F61; font-size: 36px; padding-top: 20px;'>🎯 Personalized Career Recommendations</h2>
                                                                                <hr style='border: 2px solid #FF6F61; width: 50%; margin: auto; box-shadow: 0 0 10px #FF6F61;'>
                                                                                """,
                                                                                unsafe_allow_html=True
                                                                            )
                                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                                            # ✅ Initialize session state for Learning Path and Career Growth
                                                                            if 'learning_output' not in st.session_state:
                                                                                st.session_state.learning_output = ""
                                                                            if 'career_output' not in st.session_state:
                                                                                st.session_state.career_output = ""
                                                                            if 'expand_learning' not in st.session_state:
                                                                                st.session_state.expand_learning = False
                                                                            if 'expand_career' not in st.session_state:
                                                                                st.session_state.expand_career = False

                                                                            # ✅ Selectbox to choose Field
                                                                            selected_field = st.selectbox(
                                                                                "Select a Career Field for Personalized Guidance",
                                                                                out,  # 'out' list you already have
                                                                                key='guidance_selectbox_final'
                                                                            )

                                                                            # ✅ Two buttons side-by-side
                                                                            col1, col2 = st.columns(2)

                                                                            with col1:
                                                                                if st.button("🎓 Generate Learning Path", key='learning_path_button'):
                                                                                    with st.spinner('Generating your personalized learning path... ⏳'):
                                                                                        response = genai_model.generate_content(
                                                                                            f"Generate a detailed learning path with courses, certifications, and skills needed to excel as a {selected_field}."
                                                                                        )
                                                                                        st.session_state.learning_output = response.text
                                                                                        st.session_state.expand_learning = True
                                                                                        st.session_state.expand_career = False

                                                                            with col2:
                                                                                if st.button("🚀 Generate Career Growth Path", key='career_growth_button'):
                                                                                    with st.spinner('Mapping your career growth... 📈'):
                                                                                        response = genai_model.generate_content(
                                                                                            f"Show how the career of a {selected_field} evolves over the next 5-10 years including promotions and leadership positions."
                                                                                        )
                                                                                        st.session_state.career_output = response.text
                                                                                        st.session_state.expand_career = True
                                                                                        st.session_state.expand_learning = False

                                                                            # ✅ Custom CSS for Fancy Dark Cards (because your theme is dark)
                                                                            st.markdown("""
                                                                                <style>
                                                                                .card {
                                                                                    background-color: #1e1e1e;
                                                                                    padding: 25px;
                                                                                    margin-top: 10px;
                                                                                    margin-bottom: 20px;
                                                                                    border-radius: 15px;
                                                                                    box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.05);
                                                                                    font-size: 16px;
                                                                                    line-height: 1.6;
                                                                                    color: #ffffff;
                                                                                    transition: all 0.3s ease-in-out;
                                                                                }
                                                                                .card:hover {
                                                                                    transform: translateY(-5px);
                                                                                    box-shadow: 0px 8px 20px rgba(255, 255, 255, 0.15);
                                                                                }
                                                                                .learning-card {
                                                                                    background-color: #292929;
                                                                                }
                                                                                .career-card {
                                                                                    background-color: #232b2b;
                                                                                }
                                                                                </style>
                                                                            """, unsafe_allow_html=True)

                                                                            # ✅ Expander Cards
                                                                            if st.session_state.learning_output:
                                                                                with st.expander("🎓 View Your Personalized Learning Path", expanded=st.session_state.expand_learning):
                                                                                    st.markdown(f"<div class='card learning-card'>{st.session_state.learning_output}</div>", unsafe_allow_html=True)

                                                                            if st.session_state.career_output:
                                                                                with st.expander("🚀 View Your Career Growth Roadmap", expanded=st.session_state.expand_career):
                                                                                    st.markdown(f"<div class='card career-card'>{st.session_state.career_output}</div>", unsafe_allow_html=True)
                                                                            st.markdown("---")
                                                                            st.markdown(
                                                                                """
                                                                                <h4 style='text-align: center; color: #BBBBBB;'>Thanks for using <b>PathFinder</b> — Your Future Awaits! 🌟</h4>
                                                                                """,
                                                                                unsafe_allow_html=True)

            ######################################  UNDERGRADUATE ##########################################

            elif (kr1 == "Undergraduate"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("I can be the person who handles all aspects of information security and protects the virtual data resources of a company")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("I enjoy studying business and information requirements of an organisation and using this data to develop processes that help achieve strategic goals.")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("I can assess a problem and design a brand new system or improve an existing system to make it better and more efficient. ")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("Designing, developing, modifying, editing and working with databases and large datasets is my cup of tea")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write(
                                        "I can mine data using BI software tools, compare, visualize and communicate the results with ease")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                            "Implementing and providing support for Microsoft's Dynamics CRM is a skill I possess")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                                "I can be innovative and creative when it comes to making user-friendly mobile applications")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                    "I can perform well in a varied discipline, combining aspects of psychology, business, market research, design, and technology.")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                        "I am responsible enough to maintain the quality systems, such as laboratory control and document control and training, to ensure control of the manufacturing process.")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "Be it front-end or back-end, I would love designing and developing websites more than anything else")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed")
                                                            #st.write(lis)

                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r'Occupations.csv', encoding= 'windows-1252')

                                                            input_list = lis

                                                            professions = {1: "Systems Security Administrator",
                                                                        2: "Business Systems Analyst",
                                                                        3: "Software Systems Engineer",
                                                                        4: "Database Developer",
                                                                        5: "Business Intelligence Analyst",
                                                                        6: "CRM Technical Developer",
                                                                        7: "Mobile Applications Developer",
                                                                        8: "UX Designer",
                                                                        9: "Quality Assurance Associate",
                                                                        10: "Web Developer"}

                                                            def output(listofanswers):
                                                                # --- Create a dictionary to map questions to answers
                                                                ques = {i: listofanswers[i] for i in range(len(listofanswers))}
                                                                
                                                                all_scores = [ques[i] / 5 for i in range(len(listofanswers))]
                                                                sorted_scores = sorted([(score, idx) for idx, score in enumerate(all_scores)], reverse=True)
                                                                
                                                                top_indices = [idx + 1 for _, idx in sorted_scores[:5]]
                                                                top_scores = [score for score, _ in sorted_scores[:5]]
                                                                
                                                                total = sum(top_scores)
                                                                percentages = [(s * 100) / total for s in top_scores]
                                                                
                                                                return top_indices, percentages

                                                            # --- Process input
                                                            l, data = output(input_list)

                                                            out = [professions[n] for n in l]

                                                            # --- Heading
                                                            st.markdown("<h1 style='text-align: center; color: #4B0082;'>🏆 Your Top Career Matches</h1>", unsafe_allow_html=True)
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            # --- Pie Chart
                                                            output_file("pie.html")
                                                            graph = figure(title="Recommended Professions", height=450, width=450)
                                                            radians = [math.radians((percent / 100) * 360) for percent in data]

                                                            start_angles = [math.radians(0)]
                                                            for angle in radians[:-1]:
                                                                start_angles.append(start_angles[-1] + angle)
                                                            end_angles = start_angles[1:] + [math.radians(0)]

                                                            colors = Greens[len(out)]

                                                            for i in range(len(out)):
                                                                graph.wedge(
                                                                    x=0, y=0, radius=0.8,
                                                                    start_angle=start_angles[i],
                                                                    end_angle=end_angles[i],
                                                                    color=colors[i],
                                                                    legend_label=f"{out[i]} - {round(data[i])}%"
                                                                )

                                                            graph.xgrid.visible = graph.ygrid.visible = False
                                                            graph.xaxis.visible = graph.yaxis.visible = False
                                                            graph.add_layout(graph.legend[0], 'right')

                                                            st.bokeh_chart(graph, use_container_width=True)

                                                            # --- Details of Each Profession
                                                            st.markdown("---")
                                                            st.markdown("## 📚 More Information on the Professions")

                                                            for i in range(5):
                                                                st.subheader(out[i])
                                                                st.write(df['Information'][l[i] - 1])

                                                            # --- Monthly Income
                                                            st.markdown("---")
                                                            st.markdown("## 💸 Expected Monthly Income")

                                                            for i in range(5):
                                                                st.subheader(out[i])
                                                                st.write(f"Rs. {df['Income'][l[i]-1]}")

                                                            # --- Industry Growth Trends
                                                            st.markdown("---")
                                                            st.markdown("## 📈 Industry Growth Trends Over Time")

                                                            def Convert(string):
                                                                return list(map(float, string.split(",")))

                                                            x_years = ['2000', '2005', '2010', '2015', '2020']
                                                            y_trends = [Convert(df['trends'][l[i] - 1]) for i in range(5)]

                                                            graph2 = figure(title="Industry Trends", height=400, width=600)
                                                            colors_trend = ["Purple", "Blue", "Green", "Magenta", "Red"]

                                                            for i in range(5):
                                                                graph2.line(x_years, y_trends[i], line_color=colors_trend[i], legend_label=out[i])

                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)

                                                            # --- Personalized Guidance
                                                            st.markdown("---")
                                                            st.markdown(
                                                                """
                                                                <h2 style='text-align: center; color: #FF6F61; font-size: 36px; padding-top: 20px;'>🎯 Personalized Career Recommendations</h2>
                                                                <hr style='border: 2px solid #FF6F61; width: 50%; margin: auto; box-shadow: 0 0 10px #FF6F61;'>
                                                                """,
                                                                unsafe_allow_html=True)
                                                            st.markdown("<br>", unsafe_allow_html=True)

                                                            # Initialize session state for outputs
                                                            # --- Initialize session state
                                                            if 'learning_output' not in st.session_state:
                                                                st.session_state.learning_output = ""
                                                            if 'career_output' not in st.session_state:
                                                                st.session_state.career_output = ""
                                                            if 'expand_learning' not in st.session_state:
                                                                st.session_state.expand_learning = False
                                                            if 'expand_career' not in st.session_state:
                                                                st.session_state.expand_career = False

                                                            # Dropdown to select field
                                                            selected_field = st.selectbox(
                                                                "Select a Career Field for Personalized Guidance",
                                                                out,
                                                                key='guidance_selectbox_final'
                                                            )

                                                            # Two buttons side-by-side
                                                            col1, col2 = st.columns(2)

                                                            with col1:
                                                                if st.button("🎓 Generate Learning Path", key='learning_path_button'):
                                                                    with st.spinner('Generating your personalized learning path... ⏳'):
                                                                        response = genai_model.generate_content(
                                                                            f"Generate a detailed learning path with courses, certifications, and skills needed to excel as a {selected_field}."
                                                                        )
                                                                        st.session_state.learning_output = response.text
                                                                        st.session_state.expand_learning = True  # Auto expand Learning Path
                                                                        st.session_state.expand_career = False   # Collapse Career if open

                                                            with col2:
                                                                if st.button("🚀 Generate Career Growth Path", key='career_growth_button'):
                                                                    with st.spinner('Mapping your career growth... 📈'):
                                                                        response = genai_model.generate_content(
                                                                            f"Show how the career of a {selected_field} evolves over the next 5-10 years including promotions and leadership positions."
                                                                        )
                                                                        st.session_state.career_output = response.text
                                                                        st.session_state.expand_career = True  # Auto expand Career
                                                                        st.session_state.expand_learning = False  # Collapse Learning if open


                                                            # --- Fancy Expander Card Design
                                                            st.markdown("""
                                                            <style>
                                                            .card {
                                                                background-color: #1e1e1e; /* Dark grey */
                                                                padding: 25px;
                                                                margin-top: 10px;
                                                                margin-bottom: 20px;
                                                                border-radius: 15px;
                                                                box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.05); /* Softer white shadow */
                                                                font-size: 16px;
                                                                line-height: 1.6;
                                                                transition: all 0.3s ease-in-out;
                                                                color: #ffffff; /* White text inside card */
                                                            }
                                                            .card:hover {
                                                                transform: translateY(-5px);
                                                                box-shadow: 0px 8px 20px rgba(255, 255, 255, 0.15); /* Brighter hover shadow */
                                                            }
                                                            .learning-card {
                                                                background-color: #292929; /* Slightly lighter dark */
                                                            }
                                                            .career-card {
                                                                background-color: #232b2b; /* Dark teal shade */
                                                            }
                                                            </style>
                                                            """, unsafe_allow_html=True)

                                                            # Output Sections inside Expanders
                                                            # --- Expander Cards
                                                            if st.session_state.learning_output:
                                                                with st.expander("🎓 View Your Personalized Learning Path", expanded=st.session_state.expand_learning):
                                                                    st.markdown(f"<div class='card learning-card'>{st.session_state.learning_output}</div>", unsafe_allow_html=True)

                                                            if st.session_state.career_output:
                                                                with st.expander("🚀 View Your Career Growth Roadmap", expanded=st.session_state.expand_career):
                                                                    st.markdown(f"<div class='card career-card'>{st.session_state.career_output}</div>", unsafe_allow_html=True)

                                                            def generate_career_growth_path(career_field):
                                                                with st.spinner('Mapping your career growth... 📈'):
                                                                    response = genai_model.generate_content(
                                                                        f"Show how the career of a {career_field} evolves over the next 5-10 years including promotions and leadership positions."
                                                                    )
                                                                    output = response.text
                                                                st.success("🚀 Career Growth Path Generated!")
                                                                st.write(output)
                                                                st.markdown("---")
                                                                st.markdown(
                                                                    """
                                                                    <h4 style='text-align: center; color: #BBBBBB;'>Thanks for using <b>PathFinder</b> — Your Future Awaits! 🌟</h4>
                                                                    """,
                                                                    unsafe_allow_html=True)



if __name__=="__main__":
    main()
