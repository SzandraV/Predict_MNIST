# Hej :)
# This is a "MNIST prediction" Project

# UseCase: MNIST prediction

## DataSet: https://www.openml.org/search?type=data&status=active&id=554&sort=runs

## GitREPO Structure

    ../digits             - Own handwritten digits

    ../models             - Saved trained modell

    ../notebooks          - Workfiles (Jupiter Notebooks)

    ../temp               - Temporary files

# Environment       - CHK IT and Update if it is necessary
- Windows
- Python 3.9.15
- Packages: numpy, matplotlib, sklearn, pandas, seaborn, cv2, joblib, time, PIL, random, streamlit, streamlit_drawable_canvas
- Jupiter NoteBook

# Files in this REPO:
- README.md - Good to know information about this Project :D
- requirements.txt - Necessary Python packages
- DB_Handling.py - SQLite3 - DataBase managing - Class and methods
- ML_Digit_Prediction.py - This is the Main StreamLit Program
- digit_utils.py - Image handling Functions

# Necessary preparation before you start the Program
1. You need to clone this REPO
2. You need to create an appropriate Environment for the Program also - Use requirements.txt in this REPO

# How to start the program
1. step - Start your dedicated Environment for this Program :)
2. step - In your Terminal start the program from the REPOs root DIR:
> streamlit run ML_Digit_Prediction.py

3. "The Digit Prediction" webpage (GUI) is going to pops up in your main Internet Explorer