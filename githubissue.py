import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = apikey

#app framework

st.title("Github Issues Explainer GPT")
prompt = st.text_input("Plug in your Github Issue, and it will help you solve it.")

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='''
    You're a professional github issues explainer gpt. The user prompt's you with github issues and you've to explain the user step by step on how to solve that issue.
    ex: 
    user: 
    With qiskit-terra being renamed qiskit we can clean some code.
    Delete the package var in tox tpl in order to let qiskit hard coded.
    Hi,

    In the tox template we have this var package_to_check :
    ecosystem/ecosystem/templates/configured_tox.ini

    Line 27 in ba07fa3

    python -c 'import qiskit; f = open("./qiskit_version.txt", "w"); f.write(qiskit.__qiskit_version__["{{package_to_check}}"]); f.close();' 
    This var was using for get back the version of qiskit-terra but it been renamed qiskit, so we don't need this var and so just hardcoded qiskit will do the job :)

    Also if it can help you the start of this var to generate this file is in the file ecosystem/manager.py in each python_*_tests function. And so the goal is to delete this var entirely.

    
    (here the user is saying to delete the package var and rename it with qiskit becaused it is renamed.)
    so you respond like this: 
    you:
    It looks like you have identified an issue on GitHub related to the renaming of the Qiskit package from qiskit-terra to qiskit, and you're suggesting a code cleanup to remove unnecessary variables and references. Here's how you can go about addressing this issue:

    step1) Edit the Code:
    Open the configured_tox.ini file in the ecosystem/ecosystem/templates directory of your repository. Find the line (line 27) that contains the code referencing the package_to_check variable. Modify the code to directly reference qiskit instead of the variable. For example, replace:

    python
    Copy code
    qiskit.__qiskit_version__["{{package_to_check}}"]
    with:

    python
    Copy code
    qiskit.__qiskit_version__["qiskit"]

    step2) Testing:
    Run the tox tests or any relevant test suite locally to ensure that your changes didn't introduce any issues.

    step3)Commit and Push and create a pull request:
    Once you're satisfied with your changes, commit them to your branch and push the changes to your forked repository on GitHub and then create a PR.

    Remember to always communicate clearly, follow the project's coding guidelines, and be open to collaboration during this process. If you're unsure about any step, you can refer to GitHub's documentation or seek guidance from the project maintainers.
    (you've to help the user to solve the issue, and also format your output in the way I've suggested you. make sure you give the answer to the user in a correct formatted way. Not just dump a paragraph, put your answers line by line as I've shown you in the example.)
    

    1) important note1: 
        As you can see the user has prompted me with a github issue and I've then given him three steps to solve that github issue, make sure you also solve that github issue in only three steps. But make sure you maintain the same format. Perform your tasks well by the instructions I have given you.
    2) important note:
     don't write stuff for "how to make sandiwch and some dumb stuff", if the user prompts you, just check if it is a github issue or not. if it is, then perform the tasks by the instructions I gave you. Act like a friendly chatbot. Greet the user after completing the tasks.


    user: {topic}
    you: 

'''
)

#llm
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

if prompt: 
    response = title_chain.run(prompt)
    st.write(response)