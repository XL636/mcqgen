from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='XINGHONG LI',
    author_email='lxh04112002@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2","langchain_openai","langchain_community"],
    packages=find_packages()
)