# NLP-exam-prep-assignment

## Table of Contents
1. [Project Description](#description)
2. [Repository Structure](#structure)
3. [Installation](#installation)
4. [Data Source](#data)
5. [Usage](#usage)


## Project Description
This script investigates a dataset of informally written Reddit comments, and loads a Hugging Face sarcasm classification model (helinivan/english-sarcasm-detector) that will be applied to the text. The data is not stored in this repository due to it's size but instructions for downloading the data can be found in point 4. #data in this ReadMe file.


## Repository Structure
```
NLP_TestReport/
│ 
├── data/
│   └ train-balanced-sarcasm.csv
│ 
├── output/
│   ├── 
│   └── 
│
├── src/
│   └── Main.py
│
├── run.sh
├── setup.sh
├── requirements.txt
└── README.md
```

## Installation
To get started with this project, follow these steps:

1. change directories into the  projects repository
2. Follow the instructions under the Data Source section to collect the data
3. In the terminal, run `./setup.sh` to set up the Python virtual environment and install all dependencies from requirements.txt . 
   - If you encounter a permission error, run `chmod +x setup.sh` and try again.
4. Run `./run.sh` to execute the training pipeline.
   - Adjust the script if you are using custom paths.
   - If you encounter a permission error, run `chmod +x run.sh` and try again

## Data Source
The dataset comes from a Kaggle dataset called Sarcasm on Reddit (Sarcasm on Reddit, 2018). 
- Follow the link here: https://www.kaggle.com/datasets/danofer/sarcasm
- Click on the download button found on the page
- Select download Zip file
- Unzip the file and find the file called "train-balanced-sarcasm.csv"
- Add the "train-balanced-sarcasm.csv" to the 'data' folder


The final structure will appear like this:
```
NLP_TestReport/
│ 
├── data/
│   └ train-balanced-sarcasm.csv
```


## Usage
- Coder Python version 1.96.2, has proven to work best on Ucloud with this project.
- Double check the folder_path and output_path (defined under the "Application" section of the main script)are set correctly based on your directory structure.
- The output will be saved in the `output/` folder inside the `Project` directory, unless a different output_path is defined.
