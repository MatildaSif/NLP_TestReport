# Does the Helinivan/English-Sarcasm-Detector Transfer to Informal Instances of Sarcasm, and Are There Some Emotional Instances Where Sarcasm Is Harder to Detect?

## Table of Contents
1. [Project Description](#description)
2. [Repository Structure](#structure)
3. [Installation](#installation)
4. [Data Source](#data)
5. [Usage](#usage)
6. [Introduction](#introduction)
7. [Methods/Analysis](#methods)
8. [Results](#results)
9. [Discussion](#discussion)
10. [References](#references)
11. [Appendix](#appendix)


## Project Description
This script evaluates how well a Hugging Face sarcasm classification model (helinivan/english-sarcasm-detector) performs on a novel dataset of informally written Reddit comments.
It further investigates the relationship between sarcasm detection errors and emotion labels, using a second model (j-hartmann/emotion-english-distilroberta-base).

## Repository Structure
```
Assignment05/
│ 
├── data/
│   └ train-balanced-sarcasm.csv
│ 
├── output/
│   ├── mismatch_results.txt
│   └── emotion_analysis_stats.txt
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

1. change directories into the  projects repository: /Assignment05
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
Assignment04/
│ 
├── data/
│   └ train-balanced-sarcasm.csv
```


## Usage
- Coder Python version 1.96.2, has proven to work best on Ucloud with this project.
- Double check the folder_path and output_path (defined under the "Application" section of the main script)are set correctly based on your directory structure.
- The output will be saved in the `output/` folder inside the `Assignment05` directory, unless a different output_path is defined.


## Introduction

Sarcasm is a complex communication mechanism whereby seemingly conflicting language is employed to express opinions (Bagga et al., 2024; Jamil et al., 2021; Samer Muthana Sarsam et al., 2020). Hence, sarcasm is the language of secondary meanings and inferred intentions (Ghosh & Veale, 2016). Texts from online platforms are largely characterized by sarcastic and figurative language, hence it is key that language models are able to detect these instances to assuage incorrectly interpreted meanings. For instance, the contrast between “love” and the emotions tied to being ignored, reveals sarcasm in: “I love being ignored” (Joshi et al., 2015). This also highlights how sarcasm often reflects a multitude of emotions. Yet, its relationship with emotion is largely unexplored, despite its relevance. 
The helinivan/english-sarcasm-detector is a text classification model created to detect sarcasm from news headlines (Helinivan/English-Sarcasm-Detector · Hugging Face, n.d.). It has been fine-tuned on bert-base-uncased and trained on formal news headlines from “The Onion”, who produce sarcastic articles, and headlines from “HuffPost”, who create non-sarcastic articles (News Headlines Dataset For Sarcasm Detection, n.d.). Whilst this model has impressive F1 (92.38) and accuracy scores (92.42) (Helinivan/English-Sarcasm-Detector · Hugging Face, n.d.), little is known about its generalizability to other text types, such as informal text. Such an evaluation is critical in order to understand the usability of this model. This is especially pertinent given that most text is informal and much more noisy than professionally written headlines. 
Despite the high F1 and accuracy scores on the training data, we hypothesize that the model will have difficulty classifying sarcasm in informally written text. Additionally, we investigate the relationship between sarcasm detection errors and emotions using an emotion model (J-Hartmann/Emotion-English-Distilroberta-Base · Hugging Face, n.d.). This model was trained on six diverse datasets and associates scores to a neutral class and Ekman’s basic emotions. We hypothesize that neutral emotions hinder sarcasm detection, given sarcasm's usual reliance on contrasting sentiments.


## Methods/Analysis

A balanced sample of 5000 comments is taken from a 1.3M Reddit comment Kaggle dataset. The dataset uses the \s tag, a generally reliable sarcasm indicator, to label sarcastic comments. Binary labels are used (Non-sarcastic: 0 and sarcastic: 1). 
The aforementioned Hugging Face sarcasm classification model (Helinivan/English-Sarcasm-Detector · Hugging Face, n.d.) is loaded, and applied to the subset of comments. Sarcasm is predicted for each comment. For the emotion analysis, the incorrect sarcasm predictions are extracted to a new dataframe. The Hugging Face emotion classification model (J-Hartmann/Emotion-English-Distilroberta-Base · Hugging Face, n.d.) is loaded and applied to the dataframe. The highest scoring emotion for each comment is saved as the “emotion_label”. The percentage of mismatched comments by emotion label relative to the emotions overall presence is calculated.


## Results

The results of the transfer analysis can be seen in Table 01 below with a confusion matrix for the same analysis shown in Table 02. The classifier predicted 49.97% correctly and 50.03% correctly.


| Label         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Not Sarcastic | 0.50      | 0.92   | 0.65     | 4999    |
| Sarcastic     | 0.50      | 0.08   | 0.14     | 5000    |
| **Accuracy**      | —         | —      | **0.50**  | **9999**  |
| Macro Avg     | 0.50      | 0.50   | 0.39     | 9999    |
| Weighted Avg  | 0.50      | 0.50   | 0.39     | 9999    |

**Table 01:** Classification report showing accuracy and F1-scores of the sarcasm detector model on novel data of Reddit comments.

|                      | Predicted Sarcastic | Predicted Not Sarcastic |
|----------------------|---------------------|--------------------------|
| **Actual Sarcastic**     | 397                 | 4603                    |
| **Actual Not Sarcastic** | 399                 | 4600                    |

**Table 02:** Confusion Matrix of correctly and incorrectly labelled comments.

From the emotion analysis, “ anger” had the highest proportion (60.28%) of incorrectly classified comments. Thereafter, the remaining emotions had similar proportions of incorrectly labelled comments (40.45% - 54.66%). A full overview can be found in the appendix (Appendix01, #appendix)


## Discussion

The results of the classification analysis show that the sarcasm classification model does not generalize well from formal news headlines to informal Reddit comments. The model is able to identify most non-sarcastic comments, however given the low precision, the F1 score is mediocre and driven mostly by the high recall. These results suggest that the model is heavily biased towards predicting “Not Sarcastic”. Only 8% of sarcastic comments are identified, and there is generally poor performance. The general accuracy of shows that the model is no better than random guessing.
These results indicate that the applied sarcasm detection model is domain specific, and should not be generalized to informal text. Certainly, news headlines are structurally and linguistically very different from informal, slang-heavy Reddit comments. Sarcasm on social platforms may use different cues, like slang, emojis, formatting and context all which may yet to be learnt by the model. 
Contrary to the hypothesis, comments of “anger” were most difficult for the model to predict. However, all emotions appear to have similar percentages of misclassified comments suggesting that the model has difficulty detecting sarcasm across a range of emotions. Despite this, due to the scope of this project, the emotion labels have not been verified to reflect true sarcastic intent versus literal meaning, so results should be interpreted cautiously.
This  project further delineates the need for domain adaptation in NLP tasks. A model trained on formal text may underperform in new domains unless fine-tuned or adapted accordingly.




## References

   Bagga, H., Bernard, J., Shaheen, S., & Arora, S. (2024). Was that Sarcasm?: A Literature Survey on Sarcasm Detection. https://doi.org/10.48550/arXiv.2412.00425

   Ghosh, A., & Veale, T. (2016). Fracking Sarcasm using Neural Network. In A. Balahur, E. van der Goot, P. Vossen, & A. Montoyo (Eds.), Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (pp. 161–169). Association for Computational Linguistics. https://doi.org/10.18653/v1/W16-0425

   Helinivan/english-sarcasm-detector · Hugging Face. (n.d.). Retrieved 30 April 2025, from https://huggingface.co/helinivan/english-sarcasm-detector

   Jamil, R., Ashraf, I., Rustam, F., Saad, E., Mehmood, A., & Choi, G. S. (2021). Detecting sarcasm in multi-domain datasets using convolutional neural networks and long short term memory network model. PeerJ. Computer Science, 7, e645. https://doi.org/10.7717/peerj-cs.645

   J-hartmann/emotion-english-distilroberta-base · Hugging Face. (n.d.). Retrieved 30 April 2025, from https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

   Joshi, A., Sharma, V., & Bhattacharyya, P. (2015). Harnessing Context Incongruity for Sarcasm Detection. In C. Zong & M. Strube (Eds.), Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 757–762). Association for Computational Linguistics. https://doi.org/10.3115/v1/P15-2124

   News Headlines Dataset For Sarcasm Detection. (n.d.). Retrieved 30 April 2025, from https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

   Samer Muthana Sarsam, Hosam Al-Samarraie, Ahmed Ibrahim Alzahrani, & Bianca Wright. (2020). Sarcasm detection using machine learning algorithms in Twitter: A systematic review. International Journal of Market Research, 62(5), 578–598.

   Sarcasm on Reddit. (2018, May 27). Kaggle. https://www.kaggle.com/datasets/danofer/sarcasm


## Appendix

| Emotion  | Percent Incorrectly Classified |
|----------|-------------------------------:|
| Anger    | 60.28%                         |
| Surprise | 54.66%                         |
| Disgust  | 52.56%                         |
| Neutral  | 48.45%                         |
| Fear     | 44.89%                         |
| Sadness  | 44.13%                         |
| Joy      | 40.45%                         |

**Appendix 01:** A table showing the percentage of incorrectly labelled comments that scored highest on each of Ekman’s 6 basic emotions and a neutral label.
