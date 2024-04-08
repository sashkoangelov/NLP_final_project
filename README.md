# NLP Course Repository: Boosting Roberta Model Performance on SQuaD Dataset with Backtranslation

Welcome to our repository for the Natural Language Processing (NLP) course project where we explored the use of backtranslation to enhance the performance of the Roberta model on the Question Answering (QA) dataset, SQuaD.


## Repository Structure

The repository is organized as follows:

- **`dataset/`:** Contains the SQuaD dataset files in JSON format.
- **`misc/`:** Directory containing miscellaneous scripts.
- **`model training/`:** Scripts and notebooks related to model training.

## Scripts

### `backtranslation.py`

This script facilitates the implementation of backtranslation for augmenting the training data. It serves the purpose of enhancing the performance of NLP models, particularly the Roberta model, on the SQuaD dataset. 

#### Purpose:
The main goal of this script is to backtranslate the SQuaD dataset.

#### Usage:
To use the script, follow these steps:

1. **Requirements:**
   - Ensure the SQuaD dataset (`train-v2.0.json`) is present in the `dataset` directory.
   - Install all required libraries from `requirements.txt`.

2. **Settings:**
    - Currently, the script is set up to create two new versions of the dataset by backtranslating with French and German. You can change the selection of languages in the main loop of the script.

3. **Execution:**
    - Run the script with `python backtranslation.py`.

#### Output:
   - Intermediate results are saved in temporary JSON files in the `dataset` directory, such as `temp_file-en-fr-*.json`.
   - The final backtranslated dataset is saved as `train-v2.0-with-back-translation-{lang}.json`, where `{lang}` represents the target language.


### `create_predictions.py`

This script facilitates the generation of predictions for a trained model on the SQuAD dataset.

#### Purpose:
The main purpose of this script is to create predictions using a trained model on the SQuAD dataset.

#### Usage:
To utilize this script, consider the following:

1. **Requirements:**
   - Ensure you have the required libraries installed as specified in `requirements.txt`.
   - Make sure you have a trained model available for making predictions.

2. **Data Preparation:**
   - Ensure the validation dataset is available for prediction.

3. **Adjust Paths:**
   - Adjust the paths in the script according to your system setup, including paths for loading the validation dataset and saving the predictions.

4. **Execution:**
   - Run the script `create_predictions.py`.

#### Output:
   - The script generates predictions for the validation set using the trained model.
   - Predictions are saved to a JSON file named `predictions.json`.

#### Acknowledgement:
Parts of this script have been adopted from the example provided by Hugging Face, available at [this link](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb).


