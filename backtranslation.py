import json
from transformers import MarianMTModel, MarianTokenizer
from contextlib import redirect_stdout, redirect_stderr
import io
from tqdm import tqdm
import os
import torch
import glob


# Check if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_squad_dataset(path: str):
    with open(path, "r") as f:
        squad_dict = json.load(f)
    return squad_dict["data"]

def save_squad_dataset(squad_dict, path: str):
    with open(path, "w") as f:
        json.dump({"data": squad_dict}, f)

def back_translate_paragraph(paragraph, src_lang: str, tgt_lang: str, tokenizer1, model1, tokenizer2, model2):
    paragraph["context"] = back_translate(paragraph["context"], src_lang, tgt_lang, tokenizer1, model1, tokenizer2, model2)
    for qa in paragraph["qas"]:
        qa["question"] = back_translate(qa["question"], src_lang, tgt_lang, tokenizer1, model1, tokenizer2, model2)
        for ans in qa["answers"]:
            ans["text"] = back_translate(ans["text"], src_lang, tgt_lang, tokenizer1, model1, tokenizer2, model2)
    return paragraph

def apply_back_translation(squad_dict, src_lang: str, tgt_lang: str):

    # Use the backtranslation API
    model1_name = "Helsinki-NLP/opus-mt-" + src_lang + "-" + tgt_lang
    tokenizer1 = MarianTokenizer.from_pretrained(model1_name)
    model1 = MarianMTModel.from_pretrained(model1_name).to(device)

    model2_name = "Helsinki-NLP/opus-mt-" + tgt_lang + "-" + src_lang
    tokenizer2 = MarianTokenizer.from_pretrained(model2_name)
    model2 = MarianMTModel.from_pretrained(model2_name).to(device)

    prev_count = 0
    try:
        file_pattern = f"dataset/temp_file-{src_lang}-{tgt_lang}-*.json"
        file_list = glob.glob(file_pattern)

        if file_list:
            # get the last file number
            prev_count = int(file_list[-1].split("-")[-1].split(".")[0])
            squad_dict = json.load(open(file_list[-1], "r"))["data"]

            print(f"Resuming from checkpoint at {prev_count} paragraphs")
    except FileNotFoundError:
        pass

    # progress bar for each paragraph starting at the prevoius count
    num_para = sum([len(article["paragraphs"]) for article in squad_dict])
    progress_bar = tqdm(total=num_para, desc=f"Back Translating from {src_lang} to {tgt_lang}")

    count = 0
    # start loop from the previous count
    for article in squad_dict:
        for paragraph in article["paragraphs"]:
            if count < prev_count:
                back_translate_paragraph(paragraph, src_lang, tgt_lang, tokenizer1, model1, tokenizer2, model2)
            progress_bar.update(1)
            count += 1
            if count % 10 == 0 and prev_count < count:
                # create temp file to save progress
                temp_file = open(f"dataset/temp_file-{src_lang}-{tgt_lang}-{count}.json", "w")
                json.dump({"data": squad_dict}, temp_file)
                temp_file.close()

                # delete the old temp file
                try:
                    os.remove(f"dataset/temp_file-{src_lang}-{tgt_lang}-{count-10}.json")
                except FileNotFoundError:
                    pass
    return squad_dict

def merge_squad_datasets(squad_dict1, squad_dict2):
    return squad_dict1 + squad_dict2

def back_translate(text: str, src_lang: str, tgt_lang: str, tokenizer1, model1, tokenizer2, model2):
    # Add the source and target language code to the text
    translated = model1.generate(**tokenizer1(">>" + src_lang + "<< " + text, return_tensors="pt", padding=True).to(device))

    # decode the translated text
    text_array = [tokenizer1.decode(t, skip_special_tokens=True) for t in translated]

    # combine array into single string
    translated_text = ' '.join(text_array)

    # back translate the text
    translated = model2.generate(**tokenizer2(">>" + tgt_lang + "<< " + translated_text, return_tensors="pt", padding=True).to(device))

    # decode the translated text
    text_array = [tokenizer2.decode(t, skip_special_tokens=True) for t in translated]
    # combine array into single string
    text = ' '.join(text_array)
    return text

if __name__ == "__main__": 
    # Load the SQuAD dataset
    squad_dataset = load_squad_dataset("dataset/train-v2.0.json")

    # Apply back translation to the dataset using Google Translate into 5 different languages
    for lang in ["fr", "de"]:#, "es", "zh", "ja"]:
        #test_string = back_translate("This is a test string to see if this works as well as it should", "en", lang)
        #print(lang, ":  ", test_string)
        squad_dataset_back_translated = apply_back_translation(squad_dataset, "en", lang)
        save_squad_dataset(squad_dataset_back_translated, f"dataset/train-v2.0-with-back-translation-{lang}.json")