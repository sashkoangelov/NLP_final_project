# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# 
# !pip install transformers datasets
# !pip install datasets
# !pip install pyngrok

#pip install git+https://github.com/huggingface/accelerate

import pandas as pd
import torch
import random
import accelerate
import transformers
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, RobertaTokenizerFast, RobertaForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset, load_dataset, load_metric
from pyngrok import ngrok
from torch.utils.tensorboard import SummaryWriter
import os
import json
import collections
from tqdm.auto import tqdm

print(transformers.__version__)

"""#Data loading"""

validation = pd.read_parquet("C:/Users/Admin/Downloads/NLP_final_project/validation-00000-of-00001.parquet")

"""# Data pre-processing"""

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

"""Trunctuating longer contexts would not work in our case as we might be losing some importants information. We are going to have our samples produce multiple features by splitting them up and allowing for some overlap so we don't split up in the middle of an answer."""

max_length = 384  # The maximum length of a feature (in tokens).
doc_stride = 128  # The amount of overlap between features.

pad_on_right = tokenizer.padding_side == "right"

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



#load model from C:/Users/Admin/Downloads/second.pth
model = torch.load("C:/Users/Admin/Downloads/model2_backtranslation_overlapV2.pth")

"""#Evaluation"""
# generate predictions and save them to a json file for evaluation using the official SQuAD evaluation script
# the json file will contain the question id, the predicted answer

trainer = Trainer(
    model=model,
    data_collator=default_data_collator,
)

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)


    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} data predictions split into {len(features)} features.")

    # Iterate over all the examples.
    #for example_index, example in enumerate(tqdm(examples)):
    # This line doesnt work because it results in example being a string "id" instead of the actual example
    # so we need to iterate over the indices of the examples instead
    for example_index in range(len(examples)):

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = examples['context'][example_index]

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[examples["id"][example_index]] = best_answer["text"]

    return predictions

validation_hf = Dataset.from_pandas(validation)

validation_features = validation_hf.map(prepare_validation_features, batched=True, remove_columns=validation_hf.column_names)


validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))


raw_predictions = trainer.predict(validation_features)

final_predictions = postprocess_qa_predictions(validation, validation_features, raw_predictions.predictions)


# save the predictions to a json file
with open("predictions.json", "w") as f:
    json.dump(final_predictions, f)




