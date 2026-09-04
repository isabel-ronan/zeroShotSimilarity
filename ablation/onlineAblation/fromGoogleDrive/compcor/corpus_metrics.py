import random
random.seed(1618)

import numpy as np
from itertools import combinations

from scipy.stats import wasserstein_distance, zscore
from scipy.stats import wasserstein_distance

from compcor.utils import Corpus
import pandas as pd

import time

# ----------------------------------------------------------------
# Added libraries to run zero-shot metrics.
import torch
from transformers import pipeline

# Remove transformers verbosity to clean up space.
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Silence HuggingFace
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Silence Python warnings.
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("pybiber").setLevel(logging.ERROR)



# Variables for zero-shot and traditional biber metrics.
BATCH_SIZE = 128
DEVICE = 0 if torch.cuda.is_available() else -1
# Set epsilon to prevent zero operations.
EPS = 1e-12

# Exact mapping taken from https://www.uni-bamberg.de/fileadmin/eng-ling/fs/Chapter_21/23DimensionsofEnglish.html which has extracted the same from Biber and Conrad's Variation in English (https://doi.org/10.4324/9781315840888)
BIBER_LABEL_MAP = {
    "factor_1": {
        "informational, dense, precise": -1,
        "involved, interactive, affective": 1
    },
    "factor_2": {
        "non-narrative, expository, informational": -1,
        "narrative, event-focused, storytelling": 1
    },
    "factor_3": {
        "situation-dependent, context-bound, implicit": -1,
        "explicit, context-independent, elaborated": 1
    },
    "factor_4": {
        "non-persuasive, non-argumentative, neutral": -1,
        "persuasive, argumentative, modalized": 1
    },
    "factor_5": {
        "non-abstract, concrete, human-centered": -1,
        "abstract, impersonal, technical": 1
    },
    "factor_6": {
        "compressed, dense, clause-poor": -1,
        "elaborated, expanded, clause-rich": 1
    }
}

# Flatten Biber label map (faster inference).
all_labels = []
label_to_factor = {}

for factor, description in BIBER_LABEL_MAP.items():
    for label in description.keys():
        all_labels.append(label)
        label_to_factor[label] = factor


# threshold below which to match distances to 0
ZERO_THRESH = 0.005

# Define models.
ZERO_SHOT_MODELS = [
"cross-encoder/nli-deberta-v3-small", # low capacity
"typeform/distilbert-base-uncased-mnli", # medium capacity
"valhalla/distilbart-mnli-12-3", # higher capacity
]

# prompt templates	
TEMPLATES = {
	"prompt1": "This example is {}.",
	"prompt2": "The writing style of this text is {}.",
	"prompt3": "This text is {}.",
	"prompt4": "This text is written in a {} style.",
	"prompt5": "This text shows {} characteristics."
}


# Make combinations (every possible non-empty model subset)
def all_nonempty_subsets(items):
	return [
		list(combo)
		for r in range(1, len(items) + 1) for combo in combinations(items, r)
	]

model_combinations = all_nonempty_subsets(ZERO_SHOT_MODELS)
prompt_combinations = all_nonempty_subsets(TEMPLATES.keys())

# --------------------------------------------------------------------
# Ablation Tests
# --------------------------------------------------------------------

# Make Wasserstein helper function.
def calculate_score(doc_ids, results, factors, model_names, prompt_names):
	model_results = []
	for model_name in model_names:
		prompt_results = []

		for prompt_name in prompt_names:
			prompt_results.append(results[model_name, prompt_name]['scores'])

		model_score = np.mean(
			prompt_results,
			axis = 0
		)
		df_model = pd.DataFrame(
					model_score.T,
					columns=factors
				)
		df_model["doc_id"] = doc_ids
		df_model["model_name"] = model_name

		model_results.append(df_model)

	df = pd.concat(model_results, ignore_index=True)
	factor_cols = factors

	df[factor_cols] = df[factor_cols].apply(zscore)
	df = df.drop(columns='model_name').groupby("doc_id", sort=False).mean().reset_index()

	df['category'] = np.where(
			df['doc_id'].str.startswith("corpus1"),
			"corpus1",
			"corpus2"
		)

	corpus1_arr = df[df["category"] == "corpus1"][factor_cols].values
	corpus2_arr = df[df["category"] == "corpus2"][factor_cols].values

	wasserstein_per_dim = []

	for i in range(len(factor_cols)):
		dim1 = corpus1_arr[:, i]
		dim2 = corpus2_arr[:, i]

		# Remove per-dimension nans.
		dim1 = dim1[~np.isnan(dim1)]
		dim2 = dim2[~np.isnan(dim2)]

		if len(dim1) == 0 or len(dim2) == 0:
			continue

		wasserstein_per_dim.append(
			wasserstein_distance(dim1, dim2)
		)

	return np.mean(wasserstein_per_dim)


def zero_wasserstein_distance(corpus1: Corpus, corpus2: Corpus):

	# ----------------------------------------------------------------
	# Prepare texts. 
	texts = corpus1 + corpus2
	n_texts = len(texts)

	doc_ids = np.array(
		[f"corpus1_{i}" for i in range(len(corpus1))] +
		[f"corpus2_{i}" for i in range(len(corpus2))]
	)

	assert n_texts == len(doc_ids), "texts and doc_ids are not of the same length"

	# Precompute label → (factor_index, weight)
	factors = list(BIBER_LABEL_MAP.keys())
	factor_to_idx = {f: i for i, f in enumerate(factors)}
	label_map = {
		label: (factor_to_idx[f], BIBER_LABEL_MAP[f][label])
		for f in BIBER_LABEL_MAP
		for label in BIBER_LABEL_MAP[f]
	}

	results = {}

	for model_name in ZERO_SHOT_MODELS:
		# Use numpy for fast accumulation
		classifier = pipeline(
			"zero-shot-classification",
			model = model_name,
			device = DEVICE
		)

		for prompt_name, template in TEMPLATES.items():

			start_time = time.time()

			factor_scores = np.zeros((len(factors), n_texts), dtype=np.float32)

			with torch.no_grad():
				outputs = classifier(
					texts,
					candidate_labels=all_labels,
					hypothesis_template=template,
					multi_label=True,
					batch_size=BATCH_SIZE
				)

			end_time = time.time()

			elapsed_time = end_time - start_time

			if isinstance(outputs, dict):
				outputs = [outputs]

			for j, output in enumerate(outputs):
				for label, score in zip(output['labels'], output['scores']):
					f_idx, weight = label_map[label]
					factor_scores[f_idx, j] += weight * score

			results[(model_name, prompt_name)] = {
				"scores": factor_scores,
				"time": elapsed_time
			}

	combination_rows = []

	for models in model_combinations:
		for prompts in prompt_combinations:
			new_time = 0
			for model in models:
				for prompt in prompts:
					new_time += results[model, prompt]['time']

			start_time = time.time()
			score = calculate_score(
				doc_ids,
				results,
				factors,
				models,
				prompts
			)	
			end_time = time.time()
			elapsed_time = end_time - start_time
			new_time += elapsed_time

			combination_rows.append({
				"models": models,
				"prompts": prompts,
				"wasserstein": score,
				"time": new_time
			})

	combinations_df = pd.DataFrame(combination_rows)
	return combinations_df
