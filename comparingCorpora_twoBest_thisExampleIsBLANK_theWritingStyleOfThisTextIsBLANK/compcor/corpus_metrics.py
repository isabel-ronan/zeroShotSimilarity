import random
random.seed(1618)
import statsmodels.stats.multitest
from collections import Counter, namedtuple
from operator import itemgetter

import numpy as np
from prdc import compute_prdc
import prdc.prdc as pr
import mauve

from scipy.linalg import sqrtm
from scipy.stats import chisquare, ttest_ind

from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score
import scipy
import dcor

from compcor.text_embedder import TextTokenizer, TextEmbedder
import compcor.utils as utils
from compcor.utils import Corpus, TCorpus
from compcor.text_tokenizer_embedder import STTokenizerEmbedder


# ----------------------------------------------------------------
# Added libraries to run zero-shot and traditional biber metrics.
import gc
import torch
import sklearn
import pybiber as pb
import polars as pl
import pandas as pd
from scipy.stats import zscore
from transformers import pipeline

# Variables for zero-shot and traditional biber metrics.
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else -1

ZERO_SHOT_MODELS = [
    "cross-encoder/nli-deberta-v3-small", # low capacity
    "typeform/distilbert-base-uncased-mnli", # medium capacity
    "valhalla/distilbart-mnli-12-3", # higher capacity
]

CLASSIFIERS = {
    model_name: pipeline(
    "zero-shot-classification",
    model=model_name,
    device=DEVICE
)
for model_name in ZERO_SHOT_MODELS
}

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

# Using different prompt templates increases robustness.
# TEMPLATES = ["This example is {}.", "This text is {}.", "This text is written in a {} style.", "The writing style of this text is {}.", "This text shows {} characteristics."]
TEMPLATES = ["This example is {}.", "The writing style of this text is {}."]
# ----------------------------------------------------------------

# threshold below which to match distances to 0
ZERO_THRESH = 0.005

PR = namedtuple('pr', 'precision recall distance')
DC = namedtuple('dc', 'density coverage distance')


def cosine_arccos_transform(c1, c2=None):
	# c1 and c2 are lists of input arrays

	def process(input):
		if input is not None:
			if isinstance(input, list) or isinstance(input, tuple):
				return np.vstack(input)
			else:
				if isinstance(input, np.ndarray):
					if len(input.shape) == 1:
						# make it have one row
						return input.reshape(1,-1)
					else:
						return input
		else:
			return input

	c1, c2 = process(c1), process(c2)

	cosine = np.arccos(np.clip(cosine_similarity(X=c1, Y=c2), -1,1)) / np.pi # if None will be X with itself
	# due to numeric precision, sometimes cosine distance between identical vectors is not 0 exactly

	cosine[ cosine <= ZERO_THRESH] = 0.0

	return cosine



def ttest_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# calculate mean and covariance statistics
	if model is not None:
		# if you just provide the matrices themselves
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	res = ttest_ind(embeddings1, embeddings2)
	return 1 - np.nanmean(res.pvalue)


def IRPR_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), components=False):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	table = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	precision = np.nansum(np.nanmin(table, axis=1)) / table.shape[1]
	recall = np.nansum(np.nanmin(table, axis=0)) / table.shape[0]
	distance = 2 * (precision * recall) / (precision + recall)

	return PR(precision, recall, distance) if components else distance


def classifier_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# distance between corpora is the F1 score of a classifier trained to classify membership of a random sample of each
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	corpus1_vecs = embeddings1
	corpus1_train_indx = random.sample(range(len(embeddings1)), k=int(0.8 * len(embeddings1)))
	corpus1_train = itemgetter(*corpus1_train_indx)(corpus1_vecs)

	corpus1_test_indx = set(range(len(embeddings1))) - (set(corpus1_train_indx))
	corpus1_test = itemgetter(*corpus1_test_indx)(corpus1_vecs)

	corpus2_vecs = embeddings2
	corpus2_train_indx = random.sample(range(len(embeddings2)), k=int(0.8 * len(embeddings2)))
	corpus2_train = itemgetter(*corpus2_train_indx)(corpus2_vecs)

	corpus2_test_indx = set(range(len(embeddings2))) - (set(corpus2_train_indx))
	corpus2_test = itemgetter(*corpus2_test_indx)(corpus2_vecs)

	train_x = corpus1_train + corpus2_train
	train_y = [0] * len(corpus1_train) + [1] * len(corpus2_train)
	test_x = corpus1_test + corpus2_test
	test_y = [0] * len(corpus1_test) + [1] * len(corpus2_test)
	clf = svm.SVC(random_state=1)
	clf.fit(train_x, train_y)

	y_pred = clf.predict(test_x)
	correct = f1_score(test_y, y_pred)

	return correct


def medoid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	# calculate mean and covariance statistics
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.mean(act1, axis=0)
	mu2 = np.mean(act2, axis=0)
	# calculate sum squared difference between means
	cosine = spatial.distance.cosine(mu1, mu2)
	return cosine

def median_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	# calculate mean and covariance statistics
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.median(act1, axis=0)
	mu2 = np.median(act2, axis=0)
	# calculate sum squared difference between medians
	cosine = spatial.distance.cosine(mu1, mu2)
	return cosine

def fid_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2
	# TODO: needs a note explaining what the resulting calculation is.  Is it an overlap/probability as approximated by Gaussian curve
	# Note that the paper says FID is a F1 score but this is a different calculation (unless it is in effect an F1 score)
	if len(embeddings1) == 0 or len(embeddings2) == 0:
		return 0
	act1 = np.vstack(embeddings1)
	act2 = np.vstack(embeddings2)
	mu1 = np.mean(act1, axis=0)
	sigma1 = np.cov(act1, rowvar=False)
	mu2 = np.mean(act2, axis=0)
	sigma2 = np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	# ssdiff = np.sum((mu1 - mu2) ** 2.0)
	ssdiff = np.square(mu1 - mu2).sum()
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def mauve_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	out = mauve.compute_mauve(p_features=embeddings1, q_features=embeddings2, device_id=0, verbose=False)
	return 1 - out.mauve


def pr_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5, cosine=False, components=False):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	f = compute_prdc_cosine if cosine else compute_prdc

	metric = f(real_features=np.vstack(embeddings1),
			   fake_features=np.vstack(embeddings2),
			   nearest_k=nearest_k)
	precision = np.clip(metric['precision'], 0, 1)
	recall = np.clip(metric['recall'] + 1e-6, 0, 1)
	distance = 1 - 2 * (precision * recall) / (precision + recall)

	return PR(precision, recall, distance) if components else distance

def dc_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5, cosine=False, components=False):
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	f = compute_prdc_cosine if cosine else compute_prdc

	metric = f(real_features=np.vstack(embeddings1),
			   fake_features=np.vstack(embeddings2),
			   nearest_k=nearest_k)

	density = np.clip(metric['density'], 0, 1)
	coverage = np.clip(metric['coverage'] + 1e-6, 0, 1)
	distance = 1 - 2 * (density * coverage) / (density + coverage)
	return DC(density, coverage, distance) if components else distance


def chi_square_distance(corpus1: TCorpus, corpus2: TCorpus, tokenizer: TextTokenizer = STTokenizerEmbedder(),
						top=5000):
	# calculate p-value of chi-square test between frequency counts of top most frequent shared tokens between corpora
	# note, does not normalize for the size of the corpora, so most common tokens may reflect more the larger corpus
	tokens1, tokens2 = utils.get_corpora_tokens(corpus1, corpus2, tokenizer)

	if type(tokens1[0]) is list:
		tokens1 = [x for xs in tokens1 for x in xs]
		tokens2 = [x for xs in tokens2 for x in xs]

	c1_word_count = Counter(tokens1)
	c2_word_count = Counter(tokens2)
	common_words = set([word for word, freq in Counter(tokens1 + tokens2).most_common(top)])
	sum_count = {word: c1_word_count[word] + c2_word_count[word] for word in common_words}

	N1 = sum([c1_word_count[word] for word in common_words])
	N2 = sum([c2_word_count[word] for word in common_words])
	N = N1 + N2
	o1 = []
	o2 = []
	e1 = []
	e2 = []
	for word in common_words:
		e1 += [sum_count[word] * N1 / N]
		o1 += [c1_word_count[word]]
		e2 += [sum_count[word] * N2 / N]
		o2 += [c2_word_count[word]]

	# low p value means two corpora are different.
	chi_stat = chisquare(f_exp=e1, f_obs=o1)[0] + chisquare(f_exp=e2, f_obs=o2)[0]
	return 1-scipy.stats.chi2.cdf(chi_stat, 2 * (len(common_words) - 1))


def zipf_distance(corpus1: TCorpus, corpus2: TCorpus, tokenizer: TextTokenizer = STTokenizerEmbedder()):
	tokens1, tokens2 = utils.get_corpora_tokens(corpus1, corpus2, tokenizer)
	
	zipf1 = utils.zipf_coeff(tokens1)
	zipf2 = utils.zipf_coeff(tokens2)
	return np.abs(zipf2 - zipf1)


def Directed_Hausdorff_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	# calculate nearest distance from each element in one corpus to an element in the other
	# like IRPR except take mean not harmonic mean (F1-score)
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	table = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	nearest_1to2 = np.nanmin(table, axis=1) # nearest in c2 from each in c1, min in each row
	nearest_2to1 = np.nanmin(table, axis=0)  # nearest in c1 from each in c2, min in each column

	return np.mean([nearest_1to2.mean(), nearest_2to1.mean()])


def Energy_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), normalize=False):
	# https://en.wikipedia.org/wiki/Energy_distance
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	between = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	within1 = cosine_arccos_transform(c1=embeddings1)
	within2 = cosine_arccos_transform(c1=embeddings2)
	A2 = 2 * between.mean()
	B = within1.mean()
	C = within2.mean()

	edist = A2 - B - C
	#  E-coefficient of inhomogeneity is between 0 and 1
	return edist/A2 if normalize else np.sqrt(edist)


def compute_nearest_neighbour_distances_cosine(real_features, nearest_k):
	d = cosine_arccos_transform(c1=real_features) # self distance
	return pr.get_kth_value(d, k=nearest_k + 1, axis=-1)

def compute_prdc_cosine(real_features, fake_features, nearest_k):
	"""
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

	print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

	real_nearest_neighbour_distances = compute_nearest_neighbour_distances_cosine(
        real_features, nearest_k)
	fake_nearest_neighbour_distances = compute_nearest_neighbour_distances_cosine(
        fake_features, nearest_k)
	distance_real_fake = cosine_arccos_transform(c1=real_features, c2=fake_features)

	# precision and recall = are fraction of internal sample distances (interchangeable for our purposes)
	# that are smaller than the distance to each kth nearest neighbor in the other sample
	# each column of the matrix is the probability that elementise, a column in distance_real_fake < real_nearest_neighbour_distances
	# precision looks at probability, for each element in -B, that it is closer to each element of A than that element a's kth NN in B,
	# (i.e whether it is contained in each element of A's NN radius
	# looks if any of these are True, then takes the mean
	# i.e. the share of elements in B that would be hit by the kth NN radius of an element in A.
	precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

	recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

	density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

	coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

	return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)

#  BIBER DISTANCE 
def traditional_biber_distance(corpus1: Corpus, corpus2: Corpus):
	df = pl.DataFrame({
		'doc_id': [f"corpus1_{i}" for i in range(len(corpus1))] + [f"corpus2_{i}" for i in range(len(corpus2))],
		'text' : corpus1 + corpus2
		})

	df = df.with_columns(
		pl.col("text")
		.str.strip_chars()
		.str.replace_all(r"\s+", " ")
		.str.replace_all(r"^\s*-\s*", "") # Remove dashes at the beginning of texts.
		.str.replace_all(r"^\s*\d+\.\s*", "") # Remove numbers in 1., 2., 3. format at the beginning of the text. 
	)

	pybiber_pipeline = pb.PybiberPipeline(model="en_core_web_sm")
	features, tokens = pybiber_pipeline.run(df, return_tokens=True)
	# Full feature list can be found here: https://browndw.github.io/pybiber/feature-categories.html
	features = features.with_columns(pl.col("doc_id").str.split("_").list.get(0).alias("category"))


	# Statistical analysis and visualization
	analyzer = pb.BiberAnalyzer(features, id_column='category')

	# Multi-Dimensional Analysis - see https://browndw.github.io/pybiber/biber-analyzer.html#comparison-with-bibers-original-dimensions for factor mapping
	# Explanation of the factor mapping to dimensions can be found here: https://www.uni-bamberg.de/fileadmin/eng-ling/fs/Chapter_21/23DimensionsofEnglish.html
	'''
	Factor 1: Involved vs. Informational Production (negative to positive)
	Factor 2: Narrative vs. Non-narrative Concerns (negative to positive)
	Factor 3: Explicit vs. Situation-dependent Reference (negative to positive)
	Factor 4: Overt Expression of Persuasion (negative to positive)
	Factor 5: Abstract vs. Non-abstract Information (negative to positive)
	Factor 6: On-line Informational Elaboration (negative to positive)
	'''

	analyzer.mda_biber()

	# Get Z-Scores from Biber analysis.
	biber_dimensions = (analyzer.mda_dim_scores)
	# Get factor columns.
	factor_cols = [c for c in biber_dimensions.columns if c.startswith("factor")]
	biber_dimensions = biber_dimensions.to_pandas()
	biber_dimensions[factor_cols] = biber_dimensions[factor_cols].apply(zscore)
	biber_dimensions = biber_dimensions.groupby('doc_cat').mean(numeric_only=True)

	u = biber_dimensions.loc['corpus1'].values
	v = biber_dimensions.loc['corpus2'].values

	mask = ~np.isnan(u) & ~np.isnan(v)
	u = u[mask].reshape(1, -1) 
	v = v[mask].reshape(1, -1) 
	return sklearn.metrics.pairwise.euclidean_distances(u, v)[0, 0] / np.sqrt(len(u[0])) # Dimension-invariant euclidean distance. 

#  ZERO-SHOT DISTANCE 
def zero_shot_biber_distance_averages_and_euclidean(corpus1: Corpus, corpus2: Corpus):
	texts = corpus1 + corpus2
	doc_ids = [f"corpus1_{i}" for i in range(len(corpus1))] + [f"corpus2_{i}" for i in range(len(corpus2))]

	assert len(texts) == len(doc_ids), "texts and doc_ids are not of the same length"

	all_dfs = []
	for model_name, classifier in CLASSIFIERS.items():
		# Initialize factor storage
		temp_factors_list = {
			'model_name': [model_name] * len(doc_ids),
			'doc_id': doc_ids,
			**{factor: [] for factor in BIBER_LABEL_MAP}
		}

		# Dictionary to accumulate scores per factor across templates
		factor_scores_accum = {factor: [0.0] * len(texts) for factor in BIBER_LABEL_MAP}
		# Loop over all templates
		for template in TEMPLATES:

			with torch.no_grad():
				outputs = classifier(
					texts,
					candidate_labels=all_labels,
					hypothesis_template=template,
					multi_label=True,
					batch_size = BATCH_SIZE
				)

				if isinstance(outputs, dict):
					outputs = [outputs]

				# Accumulate weighted scores per factor
				for j, output in enumerate(outputs):
					for label, score in zip(output['labels'], output['scores']):
						f = label_to_factor[label]
						weight = BIBER_LABEL_MAP[f][label]
						factor_scores_accum[f][j] += weight * score

		# Average over templates
		n_templates = len(TEMPLATES)
		for factor in BIBER_LABEL_MAP:
			factor_scores_accum[factor] = [s / n_templates for s in factor_scores_accum[factor]]
			temp_factors_list[factor].extend(factor_scores_accum[factor])

		# Convert to DataFrame and append to all_dfs
		all_dfs.append(pd.DataFrame(temp_factors_list))

	df = pd.concat(all_dfs)

	# Simple averaging will prevent over-confidence. 
	factor_cols = [c for c in df.columns if c.startswith("factor")]
	df[factor_cols] = df[factor_cols].apply(zscore)
	mean_scores = df.drop(columns='model_name').groupby("doc_id").mean().reset_index()
	mean_scores['category'] = [doc_id.split('_')[0] for doc_id in mean_scores['doc_id']]
	mean_scores = mean_scores.groupby('category').mean(numeric_only=True)

	u = mean_scores.loc['corpus1'].values
	v = mean_scores.loc['corpus2'].values

	mask = ~np.isnan(u) & ~np.isnan(v)
	u = u[mask].reshape(1, -1) 
	v = v[mask].reshape(1, -1) 
	return sklearn.metrics.pairwise.euclidean_distances(u, v)[0, 0] / np.sqrt(len(u[0])) # Dimension-invariant euclidean distance. 



#  ZERO-SHOT DISTANCE 
def zero_shot_biber_distance_wasserstein_distance(corpus1: Corpus, corpus2: Corpus):
	texts = corpus1 + corpus2
	doc_ids = [f"corpus1_{i}" for i in range(len(corpus1))] + [f"corpus2_{i}" for i in range(len(corpus2))]

	assert len(texts) == len(doc_ids), "texts and doc_ids are not of the same length"

	all_dfs = []
	for model_name, classifier in CLASSIFIERS.items():
		# Initialize factor storage
		temp_factors_list = {
			'model_name': [model_name] * len(doc_ids),
			'doc_id': doc_ids,
			**{factor: [] for factor in BIBER_LABEL_MAP}
		}

		# Dictionary to accumulate scores per factor across templates
		factor_scores_accum = {factor: [0.0] * len(texts) for factor in BIBER_LABEL_MAP}
		# Loop over all templates
		for template in TEMPLATES:

			with torch.no_grad():
				outputs = classifier(
					texts,
					candidate_labels=all_labels,
					hypothesis_template=template,
					multi_label=True,
					batch_size = BATCH_SIZE
				)

				if isinstance(outputs, dict):
					outputs = [outputs]

				# Accumulate weighted scores per factor
				for j, output in enumerate(outputs):
					for label, score in zip(output['labels'], output['scores']):
						f = label_to_factor[label]
						weight = BIBER_LABEL_MAP[f][label]
						factor_scores_accum[f][j] += weight * score

		# Average over templates
		n_templates = len(TEMPLATES)
		for factor in BIBER_LABEL_MAP:
			factor_scores_accum[factor] = [s / n_templates for s in factor_scores_accum[factor]]
			temp_factors_list[factor].extend(factor_scores_accum[factor])

		# Convert to DataFrame and append to all_dfs
		all_dfs.append(pd.DataFrame(temp_factors_list))

	df = pd.concat(all_dfs)

	# Get factor columns.
	factor_cols = [c for c in df.columns if c.startswith("factor")]
	# Z-score normalize factors (to reduce model bias / scale differences).
	df[factor_cols] = df[factor_cols].apply(zscore)
	# Average across models per document.
	df = df.drop(columns='model_name').groupby("doc_id").mean().reset_index()
	# Extract corpus label from doc_id
	df['category'] = df['doc_id'].apply(lambda x: x.split('_')[0])

	# Split corpora at document level.
	corpus1 = df[df["category"] == "corpus1"][factor_cols].values
	corpus2 = df[df["category"] == "corpus2"][factor_cols].values

	# Remove NaNs safely.
	corpus1 = corpus1[~np.isnan(corpus1).any(axis=1)]
	corpus2 = corpus2[~np.isnan(corpus2).any(axis=1)]

	wasserstein_per_dim = []
	# Loop over each factor separately.
	for i, factor in enumerate(factor_cols):
		w_dist = wasserstein_distance(
			corpus1[:, i],
			corpus2[:, i]
		)
		wasserstein_per_dim.append(w_dist)

	# Final score is the average wasserstein across all factors.
	return np.mean(wasserstein_per_dim)


#  ZERO-SHOT DISTANCE 
def zero_shot_biber_distance_energy_distance(corpus1: Corpus, corpus2: Corpus):
	texts = corpus1 + corpus2
	doc_ids = [f"corpus1_{i}" for i in range(len(corpus1))] + [f"corpus2_{i}" for i in range(len(corpus2))]

	assert len(texts) == len(doc_ids), "texts and doc_ids are not of the same length"

	all_dfs = []
	for model_name, classifier in CLASSIFIERS.items():
		# Initialize factor storage
		temp_factors_list = {
			'model_name': [model_name] * len(doc_ids),
			'doc_id': doc_ids,
			**{factor: [] for factor in BIBER_LABEL_MAP}
		}

		# Dictionary to accumulate scores per factor across templates
		factor_scores_accum = {factor: [0.0] * len(texts) for factor in BIBER_LABEL_MAP}
		# Loop over all templates
		for template in TEMPLATES:

			with torch.no_grad():
				outputs = classifier(
					texts,
					candidate_labels=all_labels,
					hypothesis_template=template,
					multi_label=True,
					batch_size = BATCH_SIZE
				)

				if isinstance(outputs, dict):
					outputs = [outputs]

				# Accumulate weighted scores per factor
				for j, output in enumerate(outputs):
					for label, score in zip(output['labels'], output['scores']):
						f = label_to_factor[label]
						weight = BIBER_LABEL_MAP[f][label]
						factor_scores_accum[f][j] += weight * score

		# Average over templates
		n_templates = len(TEMPLATES)
		for factor in BIBER_LABEL_MAP:
			factor_scores_accum[factor] = [s / n_templates for s in factor_scores_accum[factor]]
			temp_factors_list[factor].extend(factor_scores_accum[factor])

		# Convert to DataFrame and append to all_dfs
		all_dfs.append(pd.DataFrame(temp_factors_list))

	df = pd.concat(all_dfs)

	# Get factor columns.
	factor_cols = [c for c in df.columns if c.startswith("factor")]
	# Z-score normalize factors (to reduce model bias / scale differences).
	df[factor_cols] = df[factor_cols].apply(zscore)
	# Average across models per document.
	df = df.drop(columns='model_name').groupby("doc_id").mean().reset_index()
	# Extract corpus label from doc_id
	df['category'] = df['doc_id'].apply(lambda x: x.split('_')[0])

	# Split corpora at document level.
	corpus1 = df[df["category"] == "corpus1"][factor_cols].values
	corpus2 = df[df["category"] == "corpus2"][factor_cols].values

	# Remove NaNs safely.
	corpus1 = corpus1[~np.isnan(corpus1).any(axis=1)]
	corpus2 = corpus2[~np.isnan(corpus2).any(axis=1)]

	energy_distance_out = dcor.homogeneity.energy_test(corpus1, corpus2, num_resamples=200, random_state=1)
	return energy_distance_out.statistic
