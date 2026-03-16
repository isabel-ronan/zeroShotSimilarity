import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import umap
from sentence_transformers import SentenceTransformer

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering
)

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter

class Matave:
    '''
    Matave (Multi-Algorithm Topic Alignment via Vector Embeddings) - topic modelling class. 
    '''
    def __init__(self, texts, random_state = 1618, sentence_transformer_model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
        # Initialize variables. 
        self.texts = texts
        self.random_state = random_state
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.sentence_transformer_model = SentenceTransformer(sentence_transformer_model_name)

        self._set_random_states()
        self._make_embeddings()
        self._make_umap_reduction()

    def _set_random_states(self):
        # Set various random seeds.
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        os.environ["PYTHONHASHSEED"] = str(self.random_state)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def _make_embeddings(self):
        # Make embeddings.
        self.embeddings = self.sentence_transformer_model.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def _make_umap_reduction(self):
        # Make UMAP dimensionality reduction of embeddings.
        self.umap_reduction = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            metric="cosine",
            n_jobs = 1,
            random_state=self.random_state
            
        ).fit_transform(self.embeddings)

    # ------ Utility Functions ------
   
    def cluster_word_distribution(self, labels):
         # Get distribution of words in clusters.
        cluster_texts = defaultdict(list)

        for text, label in zip(self.texts, labels):
            if label == -1:
                continue
            cluster_texts[label].append(text)

        cluster_dists = {}

        for cluster_id, docs in cluster_texts.items():
            if not docs:
                continue

            vectorizer = TfidfVectorizer(
                min_df=1
            )
            X = vectorizer.fit_transform(docs)

            words = vectorizer.get_feature_names_out()
            weights = np.asarray(X.mean(axis=0)).ravel()

            if weights.sum() == 0:
                continue

            probs = weights / weights.sum()
            cluster_dists[int(cluster_id)] = dict(zip(words, probs))

        return cluster_dists
    
   
    def evaluate(self, labels):
        # Get cluster evaluation metric results.  
        labels = np.asarray(labels)
        # Mask to remove noise.
        mask = labels != -1

        # Catch not enough clusters and return zero scores.
        if np.unique(labels[mask]).size < 2:
            return 0.0, 0.0
        
        sil = silhouette_score(
            self.umap_reduction[mask],
            labels[mask],
            metric='cosine'
        )
        # Implicitly uses euclidean (when the embeddings are L2 normalized).
        db = davies_bouldin_score(
            self.umap_reduction[mask],
            labels[mask]
        )
        return sil, db
    
    def get_best_k(self, k_range): 
        # Get the best K using KMeans.
        silhouette_scores = []
        davies_scores = []
        # Run KMeans with all K options.
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(self.umap_reduction)
            silhouette_scores.append(silhouette_score(self.umap_reduction, labels, metric='cosine'))
            davies_scores.append(davies_bouldin_score(self.umap_reduction, labels))

        # Normalize scores.
        scaler = MinMaxScaler()
        sil_norm = scaler.fit_transform(np.array(silhouette_scores).reshape(-1, 1)).flatten()
        db_array = scaler.fit_transform(np.array(davies_scores).reshape(-1, 1)).flatten()
        db_inverted_norm = 1 - db_array

        # Get the best overall score.
        overall = sil_norm + db_inverted_norm
        best_k = k_range[np.argmax(overall)]
        self.best_k = best_k

    def fit(self, k_range = list(range(3, 20))):
        # Get best K using predefined function. 
        self.get_best_k(k_range=k_range)

        # Initialize starting dictionary.
        results = {
            "Method": [],
            "Model": [],
            "Clusters" : [],
            "Silhouette": [],
            "Davies-Bouldin": []
        }
        # Get results using best K for all clustering methods. 
        labels_dict = {}

        # Spectral clustering parameters.
        def adaptive_n_neighbors(n_samples, n_clusters):
            k = max(
                int(np.ceil(2 * np.log(n_samples))),
                2 * n_clusters
            )
            k = min(k, n_samples // 5)
            return max(k, 2)

        n_neighbors = adaptive_n_neighbors(
            n_samples=self.umap_reduction.shape[0],
            n_clusters=self.best_k
        )
        # Make clustering models.
        models = {
            "KMeans": KMeans(
                n_clusters=self.best_k,
                random_state=self.random_state
            ),
            "Hierarchical": AgglomerativeClustering(
                n_clusters=self.best_k,
                metric="cosine",
                linkage="average"
            ),
            "GaussianMixture": GaussianMixture(
                n_components=self.best_k,
                random_state=self.random_state
            ), 
            "SpectralClustering": SpectralClustering(
                n_clusters=self.best_k,
                affinity="nearest_neighbors",
                assign_labels="kmeans",
                random_state=self.random_state, 
                n_neighbors=n_neighbors # Increase to ensure every point is connected to enough neighbours.
            ) 
        }

        # Run all clustering models.
        for name, model in models.items():
            labels = model.fit_predict(self.umap_reduction)
            sil, db = self.evaluate(labels)

            results["Method"].append(name)
            results["Model"].append(model)
            results["Clusters"].append(self.best_k)
            results["Silhouette"].append(sil)
            results["Davies-Bouldin"].append(db)

            labels_dict[name] = labels
        
        self.results = results
        self.labels_dict = labels_dict

        # Get word distributions in each cluster (also get top 10 words in each cluster).
        self.cluster_word_distributions = {}
        self.cluster_keywords = {}

        for method in results["Method"]:
            dists = self.cluster_word_distribution(labels_dict[method])
            self.cluster_word_distributions[method] = dists
            self.cluster_keywords[method] = {
                cid: [w for w, _ in sorted(dist.items(), key=lambda x: -x[1])[:10]]
                for cid, dist in dists.items()
            }

        # Join top words in each cluster and encode joined top words.
        cluster_embeddings = {}
        for method, kw in self.cluster_keywords.items():
            topic_ids = list(kw.keys())
            texts = [" ".join(kw[cid]) for cid in topic_ids]
            emb = self.sentence_transformer_model.encode(texts)
            sims = cosine_similarity(emb)
            sims = sims[np.triu_indices_from(sims, k=1)]
            cluster_embeddings[method] = {
                "keys": topic_ids,
                "topics": texts,
                "embeddings": emb,
                "cosine_similarities": sims,
            }
        # Get max inter-cluster similarity (similarity between clusters of the same algorithm).
        self.max_within_similarity = max([max(v['cosine_similarities']) for v in cluster_embeddings.values() if len(v['cosine_similarities']) > 0])

        all_algos = []
        all_cluster_ids = []
        all_embeddings = []
        all_topics = []

        for method, v in cluster_embeddings.items():
            for cid, e, t in zip(v["keys"], v["embeddings"], v["topics"]):
                all_algos.append(method)
                all_cluster_ids.append(cid)
                all_embeddings.append(e)
                all_topics.append(t)

        # Cluster all clusters from different algorithms to align clusters from different algorithms.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=(1-self.max_within_similarity) # cosine similarity taken from distance derived from closest self-clusters
        )
        topic_groups = clustering.fit_predict(all_embeddings)

        cluster_to_topic = defaultdict(dict)

        for method, cid, tg in zip(all_algos, all_cluster_ids, topic_groups):
            cluster_to_topic[method][cid] = tg
            
        # Normalize to a probability distribution for category/topic weights.
        grouped_topics = Counter(topic_groups)
        total = sum(grouped_topics.values())
        self.category_weights = {
            t: c / total for t, c in grouped_topics.items()
        }
        # Make regrouped document labels (mapping from clustering labels to global labels).
        regrouped_labels = {}

        for method, doc_cluster_labels in labels_dict.items():
            regrouped_labels[method] = [
                cluster_to_topic[method][cluster_id]
                for cluster_id in doc_cluster_labels
            ]

        self.regrouped_labels = regrouped_labels

        # Compute topic votes for each note. 
        topic_ids = sorted(grouped_topics.keys())

        text_topics = []
        for text_idx, text in enumerate(self.texts):
            row = {tid: 0 for tid in topic_ids}
            row["Text"] = text

            for method, doc_labels in regrouped_labels.items():
                if text_idx >= len(doc_labels):
                    continue  # safety guard

                topic_id = doc_labels[text_idx]
                row[topic_id] += 1

            text_topics.append(row)

        counts_df = pd.DataFrame(text_topics)
        row_sums = counts_df[list(grouped_topics.keys())].sum(axis=1)
        # Make proportional df (ratios of all topics for each note).
        ratios_df = counts_df.copy()
        ratios_df[list(grouped_topics.keys())] = ratios_df[
            list(grouped_topics.keys())
        ].div(row_sums, axis=0)
        ratios_df["Text"] = counts_df["Text"]
        # Couple proportional topic score for each note with the category/topic weights (as proxy for the strength of each topic). 
        ratios_with_strength = ratios_df.copy()
        for col in grouped_topics:
            ratios_with_strength[col] *= self.category_weights[col]

        self.counts_df = counts_df
        self.ratios_df = ratios_df
        self.ratios_with_strength_df = ratios_with_strength
        self.assigned_topics = (
            ratios_with_strength[list(grouped_topics.keys())]
            .idxmax(axis=1)
            .tolist()
        )

        vectorizer = TfidfVectorizer(
            min_df=1,
            smooth_idf=True,
            sublinear_tf=True
        )

        X = vectorizer.fit_transform(self.texts)
        terms = vectorizer.get_feature_names_out()
        T = ratios_with_strength[topic_ids].values
        topic_word_matrix = T.T @ X 
        topic_word_matrix = topic_word_matrix / topic_word_matrix.sum(axis=1, keepdims=True)
        word_topic_sum = (topic_word_matrix > 0).sum(axis=0)
        entropy_weight = np.log(topic_word_matrix.shape[0] / (1 + word_topic_sum))
        topic_word_matrix = topic_word_matrix * entropy_weight 

        self.topic_word_distributions = {}

        for i, topic_id in enumerate(topic_ids):
            weights = topic_word_matrix[i] 
            self.topic_word_distributions[topic_id] = dict(
                zip(terms, weights)
            )

        self.top_topic_words = {
            t: " ".join(
                w for w, _ in sorted(d.items(), key=lambda x: -x[1])[:10]
            )
            for t, d in self.topic_word_distributions.items()
        }

        self.document_topic_distributions = (
            self.ratios_with_strength_df.drop(columns=["Text"]).values
        )

        return self

    # Transform method is not implemented. 
    def transform(self, texts=None):
        if texts is None:
            return self.document_topic_distributions
        raise NotImplementedError("Out-of-sample inference not implemented")
    
    # Get the topics as a result of the .fit() method.
    def get_topics(self, n_words=10):
        return {
            topic: sorted(words.items(), key=lambda x: -x[1])[:n_words]
            for topic, words in self.topic_word_distributions.items()
        }

    # Visualize results.
    def visualize(self, title = "", file_name = ""):
        # Text wrapping helper.
        def wrap_text(text, width=50):
            words = text.split()
            lines = []
            current_line = ""
            for w in words:
                if len(current_line) + len(w) + 1 > width:
                    lines.append(current_line)
                    current_line = w
                else:
                    current_line += (" " if current_line else "") + w
            if current_line:
                lines.append(current_line)
            return "<br>".join(lines)

        df = self.ratios_with_strength_df.copy()
        df['x'] = self.umap_reduction[:, 0]
        df['y'] = self.umap_reduction[:, 1]
        
        max_len = 75
        df['hover_text'] = df['Text'].apply(lambda x: x[:max_len] + "..." if len(x) > max_len else x)

        # Handle topics.
        topic_cols = list(self.top_topic_words.keys()) 
        df['dominant_topic'] = df[topic_cols].idxmax(axis=1)

        hover_labels = {topic: f"Topic {topic}" for topic in topic_cols}
        df['dominant_topic_label'] = df['dominant_topic'].map(hover_labels)

        # Make base colours.
        cmap = plt.get_cmap("tab10")
        topic_base_colors = {
            topic: np.array(mcolors.to_rgb(cmap(i % 10)))  # Preserve original colour mapping.
            for i, topic in enumerate(topic_cols)
        }

        # Make colour blends.
        def topic_rgb_hex(row):
            rgb = np.zeros(3)
            for topic in topic_cols:
                rgb += row[topic] * topic_base_colors[topic]
            if rgb.max() > 0:
                rgb = rgb / rgb.max()  # Normalize for vivid colors.
            rgb = np.clip(rgb, 0, 1)
            return mcolors.to_hex(rgb)

        colors = df.apply(topic_rgb_hex, axis=1).tolist()

        # Make hover template.
        hover_template = (
            "<b>%{text}</b><br>" +
            "<br>".join([f"{hover_labels[t]}: %{{customdata[{i}]}}" for i, t in enumerate(topic_cols)])
        )
        customdata = df[topic_cols].values

        scatter = go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                opacity=1
            ),
            text=df['hover_text'],
            hovertemplate=hover_template,
            customdata=customdata,
            showlegend=False
        )

        # Build legend traces with top words.
        legend_traces = [
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=mcolors.to_hex(topic_base_colors[t])
                ),
                name=wrap_text(f"Topic {t}: " + self.top_topic_words[t]),
                showlegend=True
            )
            for t in topic_cols
        ]

        # Sort legend traces by topic number for display only.
        legend_traces = sorted(legend_traces, key=lambda tr: int(tr.name.split()[1].strip(":")))

        fig = go.Figure(data=[scatter] + legend_traces)
        fig.update_layout(
            title="",
            legend_title="Topics",
            legend=dict(font=dict(size=16), itemclick=False, itemdoubleclick=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        fig.show()
        if file_name != "":
            fig.write_image(
                file_name + ".png",
                width=1920,     
                height=1080         
            )

            fig.write_html(
            file_name + ".html",
            full_html=True
            )