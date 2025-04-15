# Recommendation System with RAG Pipeline

This repository contains a recommendation system that leverages both collaborative filtering approaches and retrieval-augmented generation (RAG) to provide personalized product recommendations.

## Repository Structure

### Data Cleaning
* `metadata_cleaned.ipynb`: EDA, cleaning, and exporting of metadata CSV, embedding of metadata using SBERT
* `reviews_cleaned.ipynb`: EDA, cleaning, exporting of reviews and embedding of cleaned reviews using SBERT

### Files

#### Binary Classification Models
* `NCF_BCE.ipynb`: Training and running of the baseline NCF Model which optimizes for BCE Loss (Binary Interaction)
* `NeuMF++_Binary.ipynb`: Training and running of the NeuMF++ Model (with SDAE AutoEncoder) which optimizes for BCE Loss
* `SBERT+NCF.ipynb`: Training and running of Sentence BERT with NCF which optimizes for BCE Loss

#### NCF Model and Embeddings
* `encoders.pkl`: User and item matrix used in the baseline NCF Model
* `ncf_binary_model.pt`: The saved model of baseline NCF Model

#### RMSE Models
* `NCF_RMSE.ipynb`: Training and running of the baseline NCF Model which optimizes for RMSE Loss (Exact Ratings)
* `NeuMF++.ipynb`: Training and running of the NeuMF++ Model which optimizes for RMSE Loss (Exact Ratings)

#### SBERT + NCF Model, Encoders, Embeddings
* `item_sbert_embeddings.npy`: Sentence BERT embedding
* `neumf_sbert_model.pt`: The saved model of SBERT + NCF Model
* `sbert_encoders.pkl`: Sentence BERT encoders

#### RAG Pipeline
* `RAG_recommender_system.ipynb`: Contains the pipeline to get user input, generate recommendations, retrieve relevant information, augment prompt and generate LLM output
* `related_fallback_recommendation.py`: Python file for retrieving related products

#### Recommendation Pipeline
* `item_based_cf_embeddings.ipynb`: Embeddings of the user-item matrix used in the item-based collaborative filtering model
* `recommendation_pipeline.ipynb`: Contains the notebook file for all models except related fallback model
* `recommendation_pipeline.py`: The python file of recommendation pipeline to be imported into RAG recommender system

#### Embeddings
* `filtered_metadata_embeddings.npz`: Embeddings of filtered metadata
* `item_embeddings.pkl`: Embeddings of items for item-based collaborative filtering model
* `metadata_embeddings.npz`: Embeddings of metadata
* `reviews_embeddings.npz`: Embeddings of reviews data

## Setup Instructions

### 1. Download Required Files
1. Download all the models from this repository
2. Download the datasets & embeddings through the link

### 2. Adjust File Paths
Update the following variables with the correct file paths:

| **Variable / File** | **Replace with path to the following file** |
|---------------------|---------------------------------------------|
| Base path | Your drive base path |
| Metadata Path | - Use parquet version for related_fallback_model<br>- Use csv version for the rest |
| Reviews Path | - Use csv version for all |
| sbert_embeddings_path | Path to SBERT embeddings |
| neumf_sbert_model_path | Path to NeuMF SBERT model |
| sbert_encoders_path | Path to SBERT encoders |
| sbert_model_path | Path to SBERT model |
| neumf_sbert_encoders_path | - Use in the relevant NeuMF models |
| item_embeddings_path | - Used for the item-based collaborative filtering model create the similarity matrix |
| metadata_embeddings_path | - Use filtered_metadata_embeddings.npz for NeuMF model<br>- Use metadata_embeddings.npz for content-based model and RAG_recommender_system |
| recommendation_models | - Use recommendation_pipeline.py when importing into RAG_recommender_system.ipynb |
| related_fallback_model.py | - Use related_fallback_recommendation.py in the RAG_recommender_system.ipynb for fallback recommendations |

### 3. Running the System
1. Open `RAG_recommender_system.ipynb`
2. Run `main()` in a code cell
3. Enter a userID when prompted
4. Enter the query type:
   * For content-based model and NeuMF Sbert model, enter 1
   * For item-based collaborative filtering and related fallback model, enter 2
5. Enter query:
   * For query type 1, enter a text description of the product being searched
   * For query type 2, enter any product ID from `metadata['asins']`
