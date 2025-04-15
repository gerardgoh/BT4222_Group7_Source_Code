import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import sys
import traceback
from sentence_transformers import SentenceTransformer

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model for binary interaction prediction.
    """
    def __init__(self, n_users, n_items, factors=32, mlp_layers=[64, 32, 16], dropout=0.2):
        super(NCF, self).__init__()

        # GMF part
        self.user_gmf_embedding = nn.Embedding(n_users, factors)
        self.item_gmf_embedding = nn.Embedding(n_items, factors)

        # MLP part
        self.user_mlp_embedding = nn.Embedding(n_users, factors)
        self.item_mlp_embedding = nn.Embedding(n_items, factors)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * factors

        for next_size in mlp_layers:
            self.mlp_layers.append(nn.Linear(input_size, next_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = next_size

        # Output layer
        self.output_layer = nn.Linear(factors + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, user_id, item_id):
        # GMF part
        user_gmf = self.user_gmf_embedding(user_id)
        item_gmf = self.item_gmf_embedding(item_id)
        gmf_vector = user_gmf * item_gmf

        # MLP part
        user_mlp = self.user_mlp_embedding(user_id)
        item_mlp = self.item_mlp_embedding(item_id)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=1)

        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        # Concatenate GMF and MLP parts
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=1)

        # Output layer
        output = self.output_layer(concat_vector)
        output = self.sigmoid(output)

        return output.squeeze()

class NeuMFPlusPlus(nn.Module):
    def __init__(self, num_users, num_items, item_bert_dim,
                 embedding_dim=64, mlp_dims=[128, 64, 32], dropout_rate=0.2):
        super(NeuMFPlusPlus, self).__init__()

        # GMF part
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_gmf_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP part
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, embedding_dim)

        # Item content projection (Sentence-BERT embedding)
        self.item_bert_projection = nn.Linear(item_bert_dim, embedding_dim)
        self.bert_bn = nn.BatchNorm1d(embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2 + embedding_dim  # user + item + bert
        self.mlp_layers = nn.ModuleList()
        self.mlp_batch_norms = nn.ModuleList()

        # First layer
        self.mlp_layers.append(nn.Linear(mlp_input_dim, mlp_dims[0]))
        self.mlp_batch_norms.append(nn.BatchNorm1d(mlp_dims[0]))

        # Hidden layers
        for i in range(len(mlp_dims)-1):
            self.mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            self.mlp_batch_norms.append(nn.BatchNorm1d(mlp_dims[i+1]))

        # Output layer
        self.output_layer = nn.Linear(mlp_dims[-1] + embedding_dim, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices, item_bert_emb):
        # GMF part
        user_gmf_emb = self.user_gmf_embedding(user_indices)
        item_gmf_emb = self.item_gmf_embedding(item_indices)
        gmf_output = user_gmf_emb * item_gmf_emb

        # MLP part
        user_mlp_emb = self.user_mlp_embedding(user_indices)
        item_mlp_emb = self.item_mlp_embedding(item_indices)

        # Process item Sentence-BERT embedding
        item_bert_emb = self.item_bert_projection(item_bert_emb)
        item_bert_emb = self.bert_bn(item_bert_emb)
        item_bert_emb = self.relu(item_bert_emb)

        # Concatenate user, item and BERT embeddings
        mlp_input = torch.cat([user_mlp_emb, item_mlp_emb, item_bert_emb], dim=1)

        # Apply MLP layers
        for i, layer in enumerate(self.mlp_layers):
            mlp_input = layer(mlp_input)
            mlp_input = self.mlp_batch_norms[i](mlp_input)
            mlp_input = self.relu(mlp_input)
            mlp_input = self.dropout(mlp_input)

        # Concatenate GMF and MLP parts
        concat_output = torch.cat([gmf_output, mlp_input], dim=1)

        # Final prediction
        prediction = self.sigmoid(self.output_layer(concat_output))

        return prediction.squeeze()

class Recommender:
    def __init__(self, metadata_df, reviews_df, metadata_embeddings_path, 
                 ncf_model_path, ncf_encoders_path,
                 query_embeddings=None, has_reviews=None, query_type=None, 
                 item_embeddings_path=None, 
                 sbert_model_path=None, sbert_encoders_path=None,
                 sbert_embeddings_path=None):
        
        self.metadata_df = metadata_df
        self.reviews_df = reviews_df
        self.query_type = query_type
        
        # Load product embeddings for content-based filtering
        loaded_data = np.load(metadata_embeddings_path, allow_pickle=True)
        self.product_embeddings = loaded_data['embeddings']
        self.product_asins = loaded_data['asins'].tolist()
        self.product_metadata = {row['asin']: row.to_dict() for _, row in metadata_df.iterrows()}
        
        # Set device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load binary NCF model
        self.ncf_model, self.n_items = self._load_ncf_model(ncf_model_path)
        self.user_encoder, self.item_encoder = self._load_encoders(ncf_encoders_path)
        self.ncf_model = self.ncf_model.to(self.device)
        
        # Initialize for item embeddings and query embeddings
        self.item_embeddings_path = item_embeddings_path
        self.query_embeddings = query_embeddings
        
        with open(self.item_embeddings_path, 'rb') as f:
            self.item_embeddings_data = pickle.load(f)
        
        # Initialize for SBERT-enhanced NCF model
        self.sbert_model = None
        self.sbert_user_encoder = None
        self.sbert_item_encoder = None
        self.sbert_embeddings = None
        self.asin_to_idx = None
        self.n_sbert_items = 0
        
        # Load SBERT model if provided
        if sbert_model_path and sbert_encoders_path:
            # Load encoders first to get item_bert_dim
            sbert_encoders_data = self._load_sbert_encoders(sbert_encoders_path)
            self.sbert_user_encoder = sbert_encoders_data.get('user_encoder')
            self.sbert_item_encoder = sbert_encoders_data.get('item_encoder')
            self.asin_to_idx = sbert_encoders_data.get('asin_to_idx')
            
            # Load SBERT embeddings
            self.sbert_embeddings = np.load(sbert_embeddings_path)
            item_bert_dim = self.sbert_embeddings.shape[1]
            
            # Load model with the correct item_bert_dim
            self.sbert_model, self.n_sbert_items = self._load_sbert_model(
                sbert_model_path, 
                item_bert_dim=item_bert_dim
            )
            
            self.sbert_model = self.sbert_model.to(self.device)
    
    def _load_encoders(self, encoders_path):
        """Load encoders from a pickle file"""
        with open(encoders_path, 'rb') as f:
            encoders_data = pickle.load(f)
        
        user_encoder = encoders_data.get('user_encoder')
        item_encoder = encoders_data.get('item_encoder')
        
        return user_encoder, item_encoder
    
    def _load_sbert_encoders(self, encoders_path):
        """Load SBERT encoders from a pickle file"""
        with open(encoders_path, 'rb') as f:
            encoders_data = pickle.load(f)
        return encoders_data
    
    def _load_ncf_model(self, model_path):
        torch.serialization.add_safe_globals([LabelEncoder])
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)
        
        # Get basic model parameters
        n_users = checkpoint['n_users']
        n_items = checkpoint['n_items']
        
        # Get embedding dimension (factors)
        factors = checkpoint.get('factors', 32)
        
        # Try to determine MLP layer architecture from the saved model
        state_dict = checkpoint['model_state_dict']
        
        # Find all linear layer weights for MLP
        mlp_linear_keys = []
        for key in state_dict.keys():
            if 'mlp_layers' in key and 'weight' in key:
                layer_index = int(key.split('.')[1])
                if layer_index % 3 == 0:  # Every third layer is linear (Linear, ReLU, Dropout pattern)
                    mlp_linear_keys.append(key)
        
        # Sort by layer index
        mlp_linear_keys.sort(key=lambda x: int(x.split('.')[1]))
        
        # Extract layer sizes
        mlp_layers = []
        for key in mlp_linear_keys:
            weight = state_dict[key]
            out_features = weight.shape[0]
            mlp_layers.append(out_features)
        
        if not mlp_layers:
            # Default if we can't determine
            mlp_layers = [64, 32, 16]
        
        dropout = 0.2  # Default dropout rate
        
        print(f"Creating NCF model with: n_users={n_users}, n_items={n_items}, factors={factors}, mlp_layers={mlp_layers}")
        
        # Create model with the determined architecture
        model = NCF(n_users, n_items, factors=factors, mlp_layers=mlp_layers, dropout=dropout)
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded model parameters")
        except Exception as e:
            print(f"Error loading exact parameters: {e}")
            print("Attempting to load with strict=False")
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model with non-strict parameter matching")
        
        # Set model to evaluation mode
        model.eval()
        
        return model, n_items
    
    def _load_sbert_model(self, model_path, item_bert_dim=384):
        """Load the SBERT-enhanced NCF model"""
        # Add safe globals for model loading
        torch.serialization.add_safe_globals([LabelEncoder])
        
        checkpoint = torch.load(model_path, weights_only=False)
        
        num_users = checkpoint.get('num_users', 0)
        num_items = checkpoint.get('num_items', 0)
        embedding_dim = checkpoint.get('embedding_dim', 64)
        mlp_dims = checkpoint.get('mlp_dims', [128, 64, 32])
        
        model = NeuMFPlusPlus(
            num_users=num_users,
            num_items=num_items,
            item_bert_dim=item_bert_dim,
            embedding_dim=embedding_dim,
            mlp_dims=mlp_dims
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, num_items
    
    def get_cb_recommendations(self, n=5):
        """Get content-based recommendations based on query embeddings"""
        similarity_scores = cosine_similarity(
            self.query_embeddings.reshape(1, -1),
            self.product_embeddings
        )[0]
        
        top_n = similarity_scores.argsort()[-n:][::-1]
        
        # Convert indices to ASINs
        recommendations = []
        for idx in top_n:
            asin = self.product_asins[idx]
            if asin in self.product_metadata:
                recommendations.append(asin)
        
        return recommendations
    
    def get_neumf_recommendations(self, user_id, n=5):
        """Get recommendations using the binary NCF model"""
        try:
            user = self.user_encoder.transform([user_id])[0]
        except Exception as e:
            print(f"User {user_id} not found in binary NCF training data: {e}")
            return []
        
        user_tensor = torch.tensor([user] * self.n_items, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(list(range(self.n_items)), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            predictions = self.ncf_model(user_tensor, item_tensor)
        
        # Get top N items
        _, indices = torch.topk(predictions, n)
        
        # Map indices back to ASINs
        recommendations = []
        for idx in indices:
            item_idx = idx.item()
            try:
                asin = self.item_encoder.inverse_transform([item_idx])[0]
                recommendations.append(asin)
            except Exception as e:
                print(f"Error converting item index {item_idx} to ASIN: {e}")
        
        return recommendations
    
    def get_sbert_neumf_recommendations(self, user_id, n=5):
        """Get recommendations using the SBERT enhanced NCF model"""
        try:
            user = self.sbert_user_encoder.transform([user_id])[0]
        except Exception as e:
            print(f"User {user_id} not found in SBERT training data: {e}")
            return []
        
        # Get all items
        all_items = range(self.n_sbert_items)
        
        # Create tensors for user and items
        user_tensor = torch.tensor([user] * len(all_items), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(list(all_items), dtype=torch.long).to(self.device)
        
        # Get item SBERT embeddings
        item_embeddings_tensor = torch.tensor(self.sbert_embeddings, dtype=torch.float).to(self.device)
        
        # Get predictions in batches to avoid memory issues
        batch_size = 1024
        all_scores = []
        
        for i in range(0, len(all_items), batch_size):
            batch_user = user_tensor[i:i+batch_size]
            batch_items = item_tensor[i:i+batch_size]
            batch_embeddings = item_embeddings_tensor[i:i+batch_size]
            
            with torch.no_grad():
                batch_predictions = self.sbert_model(batch_user, batch_items, batch_embeddings)
                all_scores.append(batch_predictions.cpu())
        
        # Combine all predictions
        all_scores = torch.cat(all_scores)
        
        # Get top N items
        _, indices = torch.topk(all_scores, n)
        
        # Map indices back to ASINs
        recommendations = []
        for idx in indices:
            item_idx = idx.item()
            try:
                asin = self.sbert_item_encoder.inverse_transform([item_idx])[0]
                recommendations.append(asin)
            except Exception as e:
                print(f"Error converting SBERT item index {item_idx} to ASIN: {e}")
        
        return recommendations
    
    def get_item_based_recommendations(self, item_id, n=5):
        """Get item-based collaborative filtering recommendations"""
        item_to_idx = self.item_embeddings_data['item_to_idx']
        idx_to_item = self.item_embeddings_data['idx_to_item']
        embeddings = self.item_embeddings_data['embeddings']
        
        # Get item embedding and calculate similarities
        item_idx = item_to_idx[item_id]
        item_embedding = embeddings[item_idx].reshape(1, -1)
        
        # Calculate similarity with all items
        similarities = cosine_similarity(item_embedding, embeddings)[0]
        
        # Get top similar items
        similar_indices = similarities.argsort()[-(n+1):][::-1]
        
        # Remove the item itself
        similar_indices = [idx for idx in similar_indices if idx != item_idx][:n]
        
        # Convert to ASINs
        recommendations = [idx_to_item[idx] for idx in similar_indices]
        
        return recommendations
    