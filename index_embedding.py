import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer
import faiss
import openai

## testing embedding:
def sentence_embedding(df, column_name):
    """
    Computes sentence embeddings for the specified column in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_name (str): Name of the column to embed (e.g., 'Description', 'Patient', 'Doctor').

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Extract the relevant text column and convert to list
    sentences = df[column_name].tolist()
    # Generate embeddings
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

    

def embed_text_list(text_list):
        """
        Computes sentence embeddings for a list of text queries.

        Args:
            text_list (list): List of text strings to embed.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(text_list, show_progress_bar=True)
        return embeddings   