from scorer import SemanticSimilarityScorer
from sentence_transformers import SentenceTransformer

class SBERTScorer(SemanticSimilarityScorer):
    """
    Semantic similarity scorer using SBERT encoder.
    https://sbert.net/
    """
    def __init__(self, pretrain='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(pretrain)
    
    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)