import numpy as np
from numpy.linalg import norm

class SemanticSimilarityScorer:
    """
    SemanticSimilarityScorer exposes the interface to calculate semantic similarity between
    sentence pairs. 
    """
    def _score_inner(self, A, B):
        """
        Computes the inner product between A and B.
        
        return: M = AB
        """
        return np.inner(A, B)

    def _score_cos_sim(self, A, B):
        """
        Computes the cosine similarity cos_sim(A[i], B[j]) for all i and j. 
        
        return: Matrix with res[i][j] = cos_sim(A[i], B[j])
        """
        inner = np.inner(A, B) 
        a_norms = np.linalg.norm(A, axis=1)
        b_norms = np.linalg.norm(B, axis=1)
        norms = np.outer(a_norms, b_norms)
        return inner / norms
    
    def score(self, A, B, method='inner'):
        if method == 'inner':
            return self._score_inner(A, B)
        if method == 'cos_sim':
            return self._score_cos_sim(A, B)
        else:
            NotImplementedError("scoring function with '${method}' method not implemented")

    def encode(self, sentences):
        """
        Given array of sentence strings, return array of corresponding embeddings.
        """
        NotImplementedError("encode function not implemented")
