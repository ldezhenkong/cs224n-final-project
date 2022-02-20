"""
This file implements BAE-based dataset augmentation.

Our algorithm is a variation on BAE-R, described 
here https://arxiv.org/pdf/2004.01970.pdf. The goal is to
perturb a sentence S into S' in such a way where
a baseline model M will produce incorrect predictions. ie:
    - M(S) != M(S')
    - semantic_similarity(S, S') is maximized.

Class is instantiated with:
    - M :: a baseline model for predicting M(S)
    - LM :: the language model used to predict tokens
    - semantic_similarity :: a function computing the semantic 
      similarity between two sentences.

Input:
    - S = [t1, t2... tn] :: a training example sentence.
    - y :: the ground truth label for S.

Output:
    - S' :: the perturbed sentence
"""

class BERTAdversarialDatasetAugmentation:
    def __init__(
        self, baseline, language_model, semantic_sim
    ):
        self.baseline = baseline
        self.language_model = language_model
        self.semantic_sim = semantic_sim

    def _perturb_R(sentence, label):
        """
        Implements the perturbation algorthim described above.
        Based off of the BAE-R algorithm.

        TODO: Implement me!
        """

    def _perturb_I(sentence, label):
        """
        Implements the perturbation algorthim described above.
        Based off of the BAE-I algorithm.

        TODO: Implement me!
        """

    def perturb_dataset(dataset, **options):
        """
        Given a dataset, containing (S, y) pairs,
        compute a new dataset, with perturbations!
        
        TODO: Implement me!
        """
        ...
