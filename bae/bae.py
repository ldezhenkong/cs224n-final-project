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
    - k :: hyperparm-> number of masks to consider per perturbation.

PUBLIC INTERFACE:
def perturb:
    Input:
        - sentence :: a training example sentence.
        - label :: the ground truth label for S.
        - BAE_TYPE :: whether to apply replace, left insert, or right insert.
    Output:
        - perturbed_sentence :: the perturbed sentence

def perturb_dataset:
    Input:
        - dataset :: iterator of (sentence, label) pairs
    Output:
        - dataset' :: iterator of (perturbed_sentence, label) pairs

"""
import numpy as np

class BERTAdversarialDatasetAugmentation:
    def __init__(
        self, baseline, language_model, semantic_sim, k
    ):
        self.baseline = baseline
        self.language_model = language_model
        self.semantic_sim = semantic_sim # ()
        self.k = k

        self.MASK_CHAR = u"\u2047"

################################### PRIVATE ###################################
    def _estimate_importance(self, sentence):
        """
        Estimates the importance of each token in the sentence, using
        the baseline model.

        Returns mask spots in DESCENDING order of importance.
        """
        # TODO: Implement me!

    def _predict_top_k(self, masked):
        """
        Predicts the top k tokens for masked sentence, of the form
        mask = [t1, t2 ..., t_mask-1, MASK, t_mask+1, .... tn]
        """
        # TODO: Implement me!

    def _filter_tokens(self, tokens):
        """
        Filters tokens, based on https://arxiv.org/pdf/2004.01970.pdf p2-3
        (end of page 2, start of page 3)
        """
        # TODO: Implement me!

    def _baseline_fails(self, perturbed_sentences, label):
        """
        Given a list of perturbed sentences, returns the perturbed
        sentences in which the model fails to correclty predict the label.
        """
        # TODO: Implement me!
    
    def _replace_mask(self, masked, token):
        """
        Replace masked_character in masked with token
        """
        return [
            token if item == self.MASK_CHAR else item
            for item in masked 
        ]

    def _generate_mask(self, sentence, idx, BAE_TYPE='R'):
        if BAE_TYPE == 'R':
            return sentence[:idx] + [self.MASK_CHAR] + sentence[idx+1:]
        
        if BAE_TYPE == "I-LEFT":
            sentence.insert(idx, self.MASK_CHAR)
            return sentence
        
        if BAE_TYPE == "I-RIGHT":
            sentence.insert(idx+1, self.MASK_CHAR)
            return sentence

################################### PRIVATE ################################### 
# ----------------------------------------------------------------------------#
################################### PUBLIC ####################################

    def perturb(self, sentence, label, BAE_TYPE='R'):
        """
        Implements the perturbation algorthim described above.
        Based off of the BAE-R algorithm.
        """
        importances = self._estimate_importance(sentence) # I (paper)

        for (importance, idx) in importances:
            masked = self._generate_mask(sentence, idx, BAE_TYPE)
            tokens = self._predict_top_k(masked) # T (paper)
            filtered_tokens = self._filter_tokens(tokens)
            perturbed_sentences = [ # L (paper)
                self._replace_mask(masked, token)
                for token in filtered_tokens
            ]

            # This is our deviation from BAE!
            # ignore sentences that don't cause baseline to fail
            # TODO: Discuss as group. Should we also allow cases where it doesn't fail?
   
            perturbed_and_baseline_fails = self._baseline_fails(perturbed_sentences, label)

            # If no perturbation works, move to the next most important mask.
            if perturbed_and_baseline_fails:
                # Get embeddings from the similarity scorer
                embeddings = self.semantic_sim.encode([sentence] + perturbed_sentences)
                original_embedding = embeddings[:1] # 1 x embed_size matrix
                perturbed_embeddings = embeddings[1:] # |perturbed_sentences| x embed_size matrix
                # score() returns 1 x |perturbed_sentences| matrix of similarities by inner product.
                # Flatten and get the index with max score, which is the index of the best sentence.
                scores = self.semantic_sim.score(original_embedding, perturbed_embeddings, method='inner').flatten()
                best_sentence_index = np.argmax(scores)
                
                return perturbed_sentences[best_sentence_index]
        
        # Unable to find a good perturbation!
        return []

    def perturb_dataset(self, dataset, bae_type='R'):
        """
        Given a dataset, containing (S, y) pairs,
        compute a new dataset, with perturbations!
        """

        for (sentence, label) in dataset:
            yield self.perturb(sentence, label, BAE_TYPE=bae_type)

################################### PUBLIC ####################################
