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
# TODO: note that we need to run the following one time in order for pos tagger to work
# import nltk
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import WhitespaceTokenizer
# TODO: note that we need to run the following one time in order for synset to work
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
from .util import get_wordnet_pos
import torch
import random
from torch.nn import functional as F


class BERTAdversarialDatasetAugmentation:
    def __init__(
        self, baseline, language_model, semantic_sim, tokenizer, mlm, k
    ):
        self.baseline = baseline
        self.language_model = language_model
        self.semantic_sim = semantic_sim # ()
        self.k = k
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = tokenizer
        self.MASK_CHAR = self.tokenizer.mask_token
        random.seed(224)

################################### PRIVATE ###################################
    def _split_sentence(self, sentence_str):
        """
        Split a sentence string into array of word tokens.
        Return the list of word tokens, and a map of the word token index to 
            the offset in the original sentence string.
        """
        spans_generator = WhitespaceTokenizer().span_tokenize(sentence_str)
        sentence = []
        word_idx_to_offset = []
        for span in spans_generator:
            sentence.append(sentence_str[span[0]:span[1]])
            word_idx_to_offset.append(span[0])
        return sentence, word_idx_to_offset

    def _estimate_importance(self, sentence, method='random'):
        """
        Estimates the importance of each token in the sentence, using
        the baseline model.

        Returns mask indices in DESCENDING order of importance.
        """
        if method == 'random':
            idx_importance = [(i, 0.0) for i in range(len(sentence))]
            random.shuffle(idx_importance)
            return idx_importance
        else:
            # TODO: implement importance by baseline 
            raise NotImplementedError
    
    def _pos_tags(self, sentence):
        return pos_tag(sentence)

    def _predict_top_k(self, masked, tag=None, method='synonym'):
        """
        Predicts the top k tokens for masked sentence, of the form
        mask = [t1, t2 ..., t_mask-1, MASK, t_mask+1, .... tn]
        """
        if method == 'synonym':
            syns = wordnet.synsets(masked, pos=get_wordnet_pos(tag))
            lemmas = [lemma for syn in syns for lemma in syn.lemmas()]
            return [l.name() for l in lemmas[:min(self.k, len(lemmas))] if l.name() != masked.lower()]
        else:
            text = " ".join(masked)
            model_input = self.tokenizer.encode_plus(text, return_tensors = "pt")
            mask_index = torch.where(input["input_ids"][0] == self.tokenizer.mask_token_id)
            output = self.mlm(**model_input)
            logits = output.logits
            softmax = F.softmax(logits, dim = -1)
            mask_word = softmax[0, mask_index, :]
            return [
                self.tokenizer.decode([token])
                for token in torch.topk(mask_word, self.k, dim = 1)[1][0]
            ]

    def _filter_tokens(self, tokens, tag):
        """
        Filters tokens, based on https://arxiv.org/pdf/2004.01970.pdf p2-3
        (end of page 2, start of page 3)
        """
        return list(set([x[0] for x in filter(lambda x: x[1] == tag, pos_tag(tokens))]))

    def _baseline_fails(self, perturbed_sentences):
        """
        Given a list of perturbed sentence-answer pairs, returns the perturbed
        sentence-answer pairs in which the model fails to correctly predict the answer.
        """
        return [
            sentence_answer for sentence_answer in perturbed_sentences
            if self.baseline.predict(sentence_answer[0]) != sentence_answer[1] # Probably need to update this TODO
        ]

    def _replace_mask(self, masked, token, original_token, word_idx_to_offset, answer_starts, bae_type):
        """
        Replace masked_character in masked with token.
        Return the new token list, as well as the updated answer starting index. 
        """
        # TODO: we should skip any masking that is in the answer
        mask_index = masked.index(self.MASK_CHAR)
        new_answer_starts = []
        for answer_start in answer_starts:
            new_answer_start = answer_start
            
            if new_answer_start > word_idx_to_offset[mask_index]:
                # if answer starts after the replaced token, update the start
                # to reflect the shift due to token length change.
                if bae_type == 'R':
                    new_answer_start += len(token) - len(original_token)
                elif bae_type == 'I-LEFT' or bae_type == 'I-RIGHT':
                    new_answer_start += len(token)
                else:
                    raise NotImplementedError
            # For now, if we're inserting at the starting answer token, we move the answer start
            # as well
            elif new_answer_start == word_idx_to_offset[mask_index] and (bae_type == 'I-LEFT' or bae_type == 'I-RIGHT'):
                new_answer_start += len(token)
            new_answer_starts.append(new_answer_start)
        return [
            token if item == self.MASK_CHAR else item
            for item in masked 
        ], new_answer_starts

    def _generate_mask(self, sentence, idx, BAE_TYPE='R'):
        if BAE_TYPE == 'R':
            return sentence[:idx] + [self.MASK_CHAR] + sentence[idx+1:]
        
        if BAE_TYPE == "I-LEFT":
            return sentence[:idx] + [self.MASK_CHAR] + sentence[idx:]
        
        if BAE_TYPE == "I-RIGHT":
            return sentence[:idx+1] + [self.MASK_CHAR] + sentence[idx+1:]
        
        raise NotImplementedError

################################### PRIVATE ################################### 
# ----------------------------------------------------------------------------#
################################### PUBLIC ####################################

    def perturb(self, sentence_str, answer_starts, BAE_TYPE='R', use_baseline=False):
        """
        Implements the perturbation algorthim described above.
        Based off of the BAE-R algorithm.
        Returns a list of perturbed sentence strings and their answer indices.
        """
        # split sentence into word spans using
        sentence, word_idx_to_offset = self._split_sentence(sentence_str)
        importance_method = 'baseline' if use_baseline else 'random'
        importances = self._estimate_importance(sentence, method=importance_method) # I (paper)
        tags = self._pos_tags(sentence)

        perturbation_results = []

        for (idx, importance) in importances:
            original_token = sentence[idx]
            tag = tags[idx][1]
            masked = self._generate_mask(sentence, idx, BAE_TYPE)
            tokens = self._predict_top_k(original_token, tag) # T (paper)
            filtered_tokens = self._filter_tokens(tokens, tag)
            perturbed_sentences = [ # L (paper)
                self._replace_mask(masked, token, original_token, word_idx_to_offset, answer_starts, BAE_TYPE)
                for token in filtered_tokens
            ]

            # This is our deviation from BAE!
            # ignore sentences that don't cause baseline to fail
            # TODO: Discuss as group. Should we also allow cases where it doesn't fail?
            if use_baseline:
                perturbed_and_baseline_fails = self._baseline_fails(perturbed_sentences)
            else:
                # just use all of them
                perturbed_and_baseline_fails = perturbed_sentences

            # If no perturbation works, move to the next most important mask.
            if perturbed_and_baseline_fails:
                # Create the sentence strings from the perturbed sentence list as well as the original sentence.
                sim_input_delimited = [sentence] + [sentence_answer[0] for sentence_answer in perturbed_and_baseline_fails]
                sim_input = [" ".join(word_list) for word_list in sim_input_delimited]
                # Get embeddings from the similarity scorer
                embeddings = self.semantic_sim.encode(sim_input)
                # first embedding is of the original sentence.
                original_embedding = embeddings[:1] # 1 x embed_size matrix
                perturbed_embeddings = embeddings[1:] # |perturbed_sentences| x embed_size matrix
                # score() returns 1 x |perturbed_sentences| matrix of similarities by inner product.
                # Flatten and get the index with max score, which is the index of the best sentence.
                scores = self.semantic_sim.score(original_embedding, perturbed_embeddings, method='inner').flatten()
                best_sentence_index = np.argmax(scores)
                # add the sentence string (already created from sim_input) and the answer start.
                perturbation_results.append((sim_input[best_sentence_index+1], perturbed_and_baseline_fails[best_sentence_index][1]))
        
        # Unable to find a good perturbation!
        return perturbation_results

    def perturb_dataset(self, dataset, bae_type='R'):
        """
        Given a dataset, containing (S, y) pairs,
        compute a new dataset, with perturbations!
        """

        for (sentence, answer_starts) in dataset:
            yield self.perturb(sentence, answer_starts, BAE_TYPE=bae_type)

################################### PUBLIC ####################################
