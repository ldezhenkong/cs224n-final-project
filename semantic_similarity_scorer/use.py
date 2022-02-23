from .scorer import SemanticSimilarityScorer

import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re

DEFAULT_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

class USEScorer(SemanticSimilarityScorer):
    """
    Semantic similarity scorer using Universal Sentence Encoder.
    https://sbert.net/
    """
    def __init__(self, module_url=DEFAULT_MODULE_URL):
        self.model = hub.load(module_url)
    
    def encode(self, sentences):
        return self.model(sentences)