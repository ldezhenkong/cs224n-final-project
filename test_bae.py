import util
import sys, os, json
import torch
from train import get_dataset
from args import get_train_test_args
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from bae.bae import BERTAdversarialDatasetAugmentation
from semantic_similarity_scorer.sbert import SBERTScorer

# define parser and arguments
def get_training_data():
    args = get_train_test_args()
    util.set_seed(args.seed)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _, data_dict = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
    return data_dict

def get_perturbed_sentences(data_dict, perturber):
    for i, context in data_dict['context']:
        # TODO: generate and pass in answer_end indices 
        perturbation_results = perturber.perturb(context, data_dict['answer'][i])
        print(perturbation_results)

def main():
    data_dict = get_training_data()
    perturber = BERTAdversarialDatasetAugmentation(None, None, SBERTScorer(), 10)
    get_perturbed_sentences(data_dict, perturber)
    # TODO write the perturbed sentences to disk

if __name__ == '__main__':
    main()
