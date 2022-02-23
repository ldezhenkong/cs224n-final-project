import util
import sys, os, json
import torch
from train import get_dataset
from args import get_train_test_args
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW

# define parser and arguments
args = get_train_test_args()
util.set_seed(args.seed)
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
log = util.get_logger(args.save_dir, 'log_train')
log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
log.info("Preparing Training Data...")
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
for item in train_dataset:
    print(item)
