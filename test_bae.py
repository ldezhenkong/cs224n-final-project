import util
import sys, os
from args import get_train_test_args
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW

# define parser and arguments
print('brew')
args = get_train_test_args()
print('buzz')
util.set_seed(args.seed)
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print('foobar!')
if args.do_train:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(args, log)
    train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
    log.info("Preparing Validation Data...")
    val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=SequentialSampler(val_dataset))
    best_scores = trainer.train(model, train_loader, val_loader, val_dict)