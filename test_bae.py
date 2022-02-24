import util
import sys, os, json
import torch
from train import get_dataset
from args import get_train_test_args
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from bae.bae import BERTAdversarialDatasetAugmentation
from semantic_similarity_scorer.sbert import SBERTScorer
import pprint

pp = pprint.PrettyPrinter(indent=4)

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
    _, data_dict = get_dataset(args, args.train_dir_and_datasets, args.train_datasets, args.train_dir, tokenizer, 'train')
    return data_dict

def get_perturbed_sentences(old_data_dict, perturber):

    # initialize dictionary for perturbed data
    new_data_dict = {
        'question': [],
        'context': [],
        'id': [],
        'answer': []
    }
    
    num_data_dict_entires = len(old_data_dict['context'])

    for i in range(num_data_dict_entires):

        if i == 100:
            break

        # TODO: generate and pass in answer_end indices
        old_question = old_data_dict['question'][i]
        old_id = old_data_dict['id'][i]
        old_context = old_data_dict['context'][i]
        perturbation_results = perturber.perturb(old_context, old_data_dict['answer'][i]['answer_start'])

        old_answer_start = old_data_dict['answer'][i]['answer_start'][0]
        old_answer_text = old_data_dict['answer'][i]['text'][0]
        old_answer_num_tokens = len(old_answer_text.split())

        # test #1 that data old_answer_start, old_answer_text, and old_context are consistent
        old_context_slice_1 = old_context[old_answer_start:old_answer_start+len(old_answer_text)]
        assert old_answer_text == old_context_slice_1
        
        # test #2 - in progress
        old_context_slice_2 = ' '.join(old_context[old_answer_start:].split()[0:old_answer_num_tokens])
        if old_context_slice_2[-1] in [',', ';', '.']:
            old_context_slice_2 = old_context_slice_2[:-1]
        old_answer_text == old_context_slice_2
        assert old_answer_text == old_context_slice_1
    
        for elem in perturbation_results:
            new_context, new_answer_start = elem[0], elem[1][0]

            # check that the perturbation does not change the number of tokens
            assert len(new_context.split()) == len(old_context.split()) # ensures tokens are comma delimited

            # calculate new_answer - TODO: figure out punctuation issues
            new_answer_text = ' '.join(new_context[new_answer_start:].split()[:old_answer_num_tokens])

            new_data_dict['question'].append(old_question)
            new_data_dict['id'].append(old_id) # TODO: ids are not currently unique in new_data_dict
            new_data_dict['context'].append(new_context)
            new_data_dict['answer'].append({'answer_start': new_answer_start, 'text': new_answer_text})

    return new_data_dict

def write_to_disk(out_path, new_data_dict):
    # pp.pprint(new_data_dict)

    disk_dict = {'version': '1.1', 'data': []}

    for i in range(len(new_data_dict['question'])):
        question = new_data_dict['question'][i]
        context = new_data_dict['context'][i]
        id = new_data_dict['id'][i]
        answer = new_data_dict['answer'][i]

        qas_obj = [{
            'question': question,
            'id': id,
            'answers' : [answer]
        }]

        TITLE_LENGTH = 50
        padded_title = question[0:TITLE_LENGTH] + ' ' * max(TITLE_LENGTH - len(question), 0)

        disk_dict['data'].append({
            'title': padded_title,
            'paragraphs': [{
                'context': context,
                'qas': qas_obj
            }]}
        )

    print(disk_dict)

    with open(out_path, 'w') as f:
        json.dump(disk_dict, f)
    
    return disk_dict

def main():
    data_dict = get_training_data()
    print('data dict keys:', data_dict.keys())
    q, c, i, a = data_dict['question'], data_dict['context'], data_dict['id'], data_dict['answer']
    for idx in range(len(q)):
        # if idx < len(q)-50:
        #     continue
        if idx == 22:
            break
        print('q:', q[idx])
        print('c:', c[idx])
        print('i:', i[idx])
        print('a:', a[idx])
        # if idx == 5:
        #     break
    perturber = BERTAdversarialDatasetAugmentation(None, None, SBERTScorer(), 10)
    new_data_dict = get_perturbed_sentences(data_dict, perturber)

    out_path = 'datasets/oodomain_train/relation_extraction_perturbed'
    write_to_disk(out_path, new_data_dict)
    # write_to_disk('', all_perturbation_results)
    # for i, perturbation_result in enumerate(all_perturbation_results):
    #     print('old question:', q[i])
    #     print('old context:', c[i])
    #     print('old answer:', a[i])
    #     for elem in perturbation_result:
    #         print('elem:', elem)
    #     print('\n\n')

    # all_perturbation_results.append(perturbation_results)

    # TODO write the perturbed sentences to disk

if __name__ == '__main__':
    main()
