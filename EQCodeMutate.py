import sys
sys.path.append('./python_parser')
from python_parser.run_parser import get_identifiers, get_example, get_example_batch
import torch
import copy
from transformers import (RobertaForMaskedLM, RobertaTokenizer)
from datasets import load_dataset
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_substitues, is_valid_substitue
import json
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

checkpoint = 'microsoft/codebert-base-mlm'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
dataset = load_dataset('AhmedSSoliman/CodeXGLUE-CONCODE')
codebert = RobertaForMaskedLM.from_pretrained(checkpoint)
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
codebert.to(device)
block_size = tokenizer.model_max_length

def get_EQCodeMutate(original_code: str, json_file: str = "EQCodeMutate.json", num_mutants: int = 10) -> None: 
    """
    Args:
        original_code: the code to be mutated
        json_file: the path to save the mutated code
        num_mutants: the number of mutants to be generated
    """
    
    identifiers, _ = get_identifiers(original_code, 'java')
    identifiers = [item[0] for item in identifiers]
    words, sub_words, keys = _tokenize(original_code, tokenizer)
    sub_words = [tokenizer.cls_token] + sub_words[:block_size - 2] + [tokenizer.sep_token]
        
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])

    word_predictions = codebert(input_ids_.to(device))[0].squeeze()  # torch.Size([seq-len(sub), vocab])
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, num_mutants * 2, -1)  # torch.Size([seq-len(sub), k])

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    
    names_positions_dict = get_identifier_posistions_from_code(words, identifiers)

    variable_substitue_dict = {}

    with torch.no_grad():
        orig_embeddings = codebert.roberta(input_ids_.to(device))[0]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for tgt_word, tgt_positions in names_positions_dict.items():
        if not is_valid_variable_name(tgt_word, lang='java'):
            continue   
        ## get substitues for all positions of the same identifier
        all_substitues = []
        for one_pos in tgt_positions:
            if keys[one_pos][0] >= word_predictions.size()[0]:
                continue
            substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]
            
            orig_word_embed = orig_embeddings[0][keys[one_pos][0]+1:keys[one_pos][1]+1]

            similar_substitutes = []
            similar_word_pred_scores = []
            sims = []
            subwords_leng, nums_candis = substitutes.size()

            for i in range(nums_candis):

                new_ids_ = copy.deepcopy(input_ids_)
                new_ids_[0][keys[one_pos][0]+1:keys[one_pos][1]+1] = substitutes[:,i]

                with torch.no_grad():
                    new_embeddings = codebert.roberta(new_ids_.to(device))[0]
                new_word_embed = new_embeddings[0][keys[one_pos][0]+1:keys[one_pos][1]+1]

                sims.append((i, sum(cos(orig_word_embed, new_word_embed))/subwords_leng))
            
            sims = sorted(sims, key=lambda x: x[1], reverse=True)

            for i in range(num_mutants):
                similar_substitutes.append(substitutes[:,sims[i][0]].reshape(subwords_leng, -1))
                similar_word_pred_scores.append(word_pred_scores[:,sims[i][0]].reshape(subwords_leng, -1))

            similar_substitutes = torch.cat(similar_substitutes, 1)
            similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

            substitutes = get_substitues(similar_substitutes, 
                                        tokenizer, 
                                        codebert, 
                                        1, 
                                        similar_word_pred_scores, 
                                        0)
            all_substitues += substitutes
        all_substitues = set(all_substitues)

        for tmp_substitue in all_substitues:
            if tmp_substitue.strip() in identifiers:
                continue
            if not is_valid_substitue(tmp_substitue.strip(), tgt_word, 'java'):
                continue
            try:
                variable_substitue_dict[tgt_word].append(tmp_substitue)
            except:
                variable_substitue_dict[tgt_word] = [tmp_substitue]
    
    # Get all mutated code
    mutated_code_list = []
    while(len(mutated_code_list) < num_mutants):
        mutated_code = original_code
        num_vars_to_mutate = random.randint(1, len(variable_substitue_dict))
        which_vars_to_mutate = random.sample(variable_substitue_dict.keys(), num_vars_to_mutate)
        for var in which_vars_to_mutate:
            substitute = random.choice(variable_substitue_dict[var])
            mutated_code = get_example(mutated_code, var, substitute, 'java')
        if mutated_code not in mutated_code_list:
            mutated_code_list.append(mutated_code)
            logger.info(f"The original code is {original_code}")
            logger.info(f"The mutated code is {mutated_code}")

    json.dump({'code': mutated_code_list}, open(json_file, 'w'))

def main():
    print(get_EQCodeMutate('int multiply (int factor, int another_factor) { int result = factor * another_factor; return result; }'))

if __name__ == '__main__':
    main()