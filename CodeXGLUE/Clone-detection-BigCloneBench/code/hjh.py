import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

from run_parser import get_identifiers, get_example, get_example_batch
import torch
import copy
from transformers import (RobertaForMaskedLM, RobertaTokenizer)
from datasets import load_dataset
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_substitues, is_valid_substitue

checkpoint = 'microsoft/codebert-base'
block_size = 512
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def main():
    # # We could get all the identifiers from a java code snippet in the following way:
    # code = "List < String > function ( ) { if ( bundleUrls == null && hjh = True ) { bundleUrls = new ArrayList < String > ( ) ; } return this . bundleUrls ; }"
    # identifier, processed_code = get_identifiers(code, 'java')
    # replaced_code = get_example(code, 'hjh', 'hjhsb', 'java')
    # print()
    dataset = load_dataset('AhmedSSoliman/CodeXGLUE-CONCODE')
    codebert = RobertaForMaskedLM.from_pretrained(checkpoint)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
    codebert.to(device)
    train_data = dataset['train']
    for entry in train_data:
        code = entry['code']
        identifiers, _ = get_identifiers(code, 'java')
        identifiers = [item[0] for item in identifiers]    
        words, sub_words, keys = _tokenize(code, tokenizer)
        print()
        sub_words = [tokenizer.cls_token] + sub_words[:block_size - 2] + [tokenizer.sep_token]
            
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])

        word_predictions = codebert(input_ids_.to(device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k
        # 得到前k个结果.

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
            ## 得到(所有位置的)substitues
            all_substitues = []
            for one_pos in tgt_positions:
                ## 一个变量名会出现很多次
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
                    # 替换词得到新embeddings

                    with torch.no_grad():
                        new_embeddings = codebert.roberta(new_ids_.to(device))[0]
                    new_word_embed = new_embeddings[0][keys[one_pos][0]+1:keys[one_pos][1]+1]

                    sims.append((i, sum(cos(orig_word_embed, new_word_embed))/subwords_leng))
                
                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                # 排序取top 30 个

                for i in range(int(nums_candis/2)):
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

if __name__ == '__main__':
    main()