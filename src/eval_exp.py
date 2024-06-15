import os
import csv
import pandas as pd
import re
import argparse
from rouge import Rouge
from bert_score import BERTScorer
from bart_score import BARTScorer
from bleurt import score as bleurt_score
from ctc_score import SummarizationScorer, StyleTransferScorer
from tqdm import tqdm


def replace_substring(text):
    # Define the mapping of numbers to letters
    replacements = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D',
        '5': 'E'
    }
    
    # Regular expression to match "Answer" or "Answers" followed by number ranges and then a colon
    pattern = re.compile(r'(Answers?\s+)([1-5](-[1-5])?(\s*&\s*[1-5](-[1-5])?)*)\s*:')

    # Function to replace the matched number sequences with corresponding letters
    def replace_match(match):
        prefix = match.group(1)
        number_sequence = match.group(2)
        
        # Function to convert individual numbers or ranges to their letter equivalents
        def convert_to_letters(seq):
            if '-' in seq:
                start, end = seq.split('-')
                return replacements[start] + '-' + replacements[end]
            else:
                return replacements[seq]
        
        # Split the sequence on '&' and convert each part
        parts = [convert_to_letters(part.strip()) for part in number_sequence.split('&')]
        replaced_sequence = ' & '.join(parts)
        
        return f"{prefix}{replaced_sequence}:"
    
    # Replace using the defined function
    return pattern.sub(replace_match, text)


def compute_score(input, gold_exp, pred_exp, metric, scorer):
    if metric == 'rouge':
        try:
            score = scorer.get_scores(pred_exp, gold_exp)[0]["rouge-l"]["f"]
        except:
            score = 0
    elif metric == 'bertscore':
        P, R, F1 = scorer.score([pred_exp], [gold_exp])
        score = float(F1.mean())
    elif metric == 'bartscore_cnn':
        score = scorer.score([gold_exp], [pred_exp], batch_size=4)
        score = float(score[0])
    elif metric == 'bartscore_cnn_para':
        score = scorer.score([gold_exp], [pred_exp], batch_size=4)
        score = float(score[0])
    elif metric == 'bleurt':
        score = scorer.score(references=gold_exp, candidates=pred_exp)
    elif metric == 'ctc_consist':
        score = scorer.score(doc=input, refs=[], hypo=pred_exp, aspect='consistency')
    elif metric == 'ctc_relev':
        try:
            score = scorer.score(doc=input, refs=[gold_exp], hypo=pred_exp, aspect='relevance')
        except:
            score = 0
    elif metric == 'ctc_presv':
        try:
            score = scorer.score(input_sent=gold_exp, hypo=pred_exp, aspect='preservation')
        except:
            score = 0

    return score

def score_func(input, gold_exp, output, metric, scorer):
    pred_exp = ""
    # replace numbers with letters
    gold_exp = replace_substring(gold_exp)
    try:
        output = re.sub("Answer:\n", "Answer: ", output)
        pred_exp = re.sub(r'Answer:[^\n]*\n', '', output)
        pred_exp = re.sub("Explanation: ", "Explanation:", pred_exp)
        pred_exp = re.sub('Explanation:', '', pred_exp)
        score = compute_score(input, gold_exp, pred_exp, metric, scorer)
    except:
        score = compute_score(input, gold_exp, pred_exp, metric, scorer)
    return gold_exp, pred_exp, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_type",
        default='R',
        type=str,
        help="YR: output answer and explanation; R: output explanation given correct answer; RY: chain-of-thought; Y: answer",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="model name: gpt-3.5-turbo, llama-2-70b-chat, gpt-3.5-turbo-instruct, palm2, gpt-4, llama3, medalpaca, meerkat",
    )
    parser.add_argument(
        "--data_name",
        default="medbullets",
        type=str,
        help="data name",
    )
    parser.add_argument(
        "--input_file",
        default="medbullets_op5.csv",
        type=str,
        help="input file: medbullets_op5.csv, jama.csv",
    )
    parser.add_argument(
        "--output_file",
        default="explanations",
        type=str,
        help="output file",
    )
    parser.add_argument(
        "--metric",
        default="rouge",
        type=str,
        help="rouge, bertscore, bartscore_cnn, bartscore_cnn_para, bleurt, ctc_consist, ctc_relev, ctc_presv",
    )
    parser.add_argument(
        "--few_shot",
        default=0,
        type=int,
        help="Number of few-shot examples.",
    )
    args = parser.parse_args()

    print('prompt_type: {}'.format(args.prompt_type))
    print('model_name: {}'.format(args.model_name))
    print('metric: {}'.format(args.metric))

    # read data
    current_path = os.path.dirname(os.path.abspath(__file__))
    filename = args.input_file.split('.')[0]
    in_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'.csv')
    header = pd.read_csv(in_file, nrows = 0).columns.to_list()
    df = pd.read_csv(in_file)
    examples = []
    for _, row in df.iterrows():
        examples.append(row)

    # set up scorer
    if args.metric == 'rouge':
        scorer = Rouge()
    elif args.metric == 'bertscore':
        scorer = BERTScorer(device='cuda', model_type='bert-base-uncased')
    elif args.metric == 'bartscore_cnn':
        scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    elif args.metric == 'bartscore_cnn_para':
        scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
        scorer.load(path='./scoreckpt/bart_score.pth')
    elif args.metric == 'bleurt':
        checkpoint = "BLEURT-20"
        scorer = bleurt_score.BleurtScorer(checkpoint)
    elif args.metric == 'ctc_consist':
        scorer = SummarizationScorer(align='E-roberta')
    elif args.metric == 'ctc_relev':
        scorer = SummarizationScorer(align='E-roberta')
    elif args.metric == 'ctc_presv':
        scorer = StyleTransferScorer(align='E-roberta-mnli')
    
    scores = []
    combs = []
    for example in tqdm(examples):
        input = example['input']
        gold_exp = example['explanation']
        output = example['output']
        gold_exp, pred_exp, score =  score_func(input, gold_exp, output, args.metric, scorer)
        scores.append(score)
        comb = [example[col] for col in header]
        comb.append(gold_exp)
        comb.append(pred_exp)
        comb.append(score)
        combs.append(comb)
    header.append('gold_explanation')
    header.append('pred_explanation')
    header.append('score')

    print('{}: {}'.format(args.metric, sum(scores) / len(scores)))
    output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type \
                               +'_exp'+'_'+args.metric+'.csv')
    if args.few_shot != 0:
        output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type+'_fs_'+\
                               str(args.few_shot)+'_exp'+'_'+args.metric+'.csv')
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combs)


if __name__ == "__main__":
    main()