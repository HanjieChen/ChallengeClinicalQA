import os
import csv
import pandas as pd
import re
import argparse
from collections import Counter

def parse_answer(args, output):
    if args.prompt_type == 'Y':
        if 'gpt' in args.model_name:
            pred_idx = ""
            try:
                pred_idx = output.split(')')[0]
                pred_idx = re.sub("[^A-E]+", "", pred_idx)
            except:
                pass
        elif 'llama' in args.model_name:
            pred_idx = ""
            try:
                pred_idx = re.sub("[*\*]", "", output)
                pred_idx = re.sub("The answer is ", "", pred_idx)
                pred_idx = re.sub("The correct answer is ", "", pred_idx)
                pred_idx = re.sub("The correct answer is option ", "", pred_idx)
                pred_idx = re.sub("Answer: ", "Answer:", pred_idx)
                pred_idx = pred_idx.split('Answer:')[1].split()[0]
                pred_idx = re.sub("[^A-E]+", "", pred_idx)
            except:
                pass
        elif 'palm' in args.model_name:
            pred_idx = ""
            try:
                pred_idx = re.sub("Answer: ", "Answer:", output)
                if "Answer:" not in pred_idx:
                    pred_idx = re.sub('answer is' + r'[:\s\n]+', 'Answer:', pred_idx)
                pred_idx = pred_idx.split('Answer:')[1]
                pred_idx = pred_idx.split()[0]
                pred_idx = re.sub("[^A-E]+", "", pred_idx)
            except:
                pass
    elif args.prompt_type == 'RY':
        if args.ensemble_num>1:
            outputs = output.split(',')
            pred_idxs = []
            for out in outputs:
                if 'gpt' in args.model_name:
                    pred_idx = ""
                    try:
                        pred_idx = out.split(')')[0]
                        pred_idx = re.sub("[^A-E]+", "", pred_idx)
                    except:
                        pass
                elif 'llama' in args.model_name:
                    pred_idx = ""
                    try:
                        pred_idx = out.split(')')[0]
                        pred_idx = re.sub("[^A-E]+", "", pred_idx)
                    except:
                        pass
                elif 'palm' in args.model_name:
                    pred_idx = ""
                    try:
                        output = re.sub("[*\*]", "", out)
                        pred_idx = re.sub("Answer: ", "Answer:", output)
                        if "Answer:" not in pred_idx:
                            pred_idx = re.sub('answer is' + r'[:\s\n]+', 'Answer:', pred_idx)
                        pred_idx = pred_idx.split('Answer:')[1]
                        pred_idx = pred_idx.split()[0]
                        pred_idx = re.sub("[^A-E]+", "", pred_idx)
                    except:
                        pass
                pred_idxs.append(pred_idx)
            # majority voting
            counts = Counter(pred_idxs)
            majority_element, majority_count = counts.most_common(1)[0]
            if majority_count > len(pred_idxs) / 2:
                pred_idx = majority_element
            else:
                pred_idx = pred_idxs[0]
        else:
            if 'gpt' in args.model_name:
                pred_idx = ""
                try:
                    pred_idx = output.split(')')[0]
                    pred_idx = re.sub("[^A-E]+", "", pred_idx)
                except:
                    pass
            elif 'llama' in args.model_name:
                pred_idx = ""
                try:
                    pred_idx = output.split(')')[0]
                    pred_idx = re.sub("[^A-E]+", "", pred_idx)
                except:
                    pass
            elif 'palm' in args.model_name:
                pred_idx = ""
                try:
                    output = re.sub("[*\*]", "", output)
                    pred_idx = re.sub("Answer: ", "Answer:", output)
                    if "Answer:" not in pred_idx:
                        pred_idx = re.sub('answer is' + r'[:\s\n]+', 'Answer:', pred_idx)
                    pred_idx = pred_idx.split('Answer:')[1]
                    pred_idx = pred_idx.split()[0]
                    pred_idx = re.sub("[^A-E]+", "", pred_idx)
                except:
                    pass

    return pred_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_type",
        default='YR',
        type=str,
        help="YR: output answer and explanation; R: output explanation given correct answer; RY: chain-of-though; Y: answer",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="model name: gpt-3.5-turbo, llama-2-70b-chat, gpt-3.5-turbo-instruct, palm2",
    )
    parser.add_argument(
        "--data_name",
        default="xmedqa",
        type=str,
        help="data name",
    )
    parser.add_argument(
        "--input_file",
        default="medbullets_test.csv",
        type=str,
        help="input file: total_medbullets_op5.csv, total_medbullets_op5.csv",
    )
    parser.add_argument(
        "--output_file",
        default="explanations",
        type=str,
        help="output file",
    )
    parser.add_argument(
        "--use_chat",
        default=False,
        action="store_true",
        help="Use the chat version.",
    )
    parser.add_argument(
        "--few_shot",
        default=0,
        type=int,
        help="Number of few-shot examples.",
    )
    parser.add_argument(
        "--option_num",
        default=5,
        type=int,
        help="Number of options.",
    )
    parser.add_argument(
        "--ensemble_num",
        default=1,
        type=int,
        help="Number of sampled outputs.",
    )
    args = parser.parse_args()

    # read data
    current_path = os.path.dirname(os.path.abspath(__file__))
    filename = args.input_file.split('.')[0]
    in_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type)
    if args.ensemble_num > 1:
        in_file += '_ensemble_'+str(args.ensemble_num)                               
    if args.few_shot != 0:
        in_file += '_fs_'+str(args.few_shot)
    in_file += '.csv'
    header = pd.read_csv(in_file, nrows = 0).columns.to_list()
    df = pd.read_csv(in_file)
    examples = []
    for _, row in df.iterrows():
        examples.append(row)

    acc = []
    combs = []
    for example in examples:
        gold = example['answer_idx']
        if args.prompt_type == 'RY':
            output = example['output_cot']
        else:
            output = example['output']
        pred =  parse_answer(args, output)

        if gold == pred:
            acc.append(1)
        else:
            acc.append(0)

        comb = [example[col] for col in header]
        comb.append(pred)
        combs.append(comb)
           
    header.append('prediction')
    print('prediction accuracy: {}/{}={}'.format(sum(acc), len(acc), sum(acc) / len(acc)))
    output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type)
    if args.ensemble_num > 1:
        output_file += '_ensemble_'+str(args.ensemble_num)                               
    if args.few_shot != 0:
        output_file += '_fs_'+str(args.few_shot)
    output_file += '_pred.csv'
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combs)


if __name__ == "__main__":
    main()