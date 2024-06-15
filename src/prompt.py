import openai
import torch
import json
import os
import csv
import argparse
import random

import pandas as pd
import google.generativeai as palm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from together import Together

os.environ['OPENAI_API_KEY'] = ''
os.environ['TOGETHER_API_KEY'] = ''
palm.configure(api_key='')

llama3_name = 'meta-llama/Llama-3-70b-chat-hf'
llama2_name = 'meta-llama/Llama-2-70b-chat-hf'
meerkat_name = "dmis-lab/meerkat-7b-v1.0"
medalpaca_name = 'medalpaca/medalpaca-13b'
max_attempt = 3

num2l = {4: 'D', 5: 'E'}

# construct inputs
def input_format(args, examples):
    inputs = []
    fewshot_examples = []

    for idx, example in enumerate(examples):
        question = example['question']
        opa = example['opa']
        opb = example['opb']
        opc = example['opc']
        opd = example['opd']
        answer_idx = example['answer_idx']

        if args.prompt_type == 'Y':
            if args.option_num == 5:
                ope = example['ope']
                if 'palm' in args.model_name:
                    inputs.append(
                        f"The following is a multiple-choice question about medical knowledge.\n"\
                        f"QUESTION: {question}\n"\
                        f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}\n"\
                        f"Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                    )
                else:
                    inputs.append(
                        f"The following is a multiple-choice question about medical knowledge.\n"\
                        f"QUESTION: {question}\n"\
                        f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}\n"
                    )
            elif args.option_num == 4:
                if 'palm' in args.model_name:
                    inputs.append(
                        f"The following is a multiple-choice question about medical knowledge.\n"\
                        f"QUESTION: {question}\n"\
                        f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd}\n"\
                        f"Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                    )
                else:
                    inputs.append(
                        f"The following is a multiple-choice question about medical knowledge.\n"\
                        f"QUESTION: {question}\n"\
                        f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd}\n"
                    )
            else:
                raise ValueError(f'Invalid option_num: {args.option_num}')

            if args.few_shot != 0:
                idxes = [i for i in range(len(examples))]
                idxes.remove(idx)
                select_idx = random.sample(idxes, k=args.few_shot)
                fewshot_exps = []
                for id_cur in select_idx:
                    if args.option_num == 5:
                        if 'palm' in args.model_name:
                            case = f"The following is a multiple-choice question about medical knowledge.\n"\
                                   f"QUESTION: {examples[id_cur]['question']}\n"\
                                   f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']} (E) {examples[id_cur]['ope']}\n"\
                                   f"Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        else:
                            case = f"The following is a multiple-choice question about medical knowledge.\n"\
                                   f"QUESTION: {examples[id_cur]['question']}\n"\
                                   f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']} (E) {examples[id_cur]['ope']}\n"
                    elif args.option_num == 4:
                        if 'palm' in args.model_name:
                            case = f"The following is a multiple-choice question about medical knowledge.\n"\
                                   f"QUESTION: {examples[id_cur]['question']}\n"\
                                   f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']}\n"\
                                   f"Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
                        else:
                            case = f"The following is a multiple-choice question about medical knowledge.\n"\
                                   f"QUESTION: {examples[id_cur]['question']}\n"\
                                   f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']}\n"
                    else:
                        raise ValueError(f'Invalid option_num: {args.option_num}')
                    answer = examples[id_cur]['answer_idx']
                    fewshot_exps.append((case, answer))
                fewshot_examples.append(fewshot_exps)
        #  Two-step CoT
        elif args.prompt_type == 'RY':
            if args.option_num == 5:
                ope = example['ope']
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}\n"\
                )
            elif args.option_num == 4:
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd}\n"\
                )
            else:
                raise ValueError(f'Invalid option_num: {args.option_num}')
        elif args.prompt_type == 'R':
            if args.option_num == 5:
                ope = example['ope']
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}\n"\
                    f"ANSWER: ({answer_idx})\n"\
                    f"You are a medical expert that just answered the above question. "\
                    f"Please explain why ({answer_idx}) is the correct answer while the rest choices are incorrect. "\
                    f"You should explain each choice in detail.\n"\
                )
            elif args.option_num == 4:
                inputs.append(
                    f"QUESTION: {question}\n"\
                    f"ANSWER CHOICES: (A) {opa} (B) {opb} (C) {opc} (D) {opd} \n"\
                    f"ANSWER: ({answer_idx})\n"\
                    f"You are a medical expert that just answered the above question. "\
                    f"Please explain why ({answer_idx}) is the correct answer while the rest choices are incorrect. "\
                    f"You should explain each choice in detail.\n"
                )
            else:
                raise ValueError(f'Invalid option_num: {args.option_num}')
            if args.few_shot != 0:
                idxes = [i for i in range(len(examples))]
                idxes.remove(idx)
                select_idx = random.sample(idxes, k=args.few_shot)
                fewshot_exps = []
                for id_cur in select_idx:
                    if args.option_num == 5:
                        case = f"QUESTION: {examples[id_cur]['question']}\n"\
                               f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']} (E) {examples[id_cur]['ope']}\n"\
                               f"ANSWER: ({examples[id_cur]['answer_idx']})\n"\
                               f"You are a medical expert that just answered the above question. "\
                               f"Please explain why ({examples[id_cur]['answer_idx']}) is the correct answer while the rest choices are incorrect. "\
                               f"You should explain each choice in detail.\n"
                    elif args.option_num == 4:
                        case = f"QUESTION: {examples[id_cur]['question']}\n" \
                               f"ANSWER CHOICES: (A) {examples[id_cur]['opa']} (B) {examples[id_cur]['opb']} (C) {examples[id_cur]['opc']} (D) {examples[id_cur]['opd']}\n" \
                               f"ANSWER: ({examples[id_cur]['answer_idx']})\n" \
                               f"You are a medical expert that just answered the above question. " \
                               f"Please explain why ({examples[id_cur]['answer_idx']}) is the correct answer while the rest choices are incorrect. " \
                               f"You should explain each choice in detail.\n"
                    else:
                        raise ValueError(f'Invalid option_num: {args.option_num}')
                    explanation = examples[id_cur]['explanation']
                    fewshot_exps.append((case, explanation))
                fewshot_examples.append(fewshot_exps)
        else:
            raise ValueError(f'Invalid Error Type: {args.prompt_type}')
    
    return inputs, fewshot_examples

# prompt LLMs
def call_gpt(message, args, fewshot_exps=None):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = args.model_name
    output = ''
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []

    if args.prompt_type == 'RY':
        messages = [ {"role": "system", "content":
                    "You are a helpful assistant that good at dealing with multiple-choice medical questions."} ]
        messages.append(
                {"role": "user", "content": message},
            )
        messages.append(
                {"role": "assistant", "content": "Let's think step by step and walk through all the choices in detail."},
            )
        # temperature Defaults to 1, top_p Defaults to 1
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for the first step of X->RY...')

        messages.pop()

        input_cot = "Let's think step by step and walk through all the choices in detail. " + output + f"\nTherefore, from (A) to ({num2l[args.option_num]}), the answer is ("
        messages.append(
                {"role": "assistant", "content": input_cot},
            )
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output_cot = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for the second step of X->RY...')

    elif args.prompt_type == 'Y':
        messages = [ {"role": "system", "content":
                "You are a helpful assistant that good at answering multiple-choice medical questions."} ]
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": exp[0]},
                )
                messages.append(
                    {"role": "assistant", "content": "Answer:("+exp[1]+")"},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": message},
        )
        messages.append(
            {"role": "assistant", "content": "Answer:("},
        )
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for X->Y prompt...')

    elif args.prompt_type == 'R':
        messages = [{"role": "system", "content":
                "You are a helpful assistant that good at explaining the answer of multiple-choice medical questions."}]
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": exp[0]},
                )
                messages.append(
                    {"role": "assistant", "content": exp[1]},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": message},
        )
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for XY*->R prompt...')

    return output, input_cot, output_cot, fewshot_exp_message

def call_llama(message, args, fewshot_exps=None):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    if 'llama2' in args.model_name:
        model_name = llama2_name
    elif 'llama3' in args.model_name:
        model_name = llama3_name
    else:
        raise ValueError(f'Wrong LLaMa name: {args.model_name}')

    output = ''
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []

    if args.prompt_type == 'RY':
        messages = [ {"role": "system", "content":
                    "You are a helpful assistant that good at dealing with multiple-choice medical questions."} ]
        messages.append(
                {"role": "user", "content": message + "\nLet's differentiate using step by step reasoning like a medical expert."},
            )
        #  Let's think step by step and walk through all the choices in detail.
        ## Let's differentiate using step by step reasoning like a medical expert.
        # temperature Defaults to 0.7, top_p Defaults to 0.7
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, 
                    messages=messages,
                    temperature = 0.85,
                    top_p = 0.95,
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for the first step of X->RY...')

        messages.append(
            {"role": "assistant", "content": output},
        )
        messages.append(
            {"role": "user", "content": f"Therefore, from (A) to ({num2l[args.option_num]}), which one is the answer? Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"},
        )
        input_cot = "Let's differentiate using step by step reasoning like a medical expert.\n" + output + f"\nTherefore, from (A) to ({num2l[args.option_num]}), which one is the answer? Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"

        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, 
                    messages=messages,
                    temperature = 0.85,
                    top_p = 0.95,
                )
                output_cot = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for the second step of X->RY...')

    elif args.prompt_type == 'Y':
        messages = [{"role": "system", "content":
                "You are a helpful assistant that good at answering multiple-choice medical questions."}]
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": exp[0]},
                )
                messages.append(
                    {"role": "assistant", "content": "Answer:("+exp[1]+")"},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": message},
        )
        messages.append(
            {"role": "assistant", "content": "Answer:("},
        )
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for X->Y prompt...')

    elif args.prompt_type == 'R':
        messages = [{"role": "system", "content":
                "You are a helpful assistant that good at explaining the answer of multiple-choice medical questions."}]
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": exp[0]},
                )
                messages.append(
                    {"role": "assistant", "content": exp[1]},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": message},
        )
        attempts = 0
        while attempts < max_attempt:
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages
                )
                output = response.choices[0].message.content
                break
            except:
                attempts += 1
                print(f'{attempts} failed attempt for XY*->R prompt...')

    return output, input_cot, output_cot, fewshot_exp_message

def call_palm(message, args, fewshot_exps=None):
    examples = None
    messages = []
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []
    if args.prompt_type == 'RY':
        messages = []
        messages.append(
            {"author": "user", "content": message + "\nLet's think step by step and walk through all the choices in detail. "},
        )
        output = ''
        attempts = 0
        while attempts < max_attempt:
            try:
                response = palm.chat(
                    model='models/chat-bison-001',
                    examples=examples,
                    messages=messages,
                    temperature=0.8,
                    context="You are a helpful assistant that good at dealing with multiple-choice medical questions."
                )
                output = response.candidates[0]['content']
                break
            except:
                attempts += 1
                print(f'{attempts} Failed attempt for the first step of X->RY...')
        input_cot = "Let's think step by step and walk through all the choices in detail. " + output
        messages.append(
            {"author": "assitant", "content": input_cot},
        )
        messages.append(
            {"author": "user", "content": f"Therefore, among (A) through ({num2l[args.option_num]}), please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"},
        )

        output_cot = ''
        attempts = 0
        while attempts < max_attempt:
            try:
                response = palm.chat(
                    model='models/chat-bison-001',
                    examples=examples,
                    messages=messages,
                    temperature=0.8,
                    context="You are a helpful assistant that good at dealing with multiple-choice medical questions."
                )
                output_cot = response.candidates[0]['content']
                break
            except:
                attempts += 1
                print(f'{attempts} Failed attempt for the second step of X->RY...')

    elif args.prompt_type == 'Y':
        if fewshot_exps:
            examples = []
            for exp in fewshot_exps:
                examples.append(
                        {
                        "input": {"content": exp[0]},
                        "output": {"content": "Answer:("+exp[1]+")"}
                    }
                )
        fewshot_exp_message = examples
        messages.append(
            {"author": "user", "content": message},
        )

        output = ''
        attempts = 0
        while attempts < max_attempt:
            try:
                response = palm.chat(
                    model='models/chat-bison-001',
                    examples=examples,
                    messages=messages,
                    temperature=0.8,
                    context="You are a helpful assistant that good at answering multiple-choice medical questions."
                )
                output = response.candidates[0]['content']
                break
            except:
                attempts += 1
                print(f'{attempts} Failed attempt for X->Y...')

    elif args.prompt_type == 'R':
        if fewshot_exps:
            examples = []
            for exp in fewshot_exps:
                examples.append(
                        {
                        "input": {"content": exp[0]},
                        "output": {"content": exp[1]}
                    }
                )
        fewshot_exp_message = examples
        messages.append(
            {"author": "user", "content": message},
        )

        output = ''
        attempts = 0
        while attempts < max_attempt:
            try:
                response = palm.chat(
                    model='models/chat-bison-001',
                    examples=examples,
                    messages=messages,
                    temperature=0.8,
                    context="You are a helpful assistant that good at explaining multiple-choice medical questions."
                )
                output = response.candidates[0]['content']
                break
            except:
                attempts += 1
                print(f'{attempts} Failed attempt for XY*->R...')
    else:
        raise ValueError(f'Wrong prompt_type {args.prompt_type}')

    return output, input_cot, output_cot, fewshot_exp_message

def call_medalpaca(pl, message, args, fewshot_exps=None):
    input = ''
    output = ''
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []
    
    if args.prompt_type == 'RY':
        input += f'The following is a multiple-choice question about medical knowledge.\n\n<USER>: {message}\n<ASSISTANT>: Let\'s think step by step like a medical expert.'
        response = pl(input, max_new_tokens = 512)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]
        
        input_cot = response.split('Let\'s think step by step like a medical expert.')[0] + output + f"Therefore, from (A) to ({num2l[args.option_num]}), the answer is ("
        response = pl(input_cot, max_new_tokens = 64)
        output_cot = response[0]['generated_text'].replace(input_cot, '')
        if '<USER>' in output_cot:
            output_cot = output_cot.split('<USER>')[0]

    elif args.prompt_type == 'Y':
        if fewshot_exps:
            for exp in fewshot_exps:
                input += f'<USER>: {exp[0]}\n<ASSISTANT>: Answer:({exp[1]})\n'
            fewshot_exp_message = input
        input += f'<USER>: {message}\nASSISTANT: Answer:('
        response = pl(input, max_new_tokens = 64)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]

    elif args.prompt_type == 'R':
        input += "You are a medical expert that good at explaining multiple-choice medical questions.\n"
        if fewshot_exps:
            for exp in fewshot_exps:
                input += f'<USER>: {exp[0]}\n<ASSISTANT>: ({exp[1]})\n'
            fewshot_exp_message = input
        input += f'<USER>: {message}\nASSISTANT:'

        response = pl(input, max_new_tokens = 512)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]

    return input, output, input_cot, output_cot, fewshot_exp_message 

def call_medalpaca(pl, message, args, fewshot_exps=None):
    input = ''
    output = ''
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []
    
    if args.prompt_type == 'RY':
        input += f'The following is a multiple-choice question about medical knowledge.\n\n<USER>: {message}\n<ASSISTANT>: Let\'s think step by step like a medical expert.'
        response = pl(input, max_new_tokens = 512)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]
        
        input_cot = response.split('Let\'s think step by step like a medical expert.')[0] + output + f"Therefore, from (A) to ({num2l[args.option_num]}), the answer is ("
        response = pl(input_cot, max_new_tokens = 64)
        output_cot = response[0]['generated_text'].replace(input_cot, '')
        if '<USER>' in output_cot:
            output_cot = output_cot.split('<USER>')[0]

    elif args.prompt_type == 'Y':
        if fewshot_exps:
            for exp in fewshot_exps:
                input += f'<USER>: {exp[0]}\n<ASSISTANT>: Answer:({exp[1]})\n'
            fewshot_exp_message = input
        input += f'<USER>: {message}\nASSISTANT: Answer:('
        response = pl(input, max_new_tokens = 64)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]

    elif args.prompt_type == 'R':
        input += "You are a medical expert that good at explaining multiple-choice medical questions.\n"
        if fewshot_exps:
            for exp in fewshot_exps:
                input += f'<USER>: {exp[0]}\n<ASSISTANT>: ({exp[1]})\n'
            fewshot_exp_message = input
        input += f'<USER>: {message}\nASSISTANT:'

        response = pl(input, max_new_tokens = 512)[0]['generated_text']
        output = response.replace(input, '')
        if '<USER>' in output:
            output = output.split('<USER>')[0]

    return input, output, input_cot, output_cot, fewshot_exp_message 

def call_meerkat(tokenizer, model, message, args, fewshot_exps=None):

    input = ''
    output = ''
    input_cot = ''
    output_cot = ''
    fewshot_exp_message = []
    messages = []
    if args.prompt_type == 'RY':
        # messages = [ {"role": "system", "content":
        #             "You are a helpful assistant that good at dealing with multiple-choice medical questions."} ]
        messages.append(
                {"role": "user", "content": message + "\nLet's think step by step like a medical expert.\n"},
            )
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input = tokenizer.batch_decode(encodeds)[0]

        model_inputs = encodeds.to(args.device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=0.7, repetition_penalty=1.0)
        decoded = tokenizer.batch_decode(generated_ids)
        output = decoded[0].replace(input, '')

        messages.append(
            {"role": "assistant", "content": f'{output}\n'},
        )
        messages.append(
            {"role": "user", "content": f"Therefore, from (A) to ({num2l[args.option_num]}), which one is the answer? Please choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'\n"},
        )

        encodeds_cot = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_cot = tokenizer.batch_decode(encodeds_cot)[0]
        model_inputs_cot = encodeds_cot.to(args.device)
        generated_ids = model.generate(model_inputs_cot, max_new_tokens=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=0.7, repetition_penalty=1.0)
        decoded = tokenizer.batch_decode(generated_ids)
        output_cot = decoded[0].replace(input_cot, '')

    elif args.prompt_type == 'Y':
        # messages.append({"role": "system", "content":
        #         "You are a helpful assistant that good at answering multiple-choice medical questions. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id])\"\n"})
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": f'{exp[0]}\n'},
                )
                messages.append(
                    {"role": "assistant", "content": f"the answer is ({exp[1]})\n"},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": f'{message}\n'},
        )

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input = tokenizer.batch_decode(encodeds)[0]
        model_inputs = encodeds.to(args.device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=0.7, repetition_penalty=1.0)
        decoded = tokenizer.batch_decode(generated_ids)
        output = decoded[0].replace(input, '')

    elif args.prompt_type == 'R':
        # messagesa.append({"role": "system", "content":
        #         "You are a helpful assistant that good at explaining the answer of multiple-choice medical questions."})
        if fewshot_exps:
            for exp in fewshot_exps:
                messages.append(
                {"role": "user", "content": f'{exp[0]}\n'},
                )
                messages.append(
                    {"role": "assistant", "content": f'{exp[1]}\n'},
                )
            fewshot_exp_message = messages
        messages.append(
            {"role": "user", "content": f'{message}\n'},
        )
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input = tokenizer.batch_decode(encodeds)[0]
        model_inputs = encodeds.to(args.device)
        generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=0.7, repetition_penalty=1.0)
        decoded = tokenizer.batch_decode(generated_ids)
        output = decoded[0].replace(input, '')

    return input, output, input_cot, output_cot, fewshot_exp_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="medbullets_op5.csv",
        type=str,
        help="input file: medbullets_op5.csv, medbullets_op4.csv, jama.csv, medqa_op5.csv, medqa_op4.csv",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="model name: gpt-3.5-turbo, gpt-4, palm2, llama2, llama3, medalpaca, medalpaca, meerkat",
    )
    parser.add_argument(
        "--prompt_type",
        default='Y',
        type=str,
        help="Y: answer; RY: chain-of-thought; R: output explanation given correct answer; YR: output answer and explanation;",
    )
    parser.add_argument(
        "--device", default="gpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--data_name",
        default="medbullets",
        type=str,
        help="data name: medqa, medbullets, jama",
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
        help="Number of sampled outputs - Self-consistent Prompting.",
    )
    parser.add_argument(
        "--output_file",
        default="explanations",
        type=str,
        help="output file",
    )
    parser.add_argument(
        "--start_id",
        default=0,
        type=int,
        help="Start index",
    )
    args = parser.parse_args()

    # Setup device
    args.device = torch.device(
        f"cuda"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    if 'jama' in args.input_file:
        args.option_num = 4
    elif 'op4' in args.input_file:
        args.option_num = 4
    elif 'op5' in args.input_file:
        args.option_num = 5

    print('Input File: {}'.format(args.input_file))
    print('Model Name: {}'.format(args.model_name))
    print('Prompt Type: {}'.format(args.prompt_type))
    print('Device: {}'.format(args.device))
    print('Few Shot: {}'.format(args.few_shot))
    print('Option Num: {}'.format(args.option_num))
    print('Ensemble Num: {}'.format(args.ensemble_num))
    print('Start Index: {}'.format(args.start_id))

    # read data
    current_path = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(current_path, 'data', args.data_name, args.input_file)
    if 'csv' in in_file:
        df = pd.read_csv(in_file)
    elif 'json' in in_file:
        df = pd.read_json(in_file, orient='records')
    else:
        raise ValueError('Input file should be csv or json')
    if 'id' in df.columns.values:
        df = df.drop(columns='id')
    header = df.columns.values.tolist()

    examples = []
    for _, row in df.iterrows():
        examples.append(row)

    examples = examples[args.start_id:]
    inputs, fewshot_examples = input_format(args, examples)

    # write into file
    filename = args.input_file.split('.')[0]
    output_file = os.path.join(current_path, args.output_file, args.data_name+'_'+args.model_name+'_'+filename+'_'+args.prompt_type)
    if args.few_shot != 0:
        output_file += '_fs_' + str(args.few_shot)
    if args.ensemble_num > 1:
        output_file += '_ensemble_' + str(args.ensemble_num)
    output_file += '.csv'

    header.append('few_exp')
    header.append('input')
    header.append('output')
    header.append('input_cot')
    header.append('output_cot')

    pl = None
    tokenizer = None
    model = None
    if 'medalpaca' in args.model_name:
        pl = pipeline("text-generation", model=medalpaca_name, tokenizer=medalpaca_name, device=0, torch_dtype=torch.bfloat16, repetition_penalty=1.5, num_beams=2)
    elif 'meerkat' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(meerkat_name)
        model = AutoModelForCausalLM.from_pretrained(
            meerkat_name,
            torch_dtype=torch.bfloat16,  # You can choose to use this when there's not enough GPU memory available.
        )
        model.to(args.device)

    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        total_len = len(inputs)
        writer = csv.writer(f)
        writer.writerow(header)
        if len(fewshot_examples) != 0:
            for example, input, fewshot_exps in tqdm(zip(examples[:total_len], inputs, fewshot_examples), total=total_len):
                if 'gpt' in args.model_name:
                    output, input_cot, output_cot, few_exp_message = call_gpt(input, args, fewshot_exps)
                elif 'llama' in args.model_name:
                    output, input_cot, output_cot, few_exp_message = call_llama(input, args, fewshot_exps)
                elif 'palm' in args.model_name:
                    output, input_cot, output_cot, few_exp_message = call_palm(input, args, fewshot_exps)
                elif 'medalpaca' in args.model_name:
                    input, output, input_cot, output_cot, few_exp_message = call_medalpaca(pl, input, args, fewshot_exps)
                elif 'meerkat' in args.model_name:
                    input, output, input_cot, output_cot, few_exp_message = call_meerkat(tokenizer, model, input, args, fewshot_exps)
                else:
                    raise ValueError(f'Invaild Model Name {args.model_name}')
                comb = [example[col] for col in header[:-5]]
                comb.append(few_exp_message)
                comb.append(input)
                comb.append(output)
                comb.append(input_cot)
                comb.append(output_cot)
                writer.writerow(comb)
                f.flush()
        else:
            for example, input in tqdm(zip(examples[:total_len], inputs), total=total_len):
                if 'gpt' in args.model_name:
                    output, input_cot, output_cot, few_exp_message = call_gpt(input, args)
                elif 'llama' in args.model_name:
                    output, input_cot, output_cot, few_exp_message = call_llama(input, args)
                elif 'palm' in args.model_name:
                    output, input_cot,output_cot, few_exp_message = call_palm(input, args)
                elif 'medalpaca' in args.model_name:
                    input, output, input_cot, output_cot, few_exp_message = call_medalpaca(pl, input, args)
                elif 'meerkat' in args.model_name:
                    input, output, input_cot, output_cot, few_exp_message = call_meerkat(tokenizer, model, input, args)
                else:
                    raise ValueError(f'Invaild Model Name {args.model_name}')
                comb = [example[col] for col in header[:-5]]
                comb.append(few_exp_message)
                comb.append(input)
                comb.append(output)
                comb.append(input_cot)
                comb.append(output_cot)
                writer.writerow(comb)
                f.flush()

if __name__ == "__main__":
    main()