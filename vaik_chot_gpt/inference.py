import argparse
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration


def inference(input_model_dir_path, test_input_file_path, model_name='t5-small', input_prefix='input:', output_prefix='output:', max_length=512) :
    model = T5ForConditionalGeneration.from_pretrained(input_model_dir_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    with open(test_input_file_path, "r") as f:
        correct_num = 0
        total_num = 0
        for line in f.readlines():
            input_text, output_text = line.strip().split(output_prefix)
            input_text = input_text.replace(input_prefix, "").strip()
            input_tokenized = tokenizer.encode_plus(
                input_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            output = model.generate(
                input_ids=input_tokenized["input_ids"],
                attention_mask=input_tokenized["attention_mask"],
                max_length=max_length,
                num_return_sequences=1
            )
            output_text = output_text.strip()
            pred_output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            if output_text == pred_output_text:
                correct_num += 1
            total_num += 1
            print(f'input_text: {input_text[:10]}, output_text: {output_text[:10]}, pred_output_text: {pred_output_text[:10]}')
    print(f'ACC:{correct_num/total_num}')

def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--input_model_dir_path', type=str, default='~/.grave/lazy_baseline_investigator/mnist_model/model')
    parser.add_argument('--test_input_file_path', type=str,
                        default='~/.grave/lazy_baseline_investigator/dataset/mnist_valid.txt')
    parser.add_argument('--model_name', type=str, default='t5-small')
    args = parser.parse_args()

    args.input_model_dir_path = os.path.expanduser(args.input_model_dir_path)
    args.test_input_file_path = os.path.expanduser(args.test_input_file_path)
    inference(**args.__dict__)


if __name__ == '__main__':
    main()
