import argparse
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from vaik_chot_gpt.text_dataset import TextDataset

def train(train_input_file_path, test_input_file_path, output_dir_path, model_name='t5-small', num_train_epochs=10, per_device_train_batch_size=4,
          learning_rate=3e-5, weight_decay=0.01):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    train_dataset = TextDataset(train_input_file_path, tokenizer)
    test_dataset = TextDataset(test_input_file_path, tokenizer)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_dir=os.path.join(output_dir_path, f'logs'),
        logging_steps=10,
        save_steps=1000,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)

    trainer.train()

    trainer.save_model(os.path.join(output_dir_path, f'model'))


def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--train_input_file_path', type=str,
                        default='~/.vaik-mnist-text-dataset/dataset/mnist_train.txt')
    parser.add_argument('--test_input_file_path', type=str,
                        default='~/.vaik-mnist-text-dataset/dataset/mnist_valid.txt')
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_chot_gpt/mnist_model/')
    parser.add_argument('--model_name', type=str, default='t5-small')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    args = parser.parse_args()

    args.output_dir_path = os.path.expanduser(args.output_dir_path)
    args.train_input_file_path = os.path.expanduser(args.train_input_file_path)
    args.test_input_file_path = os.path.expanduser(args.test_input_file_path)
    train(**args.__dict__)


if __name__ == '__main__':
    main()
