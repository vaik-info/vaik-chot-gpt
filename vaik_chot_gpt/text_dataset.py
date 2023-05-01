import random
from tqdm import tqdm
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, input_prefix='input:', output_prefix='output:'):
        self.tokenizer = tokenizer
        self.inputs = []
        self.outputs = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                input_text, output_text = line.strip().split(output_prefix)
                input_text = input_text.replace(input_prefix, "").strip()
                output_text = output_text.strip()
                self.inputs.append(input_text)
                self.outputs.append(output_text)

        self.max_length = self.__set_max_length()

    def __len__(self):
        return len(self.inputs)

    def __set_max_length(self):
        self.max_length = 0
        for index in tqdm(range(len(self)), desc='set_max_length(self)'):
            input_text = self.inputs[index]
            input_tokenized = self.tokenizer.encode_plus(
                input_text,
                padding="longest",
                return_tensors="pt"
            )
            if input_tokenized.data['input_ids'].shape[-1] > self.max_length:
                self.max_length = input_tokenized.data['input_ids'].shape[-1]

    def __getitem__(self, index):
        input_text = self.inputs[index]
        output_text = self.outputs[index]

        input_tokenized = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        output_tokenized = self.tokenizer.encode_plus(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_tokenized["input_ids"].squeeze(),
            "attention_mask": input_tokenized["attention_mask"].squeeze(),
            "labels": output_tokenized["input_ids"].squeeze(),
        }
