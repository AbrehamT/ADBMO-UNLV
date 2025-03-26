from datasets import Dataset
import numpy as np

class T5Dataset(Dataset):

    def __init__(
            self,
            texts,
            tokenizer,
            max_length = 512,
            corruption_rate = 0.15,
            mean_span_length = 3
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length

    def corrupt_text(self, text):
        tokens = text.strip().split()
        mask = self.generate_span_mask(len(tokens))
        input_tokens = []
        target_tokens = []
        sentinel = 0
        in_span = False

        for i, token in enumerate(tokens):
            if mask[i]:
                if not in_span:
                    input_tokens.append(f"<extra_id_{sentinel}>")
                    target_tokens.append(f"<extra_id_{sentinel}>")
                    sentinel += 1
                    in_span = True
                target_tokens.append(token)
            else:
                input_tokens.append(token)
                in_span = False
        
        target_tokens.append(f"<extra_id_{sentinel}")
        return " ".join(input_tokens), " ".join(target_tokens)


    def generate_span_mask(self, seq_len):
        num_tokens_to_mask = max(1, int(self.corruption_rate * seq_len))
        mask = np.zeros(seq_len, dtype=bool)
        num_masked = 0

        while num_masked < num_tokens_to_mask:
            span_start = np.random.randint(0, seq_len)
            span_length = max(1, np.random.poisson(self.mean_span_length))
            span_end = min(seq_len, span_start + span_length)

            if np.any(mask[span_start:span_end]):
                continue

            mask[span_start:span_end] = True
            num_masked += span_end - span_start
        
        return mask
    
    def __getitem__(self, idx):
        text = self.texts[id]
        tokens = text.strip().split()
        corrupted_input, target = self.corrupt_text(tokens)

        input_ids = self.tokenizer.encode(corrupted_input, truncation=True, max_length = self.max_length, return_tensors="pt").squeeze(0)
        target_ids = self.tokenizer.encode(target, truncation=True, max_length = self.max_length, return_tensors="pt").squeeze(0)

        return {"input_ids": input_ids, "labels": target_ids}
    
    def __len__(self):
        return len(self.texts)