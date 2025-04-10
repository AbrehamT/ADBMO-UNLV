# from torch.utils.data import Dataset
# import torch.nn.utils
# import numpy as np

# class T5Dataset(Dataset):
#     def __init__(
#         self,
#         texts,
#         tokenizer,
#         max_length=512,
#         corruption_rate=0.15,
#         mean_span_length=3
#     ):
#         # super().__init__()
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.corruption_rate = corruption_rate
#         self.mean_span_length = mean_span_length

#     def corrupt_text(self, tokens):
#         # tokens = text.strip().split()
#         mask = self.generate_span_mask(len(tokens))
#         input_tokens = []
#         target_tokens = []
#         sentinel = 0
#         in_span = False
#         for i, token in enumerate(tokens):
#             if mask[i]:
#                 if not in_span:
#                     input_tokens.append(f"<extra_id_{sentinel}>")
#                     target_tokens.append(f"<extra_id_{sentinel}>")
#                     sentinel += 1
#                     in_span = True
#                 target_tokens.append(token)
#             else:
#                 input_tokens.append(token)
#                 in_span = False

#         if in_span:
#             target_tokens.append(f"<extra_id_{sentinel+1}>")
#         else:
#             input_tokens.append(f"<extra_id_{sentinel}>")
#             target_tokens.append(f"<extra_id_{sentinel}>")
        
#         return " ".join(input_tokens), " ".join(target_tokens)

#     def generate_span_mask(self, seq_len):
#         num_tokens_to_mask = max(1, int(self.corruption_rate * seq_len))
#         mask = np.zeros(seq_len, dtype=bool)
#         num_masked = 0
#         while num_masked < num_tokens_to_mask:
#             span_start = np.random.randint(0, seq_len)
#             span_length = max(1, np.random.poisson(self.mean_span_length))
#             span_end = min(seq_len, span_start + span_length)
#             if np.any(mask[span_start:span_end]):
#                 continue
#             mask[span_start:span_end] = True
#             num_masked += span_end - span_start
#         return mask

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         tokens = text.strip().split()
        
#         # Handle empty text case first
#         if len(tokens) == 0:
#             input_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
#             target_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
#             return {"input_ids": input_ids, "labels": target_ids}
        
#         try:
#             corrupted_input, target = self.corrupt_text(tokens)
#             input_ids = self.tokenizer.encode(
#                 corrupted_input, 
#                 truncation=True, 
#                 max_length=self.max_length, 
#                 return_tensors="pt"
#             ).squeeze(0)
            
#             target_ids = self.tokenizer.encode(
#                 target, 
#                 truncation=True, 
#                 max_length=self.max_length, 
#                 return_tensors="pt"
#             ).squeeze(0)
            
#             return {"input_ids": input_ids, "labels": target_ids}
#         except Exception as e:
#             print(f"Error processing item {idx}: {e}")
#             # Fallback to empty tokens
#             input_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
#             target_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
#             return {"input_ids": input_ids, "labels": target_ids}

#     def __len__(self):
#         return len(self.texts)


# def collator(batch, tokenizer):
#     try:
#         input_ids = [item['input_ids'] for item in batch]
#         labels = [item['labels'] for item in batch]
#         input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
#     except:
#         print(batch)
#     return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(tokenizer.pad_token_id)}


from torch.utils.data import Dataset
import torch.nn.utils
import numpy as np


class T5Dataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        annotations=None,  # NEW: list of lists of spans [(start, end), ...] per text
        max_length=512,
        corruption_rate=0.15,
        mean_span_length=3
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.annotations = annotations if annotations is not None else [None] * len(texts)
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length

    def corrupt_text(self, tokens, annotation_spans=None):
        mask = self.generate_span_mask(len(tokens), annotation_spans)

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

        if in_span:
            target_tokens.append(f"<extra_id_{sentinel+1}>")
        else:
            input_tokens.append(f"<extra_id_{sentinel}>")
            target_tokens.append(f"<extra_id_{sentinel}>")

        return " ".join(input_tokens), " ".join(target_tokens)

    def generate_span_mask(self, seq_len, annotation_spans=None):
        num_tokens_to_mask = max(1, int(self.corruption_rate * seq_len))
        mask = np.zeros(seq_len, dtype=bool)
        num_masked = 0

        # Force mask annotated spans
        if annotation_spans:
            for start, end in annotation_spans:
                mask[start:end] = True
            num_masked = mask.sum()

        # Continue masking random spans until corruption rate is satisfied
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
        text = self.texts[idx]
        tokens = text.strip().split()
        annotation_spans = self.annotations[idx]

        if len(tokens) == 0:
            input_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
            target_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
            return {"input_ids": input_ids, "labels": target_ids}

        try:
            corrupted_input, target = self.corrupt_text(tokens, annotation_spans)
            input_ids = self.tokenizer.encode(
                corrupted_input,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).squeeze(0)

            target_ids = self.tokenizer.encode(
                target,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).squeeze(0)

            return {"input_ids": input_ids, "labels": target_ids}
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            input_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
            target_ids = self.tokenizer.encode("", return_tensors="pt").squeeze(0)
            return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self):
        return len(self.texts)


def collator(batch, tokenizer):
    try:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    except:
        print(batch)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(tokenizer.pad_token_id)}
