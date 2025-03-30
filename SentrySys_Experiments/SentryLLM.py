import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

class LLMInterface:
    def __init__(
            self,
            model_name: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token = os.getenv('HF_TOKEN'))

    def preprocesser_inference(
            self,
            input: list[str],
            pad: bool,
            truncate: bool,
            max_length: int,
            return_tensor: str = "pt"
    ):
        tokenized_text = []
        for text in input:
            tk_txt = self.tokenizer(text, max_length=max_length,padding=pad, truncation=truncate, return_tensors=return_tensor).to(self.device)
            tokenized_text.append(tk_txt)
        return tokenized_text
    
class classifierInterface(LLMInterface):
    def __init__(self, model_name: str, id2label: dict, label2id: dict, num_labels=2, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, device)
        load_dotenv()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            use_auth_token = os.getenv('HF_ACCESS_TOKEN')
        )
        # print(os.getenv('HF_ACCESS_TOKEN'))

    def inference_classify(
            self,
            inputs: list[str],
            pad: bool,
            truncate: bool,
            max_length: int,
            return_tensor: str = "pt"
    ):
        classes = []
        inputs = self.preprocesser_inference(inputs, pad, truncate, max_length, return_tensor)
        for input in inputs:
            with torch.no_grad():
                logits = self.model(**input).logits
            predicted_class = logits.argmax().item()
            classes.append(self.model.config.id2label[predicted_class])
        return classes
    
class extractorInterface(LLMInterface):
    def __init__(self, model_name: str, device: str == 'cuda' if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, device)
        from transformers import AutoModelForTokenClassification
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            use_auth_token = os.getenv('HF_ACCESS_TOKEN')
        )
    
    def inference_extract(
            self,
            input: str,
            pad: bool,
            truncate: bool,
            max_length: bool,
            return_tensor: str = "pt"
    ):
        input = self.preprocesser_inference(input, pad, truncate, max_length, return_tensor)
        with torch.no_grad():
            logits = self.model(**input).logits
        
        # TODO: RETURN THE APPROPRIATE VALUE, NOT LOGITS, FROM HERE