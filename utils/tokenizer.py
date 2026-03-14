from transformers import DistilBertTokenizer

_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_instruction(text, max_length=30):
    tokens = _tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    return tokens["input_ids"], tokens["attention_mask"]