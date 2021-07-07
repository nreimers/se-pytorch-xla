from transformers import AutoModel, AutoTokenizer
import sys

model = AutoModel.from_pretrained(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])


print(model(**tokenizer("Hello Word", return_tensors="pt")))