import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


device = "cuda"


model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-base")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-base")
model.to(device)

src = "I'm feeling under the weather today"
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
print(tokenized_text)
model.eval()
summary_ids = model.generate(
                    tokenized_text,
                    max_length=128, 
                    num_beams=5,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True
                )
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(output)