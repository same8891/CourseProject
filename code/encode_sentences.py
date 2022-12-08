from transformers import BertTokenizer, BertModel
import torch
import csv
from tqdm import tqdm

cuda_available = torch.cuda.is_available()
device = torch.device("cpu") if not cuda_available else torch.device("cuda:0")

with open("data_labeled.csv", "r") as f:
    reader = csv.reader(f)
    reader = list(reader)
    sentences = [line[1] for line in reader[1:]]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
encoded_input = tokenizer(
    sentences, return_tensors='pt',
    padding=True, truncation=True, max_length=256
)
num = encoded_input["input_ids"].shape[0]
batch_size = 256

print(encoded_input)
results = []
with torch.no_grad():
    for i in tqdm(range((num-1)//batch_size + 1)):
        output = model(
            input_ids=encoded_input["input_ids"][
                i*batch_size:(i+1)*batch_size].to(device),
            attention_mask=encoded_input["attention_mask"][
                i*batch_size:(i+1)*batch_size].to(device))
        results.append(output.pooler_output)
results = torch.cat(results, 0)
torch.save([results, sentences], "encoded.pt")
