from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Create a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
tokenizer.train(files = ["bribri-conllu-20240314-text.txt"], trainer = trainer)

# Tokenize the conllu file
with open("bribri-conllu-20240314-text.txt", "r") as f:
   with open("bribri-conllu-20240314-tokenized.txt", "w") as f2:
      for line in f:
         words = line.split()
         for word in words:
            f2.write("+".join(tokenizer.encode(word).tokens) + " ")
         f2.write("\n")

print(tokenizer.get_vocab_size())