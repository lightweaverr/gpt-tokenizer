
import os
import time
from bpe import BasicTokenizerBPE, TokenizerBPE

# open some text and train a vocab of 512 tokens
text = open("test/taylorswift.txt", "r", encoding="utf-8").read()
print(len(text))
# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()


    # construct the Tokenizer object and kick off verbose training
tokenizer = TokenizerBPE()
tokenizer.train(text, 512, verbose=True)
  
prefix = os.path.join("models", '_basic')
tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")