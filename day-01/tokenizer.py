# tokenizer.py: text -> token ids -> text, using the model's own SentencePiece file.
# Nothing here touches the weights. The vocab lives in tokenizer.model, a separate file.

import sentencepiece as spm
from huggingface_hub import hf_hub_download

# 1. Find tokenizer.model. It ships in the same HF repo as the weights, but it is
#    its own file: a list of 32000 pieces and the rules for splitting text into them.
path = hf_hub_download("Xenova/llama2.c-stories15M", "tokenizer.model")

# 2. Load it. sp is the object that knows how to encode and decode.
sp = spm.SentencePieceProcessor(model_file=path)
print("vocab size:", sp.get_piece_size())

# 3. Encode: string in, list of integers out. Each integer is an index into the vocab.
text = "Once upon a time"
ids = sp.encode(text)
print("text:  ", repr(text))
print("ids:   ", ids)

# 4. Look at what each id stands for. The leading '▁' marks a word start (a space).
print("pieces:", [sp.id_to_piece(i) for i in ids])

# 5. Decode: integers in, string out. Should be the original text.
back = sp.decode(ids)
print("back:  ", repr(back))

# 6. The M0 done condition.
assert back == text, "round trip failed"
print("round trip OK")

# 7. Two things that matter for M1.
#    Llama expects the sequence to start with <s> (id 1). encode() does not add it
#    unless asked.
ids_bos = sp.encode(text, add_bos=True)
print("with <s>:", ids_bos)
#    A word with and without a leading space is a different token.
print("'time' :", sp.encode("time"),  [sp.id_to_piece(i) for i in sp.encode("time")])
print("' time':", sp.encode(" time"), [sp.id_to_piece(i) for i in sp.encode(" time")])
