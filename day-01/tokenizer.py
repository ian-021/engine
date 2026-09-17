import sentencepiece as spm
from huggingface_hub import hf_hub_download

path = hf_hub_download("Xenova/llama2.c-stories15M", "tokenizer.model")

sp = spm.SentencePieceProcessor(model_file=path)
print("vocab size:", sp.get_piece_size())

#encode
text = "To be or not"
ids = sp.encode(text)
print("text:  ", repr(text))
print("ids:   ", ids)

#check ids
print("pieces:", [sp.id_to_piece(i) for i in ids])

#decode
back = sp.decode(ids)
print("back:  ", repr(back))

# 7. Two things that matter for M1.
#    Llama expects the sequence to start with <s> (id 1). encode() does not add it
#    unless asked.
ids_bos = sp.encode(text, add_bos=True)
print("with <s>:", ids_bos)
#    A word with and without a leading space is a different token.
print("'time' :", sp.encode("time"),  [sp.id_to_piece(i) for i in sp.encode("time")])
print("' time':", sp.encode(" time"), [sp.id_to_piece(i) for i in sp.encode(" time")])
