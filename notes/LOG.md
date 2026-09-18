# Log

One line per change that moved a number. Newest last.

| date | milestone | commit | what changed | number |
|---|---|---|---|---|
| 2026-09-15 | M0 | 3c2e8bb | export.py writes stories15M weights to flat fp32 file: 16-byte header (hidden, intermediate, vocab, layers) + 56 tensors in fixed order | 15,191,712 params, 60,766,864 bytes |
| 2026-09-16 | M0 | f159523 | run.c reads header, computes tensor sizes, loads body into one buffer, pointer per tensor | 7 spot checks match PyTorch to 6 decimals |
| 2026-09-17 | M0 | 10880e6 | SentencePiece round trip, "Once upon a time" -> [1, 9038, 2501, 263, 931] with bos | decode(encode(s)) == s. M0 done. |
