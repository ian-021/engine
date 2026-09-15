# Confusions worked through on 2026-09-15 (session 1, M0)

Each line: what I was confused about, then the resolution in my own words.
Copy the ones that matter into 01-labnot.yaml under understood_today / QA.

1. **What is a token ID?**
   Thought it might be a neuron or a parameter. It is just an integer index into the vocabulary
   list (0 to 31999 for this model). Token 42 is whatever string sits at position 42.

2. **What does embed_tokens(42) do?**
   It returns row 42 of a 32000 x 288 lookup table. No math, just a lookup. The 288 numbers in
   that row are learned weights.

3. **Why flatten a matrix to a 1D file?**
   Thought flattening was a conversion. It isn't: memory and disk are already one line of bytes.
   The grid is a note about how to read the line, not a different kind of storage.

4. **How does C know the shape if the file is flat?**
   It doesn't. The header tells it. Shape lives in the header and in the C code; if they
   disagree you get silent garbage, no error.

5. **Row 1, column 2 of [[1,2,3],[4,5,6]] is 2, not 5?**
   Zero-based counting. Row 0 is the top row. "Row 1, column 2" means second row, third column = 6.
   Position formula row*cols + col only works with zero-based indices.

6. **What is a tensor? Is it a matrix?**
   A box of numbers with a shape written on the lid. Matrix = 2D tensor. A list = 1D. A single
   number = 0D. "Array" is the same idea in other languages.

7. **Can view() take more than 2 dimensions? What are they called?**
   Any number of dims, as long as they multiply to the element count. Beyond rows/columns they
   have no fixed names, just dim 0, dim 1, dim 2. In practice named by meaning (batch, seq, heads).

8. **Finding element (a, b, c) in a 3D tensor [A, B, C].**
   Picture a stack of pages. position = a*(B*C) + b*C + c: skip whole pages, skip rows, walk
   along the row. This is counting, not linear algebra.

9. **Confused shape with index.**
   Shape [2,2,3] says how many (2 pages, 2 rows, 3 cols). Index (1,0,2) says which one.
   Same-looking triples, different jobs.

10. **"50 tensors" vs "15 million numbers": is there an equation?**
    No. 50 is a count of boxes, 15M is the total numbers across all boxes. Like 50 books, 15M pages.

11. **Is the embedding table separate from the tensors? Is it arbitrary?**
    It is one of the 56 tensors, the first one. Not chosen by us; learned during training and
    shipped with the model. 2D because it needs one row per vocab word and one column per hidden
    unit, nothing was removed to make it 2D.

12. **"The embedding table is a tensor, the model has 50 tensors, so 50 embedding tables?"**
    Category vs member. Dog is an animal, farm has 50 animals, not 50 dogs. Exactly one is the
    embedding table.

13. **What does the model DO with the tensors?**
    Tensors are ingredients (data). The forward pass is the recipe (code in run.c). HF ships both;
    I keep their ingredients and rewrite their recipe, then check my logits match theirs.

14. **What is the forward pass?**
    A function: token ID(s) in, 32000 scores out (one per vocab word). Reads every tensor once.
    Generation = forward pass, argmax, feed the result back, repeat.

15. **What is model.model.embed_tokens.weight?**
    Nested containers: wrapper (LlamaForCausalLM) -> transformer body (.model) -> embedding
    layer (.embed_tokens) -> its tensor (.weight). The doubled "model" was a naming coincidence.

16. **How do I index emb[42, 3, :5]?**
    One slot per dimension. emb is 2D so exactly two slots. emb[42, 3] = one cell,
    emb[42, :5] = row 42 first five, emb[42, 3:8] = row 42 columns 3-7. Don't subtract 1.

17. **What is grad_fn=<SelectBackward0>?**
    Training bookkeeping (autograd record). Irrelevant here. Use torch.no_grad() or .item().

18. **Why only 6 layers? What are Q, K, V, O, gate, up, down?**
    Toy model; Llama 2 7B has 32 layers with the same 9 boxes per layer. Q/K/V/O are the four
    attention grids, gate/up/down the feed-forward grids, two norms rescale. Meaning comes in M1.

19. **How do I know q0 is 2D?**
    q0.shape prints torch.Size([288, 288]); count the entries in the brackets.

## Facts established today

- Model: Xenova/llama2.c-stories15M (HF format). 56 tensors, 15,191,712 params.
- Largest tensor: embed_tokens [32000, 288] = 9,216,000 params (61% of the model).
- tie_word_embeddings is true, so no lm_head tensor. Hit the tied case earlier than the assignment expected.
- Linear weights stored [out, in]; down_proj is [288, 768].
- Machine: Apple M1, ~68 GB/s memory bandwidth.

## Open questions to carry forward

- What exactly do Q, K, V do in attention? (M1)
- What order should the 56 tensors go in the export file, and what goes in the header?
- What does row 42 of the embedding table look like when read back from C?
