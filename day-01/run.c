#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* must match struct.pack("<4i", ...) in export.py: 4 x int32, no padding */
typedef struct {
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t vocab_size;
    int32_t num_hidden_layers;
} Header;

int main(void) {
    FILE *f = fopen("stories15M.bin", "rb");
    if (!f) { perror("fopen"); return 1; }

    Header h;
    if (fread(&h, sizeof(Header), 1, f) != 1) {
        fprintf(stderr, "short read on header\n");
        return 1;
    }
    printf("header: hidden=%d intermediate=%d vocab=%d layers=%d (sizeof Header = %zu)\n",
           h.hidden_size, h.intermediate_size, h.vocab_size, h.num_hidden_layers, sizeof(Header));

    /* float counts, all derived from the header */
    long hh_floats = (long)h.hidden_size * h.hidden_size;              /* q, k, v, o */
    long ih_floats = (long)h.intermediate_size * h.hidden_size;        /* gate, up, down */
    long h_floats  = h.hidden_size;                                    /* the three norms */
    long embed_floats = (long)h.vocab_size * h.hidden_size;            /* embed */

    long per_layer = 4 * hh_floats + 3 * ih_floats + 2 * h_floats;
    long total = embed_floats + h.num_hidden_layers * per_layer + h_floats;
    printf("per layer: %ld floats\n", per_layer);
    printf("total:     %ld floats  (expect 15191712)\n", total);
    printf("body bytes: %ld  (expect 60766848)\n", total * 4);

    /* TODO: malloc total floats, fread the body, check the return value      */

    float *weights = malloc(total * sizeof(float));
    if (!weights) { fprintf(stderr, "malloc failed\n"); return 1; }

    if (fread(weights, sizeof(float), total, f) != (size_t) total) {
        fprintf(stderr, "short read on body\n");
        return 1;
    }

    /* embed is the first tensor, so it starts at weights. row r starts r*hidden floats in. */
    float *embed = weights;
    printf("embed row 42, first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", embed[42 * h.hidden_size + i]);
    printf("\n");

    /* last float in the file = last element of the final norm */
    printf("last float (final norm[287]): %.6f\n", weights[total - 1]);

    /* running-pointer walk, in file order.
       p starts at the beginning and moves forward as each tensor is claimed.
       "claim" = write down where p is, then advance p by that tensor's count. */
    float *wq[6], *wk[6], *wv[6], *wo[6];        /* [hidden, hidden]       */
    float *w_gate[6], *w_up[6], *w_down[6];       /* [inter,hidden] / [hidden,inter] */
    float *w_in_norm[6], *w_post_norm[6];         /* [hidden]               */
    float *w_final_norm;                          /* [hidden]               */

    float *p = weights;
    embed = p;                  p += embed_floats;
    for (int L = 0; L < h.num_hidden_layers; L++) {
        wq[L]          = p;     p += hh_floats;
        wk[L]          = p;     p += hh_floats;
        wv[L]          = p;     p += hh_floats;
        wo[L]          = p;     p += hh_floats;
        w_gate[L]      = p;     p += ih_floats;
        w_up[L]        = p;     p += ih_floats;
        w_down[L]      = p;     p += ih_floats;
        w_in_norm[L]   = p;     p += h_floats;
        w_post_norm[L] = p;     p += h_floats;
    }
    w_final_norm = p;           p += h_floats;
    printf("walk ended at %ld floats (expect %ld)\n", (long)(p - weights), total);

    /* spot checks against export.py / PyTorch */
    printf("layer 0 q row 0, first 5:      ");
    for (int i = 0; i < 5; i++) printf("%.6f ", wq[0][i]);
    printf("\n");
    printf("layer 0 down row 0, first 5:   ");
    for (int i = 0; i < 5; i++) printf("%.6f ", w_down[0][i]);
    printf("\n");
    printf("layer 5 o row 0, first 5:      ");
    for (int i = 0; i < 5; i++) printf("%.6f ", wo[5][i]);
    printf("\n");
    printf("layer 5 post_norm, first 5:    ");
    for (int i = 0; i < 5; i++) printf("%.6f ", w_post_norm[5][i]);
    printf("\n");
    printf("final norm, first 5:           ");
    for (int i = 0; i < 5; i++) printf("%.6f ", w_final_norm[i]);
    printf("\n");


		


    fclose(f);
    free(weights);
    return 0;
}
