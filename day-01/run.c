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


		


    fclose(f);
    free(weights);
    return 0;
}
