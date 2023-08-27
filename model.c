#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>



typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;


typedef struct {
    float* token_embedding_table;

    float* rms_att_weight;
    float* rms_ffn_weight;

    float* wq;
    float* wk;
    float* wv;
    float* wo;


    float* w1;
    float* w2;
    float* w3;

    float* rms_final_weight;
    float* freq_cis_real;
    float* freq_cis_imag;

    float* wcls;
} TransformerWeights;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    ProbIndex *probindex; // buffer used in top-p sampling
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->probindex = calloc(p->vocab_size, sizeof(ProbIndex));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->probindex) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->probindex);
    free(s->key_cache);
    free(s->value_cache);
}


unsigned long long rng_seed;

// init the w from c, prt and shared_weights
void checkpoint_init_weights(TransformerWeights* w, Config* c, float* ptr, int shared_weights) {
    printf("n_heads=%i \n", c->n_heads);
    int head_size = c->dim / c->n_heads;
    w->token_embedding_table = ptr;
    ptr += c->vocab_size * c->dim;
    w->rms_att_weight = ptr;
    ptr += c->n_layers * c->dim;
    w->wq = ptr;
    ptr += c->n_layers * c->dim * (c->n_heads * head_size);
    w->wk = ptr;
    ptr += c->n_layers * c->dim * (c->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += c->n_layers * c->dim * (c->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += c->n_layers * (c->n_heads * head_size) * c->dim;
    w->rms_ffn_weight = ptr;
    ptr += c->n_layers * c->dim;
    w->w1 = ptr;
    ptr += c->n_layers * c->dim * c->hidden_dim;
    w->w2 = ptr;
    ptr += c->n_layers * c->hidden_dim * c->dim;
    w->w3 = ptr;
    ptr += c->n_layers * c->dim * c->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += c->dim;
    w->freq_cis_real = ptr;
    ptr += c->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += c->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;

}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

typedef struct {
    char *str;
    int id;
} TokenIndex;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {

    // sort vocabulary
    TokenIndex *sorted_vocab = malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = i;
    }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    char* str_buffer = malloc((max_token_length*2 +1 +2) * sizeof(char)); // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_lenght is 1)
    size_t str_len = 0;

    // add_dummy_prefix is true by default
    tokens[0] = str_lookup(" ", sorted_vocab, vocab_size);
    *n_tokens = 1; // the number of tokens

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
    free(sorted_vocab);
}



int main(int argc, char** argv) {

    char* checkpoint = "stories15M.pt";
    char* tokenizer = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    rng_seed = 0;
    int steps = 256;
    char* prompt = NULL;

    if(argc >= 2) checkpoint = argv[1];

    for(int i = 2; i < argc; i+=2 ) {
        if(argv[i][1] == 't') temperature = atof(argv[i+1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i+1]);
        else if (argv[i][1] == 's') rng_seed = atof(argv[i+1]);
        else if (argv[i][1] == 'n') steps = atof(argv[i+1]);
        else if (argv[i][1] == 'i') prompt = argv[i+1];
        else if (argv[i][1] == 'z') tokenizer = argv[i+1];
    }
    if (rng_seed == 0) rng_seed = (unsigned int) time(NULL);

Config config;
TransformerWeights weights;
int fd = 0;
float* data = NULL;
ssize_t file_size;
{
FILE* file = fopen(checkpoint, "rb");
if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); return 1; }

if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
int shared_weights = config.vocab_size > 0 ? 1 : 0;
printf("config.n_layers=%i\n", config.n_layers);
config.vocab_size = abs(config.vocab_size);
fseek(file, 0, SEEK_END);
file_size = ftell(file);
fclose(file);

fd = open(checkpoint, O_RDONLY);
if (fd == -1) { fprintf(stderr, "open failed!\n"); return 1; }
data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); return 1; }

float* weights_ptr = data + sizeof(Config)/sizeof(float);
checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
}

char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));

unsigned int max_token_length;

{
FILE* file = fopen(tokenizer, "rb");
if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer); return 1; }
if (fread(&max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
int len;
for (int i = 0; i < config.vocab_size; i++) {
    if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1;}

    if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }

    vocab[i] = (char *)malloc(len + 1);
    if (fread(vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
    vocab[i][len] = '\0'; // add the string terminating token
    }
fclose(file);
}


RunState state;
malloc_run_state(&state, &config);

int* prompt_tokens = NULL;
int num_prompt_tokens = 0;
if (prompt != NULL) {
    prompt_tokens = (int*)malloc((strlen(prompt)+1) * sizeof(int));
    bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
}


// main inference loop
//
long start = 0;
int next;
int token = 1;
int pos = 0;
/*
while (pos < steps) {





}

*/




}
