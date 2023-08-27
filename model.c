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



void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    //#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}


unsigned long long rng_seed;
unsigned int random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32() { // random float32 in [0,1)
    return (random_u32() >> 8) / 16777216.0f;
}


long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


int sample(float* probabilities, int n) {
    // sample index from probabilities (they must sum to 1!)
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = random_f32() * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}





void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (dim * p->n_kv_heads) / p->n_kv_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = &(w->token_embedding_table[dim*token]);
    memcpy(x, content_row, dim*sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {

        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);


        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);


        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }


        int layeroffset = l*p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + layeroffset + pos * kv_dim;
        float* value_cache_row = s->value_cache + layeroffset + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim *sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        int h;

        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++ ) {
                float* k = s->key_cache + layeroffset + t * kv_dim + (h/kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t ++) {
                float* v = s->value_cache + layeroffset + t * kv_dim + (h / kv_mul) * head_size;

                for (int i = 0; i < head_size; i ++ ) {
                    xb[i] += att[t] * v[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        for (int i=0; i < dim; i++) x[i] += s->xb2[i];

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);



}

int inference(int argc, char* argv) {

    char* checkpoint = "stories15M.bin";
    char* tokenizer = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    rng_seed = 0;
    int steps = 256;
    char* prompt = NULL;

    if(argc >= 2) checkpoint = argv;

    /*
    for(int i = 2; i < argc; i+=2 ) {
        if(argv[i][1] == 't') temperature = atof(argv[i+1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i+1]);
        else if (argv[i][1] == 's') rng_seed = atof(argv[i+1]);
        else if (argv[i][1] == 'n') steps = atof(argv[i+1]);
        else if (argv[i][1] == 'i') prompt = argv[i+1];
        else if (argv[i][1] == 'z') tokenizer = argv[i+1];
    }
    */
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


while (pos < steps) {
    transformer(token, pos, &config, &state, &weights);

    if(pos < num_prompt_tokens) {
        next = prompt_tokens[pos];
        continue;
    }

    if(temperature == 0.0f) {
        next = argmax(state.logits, config.vocab_size);
    }
    else {

        for (int q = 0; q < config.vocab_size; q++) {state.logits[q] /= temperature; }

        softmax(state.logits, config.vocab_size);
        if (topp <= 0 || topp >= 1) {
            next = sample(state.logits, config.vocab_size);
        }
        else {

            next = sample_topp(state.logits, config.vocab_size, topp, state.probindex);
        }

    }

    pos ++;


    if (next == 1) break;

            // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        unsigned char byte_val;
        if (sscanf(token_str, "<0x%02hhX>", &byte_val) == 1) {
            // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            if (isprint(byte_val) || isspace(byte_val)) {
                char byte_piece[2];
                byte_piece[0] = byte_val;
                byte_piece[1] = '\0';
                printf("%s", byte_piece);
            }
        } else {
            printf("%s", token_str);
        }
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // memory and file handles cleanup
    free_run_state(&state);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens != NULL) free(prompt_tokens);
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;


}
int main(int argc, char **argv) {
    inference(argc, argv[1]);
    return 0;
}
