/***********************************************************************
 * File:    transformer_lattice.c
 * Author:  Dr. Q (Fin Pandora) – riffing on Gemini's original demo
 * Purpose: Pedagogical "Transformer Lattice" in C
 *          – Multi-head attention
 *          – Feed-forward
 *          – Residual connections & layer norm (simplified)
 *          – Positional encoding
 *
 * Build:   gcc transformer_lattice.c -lm -o lattice_demo
 **********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

/* ---------- 1. Basic Tensor ---------- */
typedef struct { int r,c; float *d; } Tensor;

Tensor new_tensor(int r,int c){
    Tensor t={r,c,(float*)calloc(r*c,sizeof(float))};
    if(!t.d){fprintf(stderr,"OOM\n");exit(1);}
    return t;
}
void free_tensor(Tensor *t){ free(t->d); t->d=NULL; }
float *T(Tensor *t,int i,int j){ return &t->d[i*t->c+j]; }

/* tiny helpers */
void rand_fill(Tensor *t,float scale){
    for(int i=0;i<t->r*t->c;i++) t->d[i]=(float)rand()/RAND_MAX*2*scale-scale;
}
void copy_tensor(Tensor *src,Tensor *dst){
    memcpy(dst->d,src->d,sizeof(float)*src->r*src->c);
}
void print_tensor(Tensor *t,const char *name){
    printf("\n--- %s (%dx%d) ---\n",name,t->r,t->c);
    for(int i=0;i<t->r;i++){
        for(int j=0;j<t->c;j++) printf("%8.4f ",*T(t,i,j));
        puts("");
    }
}

/* matmul: C = A @ B (naïve) */
void matmul(Tensor *A,Tensor *B,Tensor *C){
    if(A->c!=B->r||C->r!=A->r||C->c!=B->c){fprintf(stderr,"matmul shape!\n");exit(1);}
    for(int i=0;i<C->r;i++)
        for(int j=0;j<C->c;j++){
            float s=0;
            for(int k=0;k<A->c;k++) s+=(*T(A,i,k))*(*T(B,k,j));
            *T(C,i,j)=s;
        }
}

/* softmax rows in-place */
void softmax_rows(Tensor *X){
    for(int i=0;i<X->r;i++){
        /* max for stability */
        float maxv=*T(X,i,0);
        for(int j=1;j<X->c;j++) maxv=MAX(maxv,*T(X,i,j));
        /* exp & sum */
        float s=0;
        for(int j=0;j<X->c;j++){ float e=expf(*T(X,i,j)-maxv); *T(X,i,j)=e; s+=e;}
        /* normalize */
        for(int j=0;j<X->c;j++) *T(X,i,j)/=s;
    }
}

/* add src into dst (same shape) */
void add_(Tensor *dst,Tensor *src){
    for(int i=0;i<dst->r*dst->c;i++) dst->d[i]+=src->d[i];
}

/* simple layer norm (per row) */
void layer_norm(Tensor *X){
    for(int i=0;i<X->r;i++){
        float mu=0, var=0;
        for(int j=0;j<X->c;j++) mu+=*T(X,i,j);
        mu/=X->c;
        for(int j=0;j<X->c;j++){ float v=*T(X,i,j)-mu; var+=v*v; }
        var/=X->c; float denom=1.0f/sqrtf(var+1e-6f);
        for(int j=0;j<X->c;j++) *T(X,i,j)=(*T(X,i,j)-mu)*denom;
    }
}

/* ---------- 2. Multi-Head Attention ---------- */
typedef struct{
    Tensor Wq,Wk,Wv;
} Head;

typedef struct{
    int n_heads;
    int d_model;
    int d_head;
    Head *heads; /* array of size n_heads */
    Tensor Wo; /* output projection */
} MHA;

/* init one head */
Head new_head(int d_model,int d_head){
    Head h={new_tensor(d_model,d_head),
            new_tensor(d_model,d_head),
            new_tensor(d_model,d_head)};
    rand_fill(&h.Wq,0.5f);
    rand_fill(&h.Wk,0.5f);
    rand_fill(&h.Wv,0.5f);
    return h;
}

/* init MHA */
MHA new_mha(int n_heads,int d_model){
    MHA m; m.n_heads=n_heads; m.d_model=d_model;
    m.d_head=d_model/n_heads;
    m.heads=(Head*)malloc(sizeof(Head)*n_heads);
    m.Wo=new_tensor(d_model,d_model); /* output projection */
    rand_fill(&m.Wo,0.5f);
    for(int i=0;i<n_heads;i++) m.heads[i]=new_head(d_model,m.d_head);
    return m;
}

/* forward one head: out = softmax(QK^T/sqrt(d))*V */
void head_forward(Head *h,Tensor *X,Tensor *out){
    int N=X->r, d=h->Wq.c;
    Tensor Q=new_tensor(N,d),K=new_tensor(N,d),V=new_tensor(N,d);
    matmul(X,&h->Wq,&Q);
    matmul(X,&h->Wk,&K);
    matmul(X,&h->Wv,&V);

    Tensor Kt=new_tensor(d,N);
    for(int i=0;i<d;i++)for(int j=0;j<N;j++) *T(&Kt,i,j)=*T(&K,j,i);

    Tensor scores=new_tensor(N,N);
    matmul(&Q,&Kt,&scores);
    float sf=1.0f/sqrtf((float)d);
    for(int i=0;i<N*N;i++) scores.d[i]*=sf;
    softmax_rows(&scores);

    matmul(&scores,&V,out); /* Output directly to out (d_head dimension) */

    /* cleanup */
    free_tensor(&Q);free_tensor(&K);free_tensor(&V);
    free_tensor(&Kt);free_tensor(&scores);
}

/* MHA forward (concatenate head outputs) */
void mha_forward(MHA *m,Tensor *X,Tensor *Y){
    int N=X->r, d=m->d_model, dh=m->d_head;
    Tensor tmp=new_tensor(N,d); /* accumulate concat result */
    for(int h=0;h<m->n_heads;h++){
        Tensor h_out=new_tensor(N,dh); /* Each head outputs d_head dimensions */
        head_forward(&m->heads[h],X,&h_out);
        /* write into proper slice */
        for(int i=0;i<N;i++)
            for(int j=0;j<dh;j++)
                *T(&tmp,i,h*dh+j) = *T(&h_out,i,j);
        free_tensor(&h_out);
    }
    /* apply output projection */
    matmul(&tmp,&m->Wo,Y);
    free_tensor(&tmp);
}

/* ---------- 3. Feed-Forward ---------- */
typedef struct{
    Tensor W1,W2; /* d_model x d_ff , d_ff x d_model */
} FFN;

FFN new_ffn(int d_model,int d_ff){
    FFN f={new_tensor(d_model,d_ff),new_tensor(d_ff,d_model)};
    rand_fill(&f.W1,0.5f); rand_fill(&f.W2,0.5f);
    return f;
}
void relu(Tensor *X){ for(int i=0;i<X->r*X->c;i++) X->d[i]=MAX(0,X->d[i]); }

void ffn_forward(FFN *f,Tensor *X,Tensor *Y){
    int N=X->r, d_model=f->W1.r, d_ff=f->W1.c;
    Tensor h=new_tensor(N,d_ff);
    matmul(X,&f->W1,&h);
    relu(&h);
    matmul(&h,&f->W2,Y);
    free_tensor(&h);
}

/* ---------- 4. Transformer Block ---------- */
typedef struct{
    MHA mha;
    FFN ffn;
} Block;

Block new_block(int n_heads,int d_model,int d_ff){
    Block b={new_mha(n_heads,d_model),new_ffn(d_model,d_ff)};
    return b;
}

void block_forward(Block *b,Tensor *X){
    int N=X->r,d=X->c;
    /* Multi-head + residual + norm */
    Tensor mha_out=new_tensor(N,d);
    mha_forward(&b->mha,X,&mha_out);
    add_(&mha_out,X);
    layer_norm(&mha_out);

    /* Feed-Forward + residual + norm */
    Tensor ffn_out=new_tensor(N,d);
    ffn_forward(&b->ffn,&mha_out,&ffn_out);
    add_(&ffn_out,&mha_out);
    layer_norm(&ffn_out);

    copy_tensor(&ffn_out,X);

    free_tensor(&mha_out);free_tensor(&ffn_out);
}

/* ---------- 5. Positional Encoding ---------- */
void add_positional_encoding(Tensor *X){
    int N=X->r,d=X->c;
    for(int pos=0;pos<N;pos++)
        for(int i=0;i<d;i++){
            float angle=pos/powf(10000,(2*(i/2))/(float)d);
            if(i%2==0) *T(X,pos,i)+=sinf(angle);
            else       *T(X,pos,i)+=cosf(angle);
        }
}

/* ---------- 6. Lattice (stack of blocks) ---------- */
typedef struct{
    int n_layers;
    Block *layers;
} Lattice;

Lattice new_lattice(int n_layers,int n_heads,int d_model,int d_ff){
    Lattice L; L.n_layers=n_layers;
    L.layers=(Block*)malloc(sizeof(Block)*n_layers);
    for(int i=0;i<n_layers;i++) L.layers[i]=new_block(n_heads,d_model,d_ff);
    return L;
}

void lattice_forward(Lattice *L,Tensor *X){
    add_positional_encoding(X);
    for(int l=0;l<L->n_layers;l++) block_forward(&L->layers[l],X);
}

/* ---------- 7. Demo main ---------- */
int main(){
    srand(42);
    int seq_len=4;
    int d_model=8;
    int n_heads=2;
    int d_ff=16;
    int n_layers=2;

    printf("== Mini-Transformer Lattice Demo ==\n");
    printf("Seq=%d  d_model=%d  heads=%d  layers=%d\n\n",
            seq_len,d_model,n_heads,n_layers);

    Tensor X=new_tensor(seq_len,d_model);
    rand_fill(&X,1.0f);
    print_tensor(&X,"Input Embeddings");

    Lattice net=new_lattice(n_layers,n_heads,d_model,d_ff);
    lattice_forward(&net,&X);

    print_tensor(&X,"Output after Lattice");

    puts("\n(Each row is now a context-rich representation of its token.)");

    /* cleanup omitted for brevity in a demo */

    return 0;
}