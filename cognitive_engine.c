/***********************************************************************
 * File:    cognitive_engine.c
 * Author:  Dr. Q & Gemini/Fin
 * Purpose: An advanced cognitive architecture simulation in C. This model
 * builds on the recursive motivational loop by adding:
 * 1. A reward signal (R) to simulate reinforcement.
 * 2. Selective, salient memory (top-k) instead of a simple log.
 * 3. An efficient EMA-based motivational operator (psi).
 * 4. A visualization hook to dump state trajectories to CSV.
 *
 * Build:   gcc -O3 -march=native cognitive_engine.c -lm -o cognitive_engine_demo
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- 1. Core Data Structures ---

// A single state in time, with a measure of its importance.
typedef struct {
    float *vector;
    float salience; // Importance score for this memory
} ActivationState;

// The selective, salient memory system (Top-K storage).
typedef struct {
    int capacity;       // Max number of memories (k)
    int count;          // Current number of memories
    int state_size;
    ActivationState **states; // Array of pointers to states
} SalientMemory;

// The EMA-based motivational operator.
typedef struct {
    int size;
    float *ema_vector;  // Exponential Moving Average of motivation
    float alpha;        // Smoothing factor for the EMA
} PsiOperator;

// Encapsulates the entire state of one "mind".
typedef struct {
    int state_size;
    ActivationState *A; // Current Activation State A[t]
    SalientMemory *M;   // Salient Memory M
    PsiOperator *Psi;   // Motivational Operator ψ
    float *theta;       // Internal weights (trainable parameter)
    FILE *vis_hook;     // Visualization hook (CSV file)
} CognitiveEngine;

// --- 2. Utility & Initialization Functions ---

static float* new_vector(int size) {
    float *v = (float*)calloc(size, sizeof(float));
    if (!v) {
        fprintf(stderr, "OOM for new_vector\n");
        exit(1);
    }
    return v;
}

CognitiveEngine* new_cognitive_engine(int state_size, int memory_capacity, float psi_alpha, const char* vis_filename) {
    CognitiveEngine *engine = (CognitiveEngine*)malloc(sizeof(CognitiveEngine));
    
    engine->state_size = state_size;
    
    // Init Activation State A
    engine->A = (ActivationState*)malloc(sizeof(ActivationState));
    engine->A->vector = new_vector(state_size);
    engine->A->salience = 0.0f;

    // Init Salient Memory M
    engine->M = (SalientMemory*)malloc(sizeof(SalientMemory));
    engine->M->capacity = memory_capacity;
    engine->M->count = 0;
    engine->M->state_size = state_size;
    engine->M->states = (ActivationState**)calloc(memory_capacity, sizeof(ActivationState*));

    // Init Motivational Operator Psi
    engine->Psi = (PsiOperator*)malloc(sizeof(PsiOperator));
    engine->Psi->size = state_size;
    engine->Psi->alpha = psi_alpha;
    engine->Psi->ema_vector = new_vector(state_size);

    // Init internal weights theta (trainable)
    engine->theta = new_vector(state_size);
    for (int i = 0; i < state_size; i++) {
        engine->theta[i] = 1.0f; // Start with identity transformation
    }

    // Init visualization hook
    engine->vis_hook = fopen(vis_filename, "w");
    if (engine->vis_hook) {
        fprintf(engine->vis_hook, "timestep,reward");
        for (int i = 0; i < state_size; i++) fprintf(engine->vis_hook, ",A%d", i);
        for (int i = 0; i < state_size; i++) fprintf(engine->vis_hook, ",Psi%d", i);
        fprintf(engine->vis_hook, "\n");
    }

    return engine;
}

void free_cognitive_engine(CognitiveEngine *engine) {
    if (!engine) return;
    free(engine->A->vector);
    free(engine->A);
    for (int i = 0; i < engine->M->count; i++) {
        free(engine->M->states[i]->vector);
        free(engine->M->states[i]);
    }
    free(engine->M->states);
    free(engine->M);
    free(engine->Psi->ema_vector);
    free(engine->Psi);
    free(engine->theta);
    if(engine->vis_hook) fclose(engine->vis_hook);
    free(engine);
}

void print_vector(const float *vector, int size, const char *name) {
    printf("%-12s: [", name);
    for (int i = 0; i < size; i++) {
        printf(" %7.4f", vector[i]);
    }
    printf(" ]\n");
}

// --- 3. Salient Memory Management ---

// Calculates the "importance" of a state.
float calculate_salience(const ActivationState *a, int state_size, float reward) {
    float magnitude = 0.0f;
    for (int i = 0; i < state_size; i++) {
        magnitude += a->vector[i] * a->vector[i];
    }
    // Reward has a strong, direct impact on salience.
    return sqrtf(magnitude) + reward * 5.0f; 
}

// Adds a state to memory, evicting the least salient if full.
void update_memory(SalientMemory *m, ActivationState *a_new) {
    if (m->count < m->capacity) {
        // Memory not full, just add
        m->states[m->count++] = a_new;
    } else {
        // Memory is full, find least salient to replace
        int least_salient_idx = 0;
        for (int i = 1; i < m->capacity; i++) {
            if (m->states[i]->salience < m->states[least_salient_idx]->salience) {
                least_salient_idx = i;
            }
        }

        if (a_new->salience > m->states[least_salient_idx]->salience) {
            // New state is important enough to remember
            free(m->states[least_salient_idx]->vector);
            free(m->states[least_salient_idx]);
            m->states[least_salient_idx] = a_new;
        } else {
            // New state is not salient enough, discard it
            free(a_new->vector);
            free(a_new);
        }
    }
}

// --- 4. Core Cognitive Loop ---

// The EMA-based psi operator.
void psi_update_and_get(PsiOperator *psi, const SalientMemory *m, float* motivation_out) {
    // Simplified: Motivation is an EMA of the most salient memory.
    if (m->count == 0) {
        memset(motivation_out, 0, psi->size * sizeof(float));
        return;
    }
    
    int most_salient_idx = 0;
    for(int i = 1; i < m->count; ++i) {
        if(m->states[i]->salience > m->states[most_salient_idx]->salience) {
            most_salient_idx = i;
        }
    }
    
    float* target_vec = m->states[most_salient_idx]->vector;
    
    // EMA update: ema_new = alpha * target + (1 - alpha) * ema_old
    for(int i = 0; i < psi->size; ++i) {
        psi->ema_vector[i] = (psi->alpha * target_vec[i]) + (1.0f - psi->alpha) * psi->ema_vector[i];
    }

    memcpy(motivation_out, psi->ema_vector, psi->size * sizeof(float));
}

// The main cognitive step function.
void engine_step(CognitiveEngine *engine, float reward) {
    int size = engine->state_size;

    // --- 1. Persist current state A[t] into Memory M ---
    // Create a copy to store in memory, calculating its salience
    ActivationState *a_copy = (ActivationState*)malloc(sizeof(ActivationState));
    a_copy->vector = new_vector(size);
    memcpy(a_copy->vector, engine->A->vector, size * sizeof(float));
    a_copy->salience = calculate_salience(a_copy, size, reward);
    
    update_memory(engine->M, a_copy);

    // --- 2. Generate Motivation from Memory: psi(M[t]) ---
    float *motivation_vec = new_vector(size);
    psi_update_and_get(engine->Psi, engine->M, motivation_vec);

    // --- 3. Compute next state: A[t+1] = sigma(theta*A[t] + b + psi(M)) ---
    for (int i = 0; i < size; i++) {
        float bias = ((float)rand() / RAND_MAX - 0.5f) * 0.05f; // Small random drift
        // Apply theta, bias, and motivation
        engine->A->vector[i] = (engine->theta[i] * engine->A->vector[i]) + bias + motivation_vec[i];
    }
    
    // Apply activation function sigma (tanh)
    for (int i = 0; i < size; i++) {
        engine->A->vector[i] = tanhf(engine->A->vector[i]);
    }

    // --- 4. Visualization ---
    if (engine->vis_hook) {
        static int timestep = 0;
        fprintf(engine->vis_hook, "%d,%.4f", timestep++, reward);
        for (int i = 0; i < size; i++) fprintf(engine->vis_hook, ",%.6f", engine->A->vector[i]);
        for (int i = 0; i < size; i++) fprintf(engine->vis_hook, ",%.6f", motivation_vec[i]);
        fprintf(engine->vis_hook, "\n");
        fflush(engine->vis_hook);
    }
    
    print_vector(motivation_vec, size, "ψ(M)");
    free(motivation_vec);
}


// --- 5. Main Demonstration ---

int main() {
    srand(time(NULL));

    // --- Configuration ---
    int state_size = 3;         // For easy visualization
    int memory_capacity = 5;    // Remember top 5 moments
    float psi_alpha = 0.2f;     // How quickly motivation adapts
    int total_timesteps = 50;
    
    printf("== Advanced Cognitive Engine Demo ==\n");
    printf("State Size: %d, Memory Capacity: %d, Psi Alpha: %.2f\n\n", state_size, memory_capacity, psi_alpha);

    CognitiveEngine *mind = new_cognitive_engine(state_size, memory_capacity, psi_alpha, "trajectory.csv");

    // Main cognitive loop
    for (int t = 0; t < total_timesteps; t++) {
        // Simulate a random reward signal
        float reward = (rand() % 100 == 0) ? ((float)rand() / RAND_MAX * 2.0f) : 0.0f; // Occasional, strong rewards

        printf("--- Timestep t=%d (Reward: %.4f) ---\n", t, reward);
        print_vector(mind->A->vector, mind->state_size, "A[t]");

        // Run one step of the engine's "thought" process
        engine_step(mind, reward);

        print_vector(mind->A->vector, mind->state_size, "A[t+1]");
        printf("Memory Count: %d/%d\n\n", mind->M->count, mind->M->capacity);
    }

    printf("Simulation complete. Trajectory saved to 'trajectory.csv'.\n");
    free_cognitive_engine(mind);

    return 0;
}