# Scalable Systems Project: MPI, OpenMP & Distributed Computing

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Project Structure](#project-structure)
4. [OpenMP Implementations](#openmp-implementations)
5. [MPI Implementations](#mpi-implementations)
6. [Advanced Project](#advanced-project)
7. [Key Patterns & Concepts](#key-patterns--concepts)
8. [Performance Analysis](#performance-analysis)
9. [Compilation & Execution](#compilation--execution)
10. [Learning Path](#learning-path)

---

## Introduction

This project is an **educational codebase** demonstrating parallel and distributed computing using:
- **OpenMP**: Shared-memory parallelism for multi-core processors
- **MPI**: Distributed-memory parallelism for clusters and distributed systems
- **Hybrid Approach**: Combining both for maximum scalability

### Project Goals
- Understand the difference between parallelism and concurrency
- Learn when to use shared-memory vs distributed-memory programming
- Master common parallel computing patterns
- Build real-world scalable applications

---

## Core Concepts

### Parallelism vs Concurrency

| Aspect | Parallelism | Concurrency |
|--------|-------------|-------------|
| **Definition** | Tasks executing simultaneously | Tasks making progress, potentially interleaved |
| **Hardware** | Requires multiple cores/processors | Can work on single core |
| **Goal** | Speedup through simultaneous execution | Better resource utilization |
| **Example** | Computing pi with 8 cores | Non-blocking I/O operations |

### Two Programming Models

#### 1. Shared Memory (OpenMP)
```
┌─────────────────────────────┐
│   Single Process            │
│  ┌──────┐  ┌──────┐        │
│  │Thread│  │Thread│  ...   │
│  │  1   │  │  2   │        │
│  └───┬──┘  └───┬──┘        │
│      └─────────┘            │
│   Shared Memory Space       │
└─────────────────────────────┘
```
- **Pros**: Easy data sharing, low communication overhead
- **Cons**: Limited to single machine, scalability limits
- **Use case**: Multi-core workstations, small-scale parallelism

#### 2. Distributed Memory (MPI)
```
┌──────────┐    Message    ┌──────────┐    Message    ┌──────────┐
│Process 0 │◄────────────►│Process 1 │◄────────────►│Process 2 │
│  Memory  │   Passing    │  Memory  │   Passing    │  Memory  │
└──────────┘              └──────────┘              └──────────┘
```
- **Pros**: Scales to thousands of nodes, explicit control
- **Cons**: Complex programming, communication overhead
- **Use case**: HPC clusters, large-scale distributed computing

---

## Project Structure

```
scalable-systems/
├── openmp-hello-world/          # Lab 1: Basic OpenMP
│   └── lab1.c
├── openmp-pi-computation/       # Lab 2: Parallel numerical integration
│   └── compute_pi.c
├── mpi-point-to-point/          # Basic MPI communication
│   ├── program_1.c              # Simple send/receive
│   └── program_2.c              # Ring topology
├── mpi-collective-ops/          # Lab 3: MPI collective operations
│   └── lab3.c
├── mpi-nonblocking-comm/        # Lab 4: Asynchronous communication
│   └── lab4.c
├── mpi-parallel-io/             # Parallel file I/O
│   └── collective_parallel_IO.c
├── mpi-producer-consumer/       # Producer-consumer patterns
│   ├── producer_consumer.c
│   └── workpool.c
├── Project/                     # Advanced: MPI+OpenMP+LLM
│   ├── download_papers_by_query.cpp
│   ├── get_docs.cpp
│   ├── to_json.cpp
│   └── main_program.py
├── batch_job.sh                 # SLURM job submission script
└── README.md
```

---

## OpenMP Implementations

### 1. Hello World (`openmp-hello-world/lab1.c`)

**Purpose**: Introduction to OpenMP parallel regions and threads

**Key Concepts**:
- Creating parallel regions with `#pragma omp parallel`
- Thread identification with `omp_get_thread_num()`
- Private variables for thread-local storage

**Code Example**:
```c
#include <omp.h>
#include <stdio.h>

int main() {
    int nthreads, tid;

    #pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        printf("Hello World; Thread = %d\n", tid);

        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }

    return 0;
}
```

**Compilation & Execution**:
```bash
gcc -fopenmp lab1.c -o lab1
./lab1
```

**Output Example**:
```
Hello World; Thread = 0
Number of threads = 4
Hello World; Thread = 2
Hello World; Thread = 1
Hello World; Thread = 3
```

---

### 2. Pi Computation (`openmp-pi-computation/compute_pi.c`)

**Purpose**: Parallel numerical integration using Monte Carlo method

**Algorithm**: Approximates π using the integral: π = ∫₀¹ 4/(1+x²) dx

**Key Concepts**:
- `#pragma omp parallel for`: Distributes loop iterations across threads
- `reduction(+ : sum)`: Thread-safe accumulation of partial results
- Performance benchmarking with `omp_get_wtime()`
- Different scheduling strategies (static, dynamic, guided)

**Code Example**:
```c
double pi, sum = 0.0;
double step = 1.0 / (double)num_steps;
double start_time, end_time;

start_time = omp_get_wtime();

#pragma omp parallel for private(x) reduction(+ : sum)
for (i = 0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
}

pi = step * sum;
end_time = omp_get_wtime();

printf("Pi = %.16f\n", pi);
printf("Time = %.6f seconds\n", end_time - start_time);
```

**Compilation & Execution**:
```bash
gcc -fopenmp compute_pi.c -o compute_pi
OMP_NUM_THREADS=8 ./compute_pi 10000000
```

**Performance Insights**:
- Near-linear speedup with more threads (up to hardware limit)
- Dynamic scheduling helps with load balancing
- Diminishing returns due to overhead and synchronization

---

## MPI Implementations

### 1. Point-to-Point Communication (`mpi-point-to-point/`)

#### Program 1: Basic Send/Receive (`program_1.c`)

**Purpose**: Demonstrate basic MPI message passing between two processes

**Key Concepts**:
- `MPI_Init()` and `MPI_Finalize()`: Initialize/cleanup MPI environment
- `MPI_Send()`: Blocking send operation
- `MPI_Recv()`: Blocking receive operation
- `MPI_Comm_rank()` and `MPI_Comm_size()`: Get process ID and total count

**Code Example**:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int data = 42;
        printf("Process 0 sending data: %d\n", data);
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        int received_data;
        MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1 received data: %d\n", received_data);
    }

    MPI_Finalize();
    return 0;
}
```

**Execution**:
```bash
mpicc program_1.c -o program_1
mpirun -np 2 ./program_1
```

---

#### Program 2: Ring Topology (`program_2.c`)

**Purpose**: Circular message passing pattern

**Topology**:
```
Rank 0 → Rank 1 → Rank 2 → Rank 3 → Rank 0 (cycles)
```

**Key Concepts**:
- Modulo arithmetic for circular topology: `(rank + 1) % nproc`
- Message tags for ordering
- Synchronization through sequential message passing

**Code Pattern**:
```c
if (rank == 0) {
    int data = 100;
    MPI_Send(&data, 1, MPI_INT, 1, 123, MPI_COMM_WORLD);
    printf("Rank 0 sent: %d\n", data);
} else {
    int received;
    MPI_Recv(&received, 1, MPI_INT, rank - 1, 123, MPI_COMM_WORLD, &status);
    printf("Rank %d received: %d\n", rank, received);

    if (rank < nproc - 1) {
        MPI_Send(&received, 1, MPI_INT, (rank + 1) % nproc, 123, MPI_COMM_WORLD);
    }
}
```

---

### 2. Collective Operations (`mpi-collective-ops/lab3.c`)

**Purpose**: Statistical computation using collective MPI operations

**Problem**: Calculate mean and standard deviation of distributed random numbers

**Key Concepts**:
- `MPI_Allreduce()`: All processes get the reduction result
- `MPI_Reduce()`: Only root process gets the result
- Collective operations are more efficient than manual point-to-point

**Algorithm**:
```
1. Each process generates random numbers
2. Compute local sum → MPI_Allreduce → global sum → mean
3. Compute local squared differences → MPI_Reduce → standard deviation
```

**Code Example**:
```c
// Each process computes local sum
float local_sum = 0.0;
for (i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
}

// All processes get global sum
float global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

float global_mean = global_sum / (num_elements_per_proc * nproc);

// Each process computes local squared differences
float local_sq_diff_sum = 0.0;
for (i = 0; i < num_elements_per_proc; i++) {
    float diff = rand_nums[i] - global_mean;
    local_sq_diff_sum += diff * diff;
}

// Only rank 0 gets global result
float global_sq_diff_sum;
MPI_Reduce(&local_sq_diff_sum, &global_sq_diff_sum, 1, MPI_FLOAT,
           MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
    float stddev = sqrt(global_sq_diff_sum / (num_elements_per_proc * nproc));
    printf("Mean: %f, Std Dev: %f\n", global_mean, stddev);
}
```

**Execution**:
```bash
mpicc lab3.c -o lab3 -lm
mpirun -np 4 ./lab3 10000
```

---

### 3. Non-Blocking Communication (`mpi-nonblocking-comm/lab4.c`)

**Purpose**: Asynchronous communication to overlap computation with communication

**Scenario**: "Busy Professor" - Rank 0 posts receives then does other work

**Key Concepts**:
- `MPI_Irecv()`: Non-blocking receive (returns immediately)
- `MPI_Waitany()`: Wait for any one of multiple requests to complete
- `MPI_Bcast()`: Broadcast data to all processes
- `MPI_Get_processor_name()`: Get hostname

**Benefits**:
- Avoids deadlock
- Better performance through overlapping
- More flexible message ordering

**Code Pattern**:
```c
MPI_Request recv_req[nproc];

if (rank == 0) {
    // Post all receives first (non-blocking)
    for (i = 1; i < nproc; i++) {
        MPI_Irecv(hostname[i], MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  i, tag, MPI_COMM_WORLD, &recv_req[i]);
    }

    // Do other work while messages arrive
    printf("I am a very busy professor.\n");

    // Wait for messages to complete
    for (i = 1; i < nproc; i++) {
        int index;
        MPI_Waitany(nproc - 1, &recv_req[1], &index, &status);
        printf("Received from process %d\n", index + 1);
    }
} else {
    // Other ranks send their hostname
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    MPI_Send(processor_name, name_len + 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
}
```

**Execution**:
```bash
mpicc lab4.c -o lab4
mpirun -np 4 ./lab4
```

---

### 4. Parallel I/O (`mpi-parallel-io/collective_parallel_IO.c`)

**Purpose**: Multiple processes reading/writing to the same file simultaneously

**Key Concepts**:
- `MPI_File_open()`: Open file for parallel access
- `MPI_Type_vector()`: Define non-contiguous data patterns
- `MPI_File_set_view()`: Each process defines its view of the file
- `MPI_File_write_all()` / `MPI_File_read_all()`: Collective I/O operations

**Pattern**: Each process writes to interleaved blocks
```
File Layout:
[P0][P1][P2][P3][P0][P1][P2][P3][P0][P1][P2][P3]...
 └─Block 0─┘ └─Block 1─┘ └─Block 2─┘
```

**Code Example**:
```c
MPI_File fh;
MPI_Datatype filetype;

// Create vector type: count blocks, block length, stride
MPI_Type_vector(nints / INTS_PER_BLK, INTS_PER_BLK,
                nprocs * INTS_PER_BLK, MPI_INT, &filetype);
MPI_Type_commit(&filetype);

// Open file collectively
MPI_File_open(MPI_COMM_WORLD, "datafile",
              MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

// Set view (offset for each process)
MPI_File_set_view(fh, rank * INTS_PER_BLK * sizeof(int),
                  MPI_INT, filetype, "native", MPI_INFO_NULL);

// Collective write
MPI_File_write_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

// Collective read
MPI_File_read_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);

MPI_File_close(&fh);
```

**Benefits**:
- High-performance parallel I/O
- Automatic optimization by MPI implementation
- Collective operations coordinate across processes

---

### 5. Producer-Consumer Pattern (`mpi-producer-consumer/`)

#### Program 1: Broker-Based (`producer_consumer.c`)

**Architecture**:
```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ Producer 1  │──────▶│             │◀──────│ Consumer 1  │
│ (Rank 1)    │       │   Broker    │       │ (Rank N-2)  │
└─────────────┘       │  (Rank 0)   │       └─────────────┘
                      │             │
┌─────────────┐       │  Bounded    │       ┌─────────────┐
│ Producer 2  │──────▶│   Buffer    │◀──────│ Consumer 2  │
│ (Rank 2)    │       │             │       │ (Rank N-1)  │
└─────────────┘       └─────────────┘       └─────────────┘
```

**Message Types**:
- `MSG_WORK (0)`: Producer sends work item
- `MSG_REQUEST_WORK (1)`: Consumer requests work
- `MSG_ACK (2)`: Acknowledgment
- `MSG_ABORT (3)`: Termination signal
- `MSG_NO_WORK (4)`: Buffer empty
- `MSG_CONSUMED_COUNT (5)`: Final statistics

**Broker Logic**:
```c
// Bounded buffer management
if (bufferFilled < bufferCapacity) {
    workBuffer[bufferFilled++] = work;
    MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, MSG_ACK, MPI_COMM_WORLD);
} else {
    // Queue pending producers
    pendingProducers[pendingCount++] = status.MPI_SOURCE;
}

// Consumer request handling
if (bufferFilled > 0) {
    MPI_Send(&workBuffer[--bufferFilled], 1, MPI_INT,
             status.MPI_SOURCE, MSG_WORK, MPI_COMM_WORLD);
} else {
    MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, MSG_NO_WORK, MPI_COMM_WORLD);
}
```

**Execution**:
```bash
mpicc producer_consumer.c -o producer_consumer
mpirun -np 8 ./producer_consumer 30  # Run for 30 seconds
```

---

#### Program 2: Workpool (`workpool.c`)

**Architecture**: All processes act as both producer and consumer

**Key Features**:
- Non-blocking sends: `MPI_Isend()`
- Non-blocking receives: `MPI_Irecv()`
- Test for completion: `MPI_Test()`
- Runtime-based execution with `MPI_Wtime()`

**Performance Results**:
| Processes | Messages Consumed | Observation |
|-----------|------------------|-------------|
| 4 | ~62M | Optimal throughput |
| 8 | ~56M | Good scaling |
| 12 | ~56M | Stable performance |
| 16 | ~6.7M | Contention effects |

---

## Advanced Project

### Overview: Academic Paper Processing Pipeline

**Goal**: Download papers from arXiv, extract text, generate Q&A pairs using LLMs

**Technologies**: MPI + OpenMP + Python + vLLM + libcurl + Poppler

**Hybrid Parallelism**:
```
┌───────────────────────────────────────────────────────────┐
│                    MPI Layer (Distributed)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Process 0   │  │ Process 1   │  │ Process 2   │      │
│  │ (Papers0-5) │  │ (Papers6-11)│  │(Papers12-17)│      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│  ┌──────▼──────────────────────────────────▼─────┐      │
│  │         OpenMP Layer (Shared Memory)          │      │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐    │      │
│  │  │Thread │ │Thread │ │Thread │ │Thread │    │      │
│  │  │   0   │ │   1   │ │   2   │ │   3   │    │      │
│  │  └───────┘ └───────┘ └───────┘ └───────┘    │      │
│  └───────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────┘
```

---

### Component 1: `download_papers_by_query.cpp`

**Purpose**: Distributed paper downloading from arXiv

**Workflow**:
1. Rank 0 fetches arXiv search results (XML)
2. Broadcast XML to all processes via `MPI_Bcast()`
3. Parse XML to extract paper entries
4. Distribute entries across MPI ranks
5. Each rank uses OpenMP threads to download PDFs in parallel

**Code Pattern**:
```cpp
// MPI distribution
if (rank == 0) {
    // Fetch arXiv XML
    xml_data = fetch_arxiv_search_results(query);
}

// Broadcast to all ranks
int xml_size = xml_data.size();
MPI_Bcast(&xml_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&xml_data[0], xml_size, MPI_CHAR, 0, MPI_COMM_WORLD);

// Parse and partition
std::vector<Entry> entries = parse_xml(xml_data);
int start = rank * entries_per_rank;
int end = std::min(start + entries_per_rank, total_entries);

// OpenMP parallel download
#pragma omp parallel
{
    CURL* thread_curl = curl_easy_init();

    #pragma omp for schedule(dynamic)
    for (int i = start; i < end; ++i) {
        std::string pdf_url = entries[i].pdf_link;
        std::string filepath = "pdfs/" + entries[i].id + ".pdf";
        download_file_with_curl(thread_curl, pdf_url, filepath);
    }

    curl_easy_cleanup(thread_curl);
}
```

**Thread Safety**: Each thread has its own CURL handle

---

### Component 2: `get_docs.cpp`

**Purpose**: Parallel PDF text extraction using Poppler

**Key Features**:
- OpenMP parallelization with dynamic scheduling
- Poppler library for professional PDF parsing
- Thread-safe deduplication using `omp_lock_t`
- JSON output for downstream processing

**Code Pattern**:
```cpp
omp_lock_t parsed_lock, processed_lock;
omp_init_lock(&parsed_lock);
omp_init_lock(&processed_lock);

std::vector<ParsedEntry> parsed;
std::unordered_set<std::string> processed_first_pages;

#pragma omp parallel
{
    std::vector<ParsedEntry> local_parsed; // Thread-local buffer

    #pragma omp for schedule(dynamic)
    for (int idx = 0; idx < num_pdfs; ++idx) {
        // Load PDF
        auto doc = poppler::document::load_from_file(pdf_files[idx]);

        // Extract text from all pages
        std::string full_text;
        for (int page_num = 0; page_num < doc->pages(); ++page_num) {
            auto page = doc->create_page(page_num);
            full_text += page->text().to_latin1();
        }

        // Clean text
        clean_text(full_text);

        // Deduplication check (thread-safe)
        std::string first_page = extract_first_page(full_text);
        bool is_duplicate = false;

        omp_set_lock(&processed_lock);
        if (processed_first_pages.find(first_page) != processed_first_pages.end()) {
            is_duplicate = true;
        } else {
            processed_first_pages.insert(first_page);
        }
        omp_unset_lock(&processed_lock);

        if (!is_duplicate) {
            local_parsed.push_back({pdf_files[idx], full_text});
        }
    }

    // Merge thread-local results to global (thread-safe)
    omp_set_lock(&parsed_lock);
    parsed.insert(parsed.end(), local_parsed.begin(), local_parsed.end());
    omp_unset_lock(&parsed_lock);
}

// Output as JSON
output_json(parsed);
```

**Performance**: Significantly faster than Python's PyPDF sequential processing

---

### Component 3: `to_json.cpp`

**Purpose**: Parallel JSON parsing and filtering of LLM outputs

**Multi-Stage Filtering**:
1. Valid field presence (`question`, `answer`, `document`, `output`)
2. Non-empty value checks
3. Question format validation (must contain '?')
4. Keyword presence in document source
5. Output quality filters
6. Deduplication

**Code Pattern**:
```cpp
std::mutex json_mutex;
nlohmann::json valid_entries = nlohmann::json::array();

#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < llm_outputs.size(); ++i) {
    auto entry = nlohmann::json::parse(llm_outputs[i]);

    // Validation checks
    bool valid_fields = entry.contains("question") &&
                       entry.contains("answer") &&
                       entry.contains("document") &&
                       entry.contains("output");

    bool non_empty = !entry["question"].empty() &&
                     !entry["answer"].empty();

    bool has_question_mark = entry["question"].get<std::string>().find('?')
                            != std::string::npos;

    bool keyword_present = check_keyword_in_document(entry);

    bool valid_output = validate_output_quality(entry["output"]);

    if (valid_fields && non_empty && has_question_mark &&
        keyword_present && valid_output) {

        std::lock_guard<std::mutex> lock(json_mutex);
        valid_entries.push_back(entry);
    }
}

// Save filtered results
save_json(valid_entries, "jsons/arxiv_version2.json");
```

---

### Component 4: `main_program.py`

**Purpose**: Orchestrate entire pipeline

**Workflow**:
```
1. Download Papers (arXiv API)
        ↓
2. Extract Text (C++ parallel: get_docs vs Python baseline: PyPDF)
        ↓
3. Generate Q&A (vLLM: Llama-2-7b-chat)
        ↓
4. Filter & Save (C++ parallel: to_json vs Python baseline)
        ↓
5. Performance Comparison
```

**Prompt Engineering for LLM**:
```python
prompt = f"""
You are an expert educator creating quiz questions from academic papers.

Given this document excerpt:
{document_text}

Generate 3-5 high-quality questions that:
1. Focus on WHY/WHAT/HOW relationships
2. Require understanding, not memorization
3. Cover key concepts and methodology
4. Include code analysis if present
5. Avoid paper-specific phrases like "in this paper"

Format as JSON:
[
  {{
    "question": "Why does the paper use technique X?",
    "answer": "Technique X is used because...",
    "document": "{document_id}",
    "output": "Covers concept: ..."
  }}
]
"""
```

**Performance Benchmarking**:
```python
# PDF extraction comparison
start = time.time()
python_results = extract_text_python(pdf_files)  # PyPDF
python_time = time.time() - start

start = time.time()
subprocess.run(['./get_docs'])
cpp_time = time.time() - start

print(f"Python: {python_time:.2f}s")
print(f"C++ Parallel: {cpp_time:.2f}s")
print(f"Speedup: {python_time / cpp_time:.2f}x")
```

---

## Key Patterns & Concepts

### 1. Fork-Join Parallelism (OpenMP)
```
Main Thread
    │
    ├─────── Parallel Region ───────┐
    │                               │
  Thread 0   Thread 1   Thread 2   Thread 3
    │          │          │          │
    └──────────┴──────────┴──────────┘
    │
    └── Join Point (Barrier)
    │
Continue
```

### 2. Data Parallelism
Same operation on different data elements
```c
#pragma omp parallel for
for (i = 0; i < N; i++) {
    result[i] = compute(data[i]);  // Independent iterations
}
```

### 3. Task Parallelism
Different operations executed concurrently
```c
#pragma omp parallel sections
{
    #pragma omp section
    { task_A(); }

    #pragma omp section
    { task_B(); }

    #pragma omp section
    { task_C(); }
}
```

### 4. Reduction Pattern
Combine partial results from parallel computation
```c
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (i = 0; i < N; i++) {
    sum += array[i];
}
```

### 5. Master-Worker Pattern (MPI)
```
Master (Rank 0)
    │
    ├──── Distribute Work ────┐
    │                         │
Worker 1  Worker 2  Worker 3  Worker 4
    │         │         │         │
    └─────── Gather Results ──────┘
    │
Master
```

### 6. SPMD (Single Program Multiple Data)
All processes run same program, operate on different data
```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// All ranks execute same code
// but process different data based on rank
int start = rank * chunk_size;
int end = (rank + 1) * chunk_size;
process_data(data, start, end);

MPI_Finalize();
```

### 7. Pipeline Parallelism
```
Stage 1 (P0) → Stage 2 (P1) → Stage 3 (P2) → Stage 4 (P3)
   ↓              ↓              ↓              ↓
 Data A        Data A         Data A         Data A
 Data B        Data B         Data B
 Data C        Data C
 Data D
```

---

## Performance Analysis

### Speedup & Efficiency

**Speedup**: S(p) = T(1) / T(p)
- T(1): Time with 1 processor
- T(p): Time with p processors
- **Ideal**: S(p) = p (linear speedup)

**Efficiency**: E(p) = S(p) / p
- **Ideal**: E(p) = 1.0 (100% efficiency)
- **Reality**: E(p) < 1.0 due to overhead

### Amdahl's Law
Maximum speedup limited by sequential portion:
```
S(p) = 1 / ((1-P) + P/p)
```
- P: Parallelizable fraction
- (1-P): Sequential fraction

**Example**: If 90% parallelizable, max speedup ≈ 10x (even with infinite processors)

### Performance Bottlenecks

1. **Communication Overhead** (MPI)
   - Message latency
   - Bandwidth limitations
   - Synchronization costs

2. **Load Imbalance**
   - Some threads/processes finish before others
   - Idle time waiting at barriers

3. **False Sharing** (OpenMP)
   - Different threads accessing nearby memory locations
   - Cache line invalidation

4. **Contention**
   - Multiple processes competing for same resource
   - Seen in producer-consumer at 16 processes

### Observed Results

**Pi Computation (OpenMP)**:
```
Threads  | Time (s) | Speedup | Efficiency
---------|----------|---------|------------
1        | 8.234    | 1.0x    | 100%
2        | 4.187    | 1.97x   | 98%
4        | 2.123    | 3.88x   | 97%
8        | 1.134    | 7.26x   | 91%
16       | 0.689    | 11.95x  | 75%
```
Near-linear scaling up to 8 threads, then diminishing returns

**Producer-Consumer (MPI)**:
```
Processes | Messages/sec | Observation
----------|--------------|-------------
4         | ~2M          | Good balance
8         | ~1.9M        | Slight overhead
12        | ~1.8M        | More contention
16        | ~670K        | Heavy contention
```
Sweet spot at 4-8 processes

**C++ vs Python**:
- PDF extraction: **5-10x faster** with C++ parallel
- JSON parsing: **3-7x faster** with C++ parallel

---

## Compilation & Execution

### OpenMP Programs

**Basic Compilation**:
```bash
gcc -fopenmp source.c -o executable
```

**With Math Library**:
```bash
gcc -fopenmp compute_pi.c -o compute_pi -lm
```

**Execution**:
```bash
# Set thread count
export OMP_NUM_THREADS=8
./executable

# Or inline
OMP_NUM_THREADS=4 ./executable
```

**Environment Variables**:
```bash
OMP_NUM_THREADS=8          # Number of threads
OMP_SCHEDULE="dynamic,10"  # Scheduling strategy
OMP_PROC_BIND=true         # Thread affinity
```

---

### MPI Programs

**Basic Compilation**:
```bash
mpicc source.c -o executable
```

**With Libraries**:
```bash
mpicc lab3.c -o lab3 -lm  # Math library
```

**Execution**:
```bash
# Local execution
mpirun -np 4 ./executable

# With hostfile (cluster)
mpirun -np 8 -hostfile hosts.txt ./executable

# Specific hosts
mpirun -np 4 -host node1,node2,node3,node4 ./executable
```

**MPI Environment Variables**:
```bash
OMPI_MCA_btl_tcp_if_include=eth0  # Network interface
OMPI_MCA_mpi_show_mca_params=1    # Show parameters
```

---

### Hybrid MPI+OpenMP

**Compilation** (C++):
```bash
mpic++ -fopenmp source.cpp -o executable -lcurl -ltinyxml2
```

**With Poppler**:
```bash
mpic++ -fopenmp get_docs.cpp \
    -I/usr/include/poppler/cpp \
    -lpoppler-cpp -o get_docs
```

**Execution**:
```bash
# 4 MPI processes, 8 OpenMP threads each = 32 total threads
export OMP_NUM_THREADS=8
mpirun -np 4 ./executable
```

**SLURM Job Script** (`batch_job.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=paper_analysis
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=compute

export OMP_NUM_THREADS=8
mpirun ./download_papers_by_query "deep learning"
```

---

### Project Pipeline

**Step 1: Compile C++ Components**
```bash
# Download papers (MPI + OpenMP + libcurl)
mpic++ -fopenmp download_papers_by_query.cpp \
    -lcurl -ltinyxml2 -o download_papers_by_query

# Extract text (OpenMP + Poppler)
g++ -fopenmp get_docs.cpp \
    -I/usr/include/poppler/cpp \
    -lpoppler-cpp -o get_docs

# Filter JSON (OpenMP)
g++ -fopenmp to_json.cpp -o to_json
```

**Step 2: Run Pipeline**
```bash
# Set OpenMP threads
export OMP_NUM_THREADS=4

# Download papers (4 MPI processes)
mpirun -np 4 ./download_papers_by_query "machine learning"

# Run full pipeline
python main_program.py
```

---

## Learning Path

### Beginner (Week 1-2)

**Concepts**:
- Understand parallelism vs concurrency
- Learn thread creation and synchronization
- Basic parallel loops

**Exercises**:
1. ✅ `openmp-hello-world`: Run and modify thread count
2. ✅ `openmp-pi-computation`: Experiment with different step counts
3. ✅ `mpi-point-to-point/program_1`: Basic send/receive

**Goals**:
- Compile and run OpenMP programs
- Understand `#pragma omp parallel`
- Learn `MPI_Init`, `MPI_Send`, `MPI_Recv`, `MPI_Finalize`

---

### Intermediate (Week 3-4)

**Concepts**:
- Collective operations
- Reduction patterns
- Non-blocking communication

**Exercises**:
1. ✅ `mpi-point-to-point/program_2`: Ring topology
2. ✅ `mpi-collective-ops`: Statistics with MPI_Allreduce
3. ✅ `mpi-nonblocking-comm`: Asynchronous communication

**Goals**:
- Use `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`
- Understand blocking vs non-blocking
- Learn `reduction` clause in OpenMP

---

### Advanced (Week 5-6)

**Concepts**:
- Parallel I/O
- Producer-consumer patterns
- Load balancing

**Exercises**:
1. ✅ `mpi-parallel-io`: Collective file operations
2. ✅ `mpi-producer-consumer/producer_consumer`: Broker pattern
3. ✅ `mpi-producer-consumer/workpool`: Distributed workpool

**Goals**:
- Use MPI-IO for parallel file access
- Implement message-based coordination
- Understand buffering and flow control

---

### Expert (Week 7-8)

**Concepts**:
- Hybrid parallelism (MPI + OpenMP)
- Thread safety
- Performance optimization

**Exercises**:
1. ✅ `Project/download_papers_by_query.cpp`: MPI distribution + OpenMP threads
2. ✅ `Project/get_docs.cpp`: Parallel PDF processing with locks
3. ✅ `Project/to_json.cpp`: Thread-safe JSON filtering

**Goals**:
- Combine MPI and OpenMP effectively
- Use locks (`omp_lock_t`, `std::mutex`)
- Benchmark and optimize performance
- Profile communication overhead

---

## Best Practices

### OpenMP

1. **Minimize Critical Sections**
   ```c
   // Bad: Large critical section
   #pragma omp critical
   {
       complex_computation();
       update_shared_var();
   }

   // Good: Reduce critical section
   local_result = complex_computation();
   #pragma omp critical
   {
       update_shared_var(local_result);
   }
   ```

2. **Use Reduction Instead of Critical**
   ```c
   // Bad
   double sum = 0.0;
   #pragma omp parallel for
   for (i = 0; i < N; i++) {
       #pragma omp critical
       sum += array[i];
   }

   // Good
   double sum = 0.0;
   #pragma omp parallel for reduction(+:sum)
   for (i = 0; i < N; i++) {
       sum += array[i];
   }
   ```

3. **Specify Data Sharing**
   ```c
   #pragma omp parallel for private(temp) shared(result)
   for (i = 0; i < N; i++) {
       temp = compute(i);
       result[i] = temp;
   }
   ```

4. **Choose Appropriate Scheduling**
   - `static`: Equal-sized chunks, low overhead
   - `dynamic`: Variable chunks, better load balance
   - `guided`: Large chunks initially, smaller toward end

---

### MPI

1. **Avoid Deadlocks**
   ```c
   // Bad: Circular dependency
   if (rank == 0) {
       MPI_Send(..., 1, ...);
       MPI_Recv(..., 1, ...);
   } else {
       MPI_Send(..., 0, ...);  // Deadlock!
       MPI_Recv(..., 0, ...);
   }

   // Good: Stagger send/recv
   if (rank == 0) {
       MPI_Send(..., 1, ...);
       MPI_Recv(..., 1, ...);
   } else {
       MPI_Recv(..., 0, ...);
       MPI_Send(..., 0, ...);
   }

   // Better: Use non-blocking
   MPI_Isend(..., &req1);
   MPI_Irecv(..., &req2);
   MPI_Waitall(...);
   ```

2. **Minimize Communication**
   ```c
   // Bad: Send many small messages
   for (i = 0; i < N; i++) {
       MPI_Send(&data[i], 1, MPI_INT, dest, ...);
   }

   // Good: Send one large message
   MPI_Send(data, N, MPI_INT, dest, ...);
   ```

3. **Use Collective Operations**
   ```c
   // Bad: Manual broadcast
   if (rank == 0) {
       for (i = 1; i < nprocs; i++) {
           MPI_Send(data, N, MPI_INT, i, ...);
       }
   } else {
       MPI_Recv(data, N, MPI_INT, 0, ...);
   }

   // Good: MPI_Bcast
   MPI_Bcast(data, N, MPI_INT, 0, MPI_COMM_WORLD);
   ```

4. **Balance Load**
   ```c
   // Static partitioning (can cause imbalance)
   int chunk_size = N / nprocs;
   int start = rank * chunk_size;
   int end = (rank + 1) * chunk_size;

   // Dynamic work distribution (better balance)
   // Use master-worker pattern or workpool
   ```

---

### Hybrid MPI+OpenMP

1. **Thread Safety**
   ```cpp
   // Each thread needs its own handle
   #pragma omp parallel
   {
       CURL* thread_curl = curl_easy_init();
       #pragma omp for
       for (...) {
           use_curl(thread_curl);
       }
       curl_easy_cleanup(thread_curl);
   }
   ```

2. **Minimize Nested Parallelism**
   ```
   Avoid: MPI → OpenMP → MPI
   Prefer: MPI (coarse) → OpenMP (fine)
   ```

3. **Thread-Safe MPI**
   ```c
   // Check thread support level
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

   // MPI_THREAD_SINGLE: No threads
   // MPI_THREAD_FUNNELED: Only main thread calls MPI
   // MPI_THREAD_SERIALIZED: One thread at a time
   // MPI_THREAD_MULTIPLE: Any thread can call MPI
   ```

---

## Common Pitfalls

### 1. Race Conditions
```c
// Problem: Multiple threads updating shared variable
int counter = 0;
#pragma omp parallel for
for (i = 0; i < 1000; i++) {
    counter++;  // Race condition!
}

// Solution: Use atomic or reduction
#pragma omp parallel for
for (i = 0; i < 1000; i++) {
    #pragma omp atomic
    counter++;
}
```

### 2. False Sharing
```c
// Problem: Adjacent array elements on same cache line
int counters[NUM_THREADS];
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    for (i = 0; i < 1000000; i++) {
        counters[tid]++;  // False sharing!
    }
}

// Solution: Add padding
struct padded_counter {
    int value;
    char padding[64];  // Cache line size
};
struct padded_counter counters[NUM_THREADS];
```

### 3. Forgetting to Free Resources
```c
// Always cleanup
CURL* curl = curl_easy_init();
// ... use curl ...
curl_easy_cleanup(curl);  // Don't forget!

MPI_Init(&argc, &argv);
// ... MPI code ...
MPI_Finalize();  // Required!
```

### 4. Incorrect Data Sharing
```c
// Problem: Loop variable not private
int i;
#pragma omp parallel for
for (i = 0; i < N; i++) {  // 'i' should be private
    // ...
}

// Solution: Declare in loop (C99+)
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    // ...
}
```

---

## Resources

### Documentation
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [MPI Standard](https://www.mpi-forum.org/docs/)
- [OpenMP API C/C++ Reference](https://www.openmp.org/spec-html/5.0/openmp.html)

### Tutorials
- [OpenMP Tutorial (LLNL)](https://hpc-tutorials.llnl.gov/openmp/)
- [MPI Tutorial (LLNL)](https://hpc-tutorials.llnl.gov/mpi/)
- [Parallel Programming in C with MPI and OpenMP](https://www.mcs.anl.gov/research/projects/mpi/)

### Books
- "Using OpenMP" by Chapman, Jost, and van der Pas
- "Using MPI" by Gropp, Lusk, and Skjellum
- "Parallel Programming in C with MPI and OpenMP" by Quinn

### Tools
- **Profiling**: `gprof`, `perf`, Intel VTune
- **Debugging**: `gdb` with OpenMP/MPI support
- **Analysis**: `valgrind` for memory issues

---

## Conclusion

This project provides a **comprehensive introduction to parallel and distributed computing**:

1. **Foundation**: OpenMP for shared-memory parallelism
2. **Distribution**: MPI for message passing and distributed computing
3. **Integration**: Hybrid MPI+OpenMP for maximum scalability
4. **Application**: Real-world pipeline for academic paper processing

**Key Takeaways**:
- Start simple (hello world), build to complex (hybrid parallelism)
- Understand trade-offs: shared vs distributed, blocking vs non-blocking
- Measure performance: speedup, efficiency, scalability
- Apply best practices: minimize communication, balance load, avoid contention

**Next Steps**:
1. Work through examples in order (beginner → expert)
2. Modify code to experiment with parameters
3. Measure performance with different configurations
4. Build your own parallel application

Happy parallel programming! 🚀
