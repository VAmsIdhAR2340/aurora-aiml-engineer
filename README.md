# Aurora QA System

A production-ready question-answering system built with RAG (Retrieval-Augmented Generation) that answers natural language questions about luxury concierge member data.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)

**Live Demo**: https://aurora-aiml-engineer.onrender.com/
Above is my deployed URL, the site would be inactive if there is no one visiting, so if it all you see that, just wait for 50-60 seconds and reload the site.

**Video Recording**: https://drive.google.com/file/d/1o_MULjOSMsfH_4MvBAPYBoqE79eOpiaw/view?usp=sharing
Above is the video recording link.

## Overview

This system processes 3,349 member messages from a luxury concierge service and answers questions like:
- *"When is Layla planning her trip to London?"*
- *"How many cars does Vikram Desai have?"*
- *"What are Amira's favorite restaurants?"*

The system uses semantic search to find relevant messages and an LLM to generate accurate, context-aware answers.

---

## Features

✅ **RESTful API** - Simple `/ask` endpoint with JSON responses  
✅ **Semantic Search** - Vector embeddings for accurate context retrieval  
✅ **Fast Responses** - Cached data with <2s average response time  
✅ **Production Ready** - Error handling, logging, and retry logic  
✅ **Scalable** - Handles 3,349+ messages with room to grow  
✅ **Interactive Docs** - Auto-generated Swagger UI at `/docs`

---

## Architecture Flow

1. **User submits question** via `/ask` endpoint
2. **Data Loader** fetches and caches all member messages
3. **FAISS Vector Store** retrieves top-10 most relevant messages using semantic search
4. **Perplexity LLM** generates natural language answer from context
5. **Response** returned as clean JSON

---

## Quick Start

### Prerequisites
- Python 3.9+
- Perplexity API key ([Get one here](https://www.perplexity.ai/settings/api))

### **Create virtual environment**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### **Installation**
- Clone the repo and install dependencies using pip install -r requirements.txt

### **Configure environment**
- Copy `.env.example` to `.env` and add your Perplexity API key:

### **Run server**
- uvicorn app.main:app --reload --port 8000

### **Test**
```
In local: Visit http://localhost:8000/docs or:
In Website: Visit: https://aurora-aiml-engineer.onrender.com/

For local: curl -X POST "http://localhost:8000/ask" / To test using website link: https://aurora-aiml-engineer.onrender.com/ask
-H "Content-Type: application/json"
-d '{"question": "When is Layla planning her trip to London?"}'
```
### Tech Stack

- **Framework**: FastAPI
- **LLM**: Perplexity AI (sonar model)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Data**: Pandas

---

## API Endpoints

### `POST /ask` - Ask a Question

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

**Response:**
```
{"answer":"Layla Kawaguchi is planning her trip to London around March 11, 2025, as she requested a car service to meet her upon arrival in London on that date."}
```

## Bonus 1: Design Notes

### Chosen Approach: RAG (Retrieval-Augmented Generation)

**How it works:**
1. **Index**: All 3,349 messages embedded and stored in FAISS
2. **Retrieve**: For each query, top-10 most relevant messages retrieved
3. **Generate**: Context passed to Perplexity AI for answer generation

**Why RAG?**
-  Grounds responses in actual data (no hallucinations)
-  Efficient for large datasets
-  Explainable (can show source messages)
-  No fine-tuning required

### Alternative Approaches Considered

#### 1. Fine-Tuned LLM
**Approach**: Fine-tune GPT-2/LLaMA on message dataset

**Pros**: No API calls, faster inference  
**Cons**: 
-  Insufficient training data (3,349 messages)
-  Expensive GPU requirements
-  Risk of hallucinations
-  Hard to update

**Final Call**: Not suitable for this dataset size.

#### 2. Keyword Search + Rules
**Approach**: Regex patterns and keyword matching

**Pros**: Simple, fast, no dependencies  
**Cons**:
-  Manual rule creation needed
-  Poor generalization
-  Breaks with paraphrasing
-  Can't handle complex queries

**Final Call**: Too brittle for production.

#### 3. BERT Question Answering
**Approach**: Pre-trained extractive QA model

**Pros**: Open-source, good for span extraction  
**Cons**:
-  Extractive only (no synthesis)
-  Limited to exact text spans
-  Computationally expensive

**Final Call**: Less flexible than RAG.

#### 4. Graph Database (Neo4j)
**Approach**: Model as knowledge graph

**Pros**: Excellent for relationships  
**Cons**:
-  Requires entity extraction
-  Poor for free-form questions
-  High engineering effort

**Final Call**: Overkill for this use case.

#### 5. Hybrid BM25 + Semantic
**Approach**: Combine lexical and semantic search

**Pros**: Better retrieval accuracy  
**Cons**: More complex

**Final Call**: Good for production enhancement.

### Why RAG Wins

| Criteria | RAG | Fine-Tuned | Keywords | BERT QA | Graph |
|----------|-----|-----------|----------|---------|-------|
| Accuracy | ✅ High | ⚠️ Medium | ❌ Low | ⚠️ Medium | ⚠️ Medium |
| Flexibility | ✅ Excellent | ⚠️ Good | ❌ Poor | ⚠️ Good | ❌ Poor |
| Time to Deploy | ✅ Fast | ❌ Slow | ✅ Fast | ⚠️ Medium | ❌ Slow |
| Maintenance | ✅ Easy | ❌ Hard | ⚠️ Medium | ⚠️ Medium | ❌ Hard |

## Bonus 2: Data Insights

### Dataset Overview
- **Total Messages**: 3,349
- **Unique Members**: 10
- **Date Range**: Nov 8, 2024 - Nov 8, 2025 (1 year)
- **Avg Messages/Member**: 334.9

### Key Findings

#### 1. Excellent Data Quality
- No duplicates (0%)
- No missing fields
- No timestamp anomalies
- Consistent user IDs

#### 2. Unicode Encoding Issues
- **314 messages (9.4%)** contain non-ASCII characters
- Example: "Hans Müller" (umlaut: ü)
- **Recommendation**: Ensure UTF-8 encoding; normalize for search

#### 3. PII Exposure
**43 instances** of sensitive data in plain text:
- 33 phone numbers
- 8 email addresses
- 2 potential credit card numbers

**Recommendation**: Implement PII detection and masking

#### 4. Message Statistics
- Shortest: 9 chars
- Longest: 105 chars
- Average: 68 chars
- Very short messages may lack context

#### 5. Activity Distribution

| Member | Messages | % |
|--------|----------|---|
| Lily O'Sullivan | 365 | 10.9% |
| Thiago Monteiro | 361 | 10.8% |
| Fatima El-Tahir | 349 | 10.4% |
| Sophia Al-Farsi | 346 | 10.3% |
| Amina Van Den Berg | 342 | 10.2% |
| Vikram Desai | 335 | 10.0% |
| Layla Kawaguchi | 330 | 9.9% |
| Armand Dupont | 319 | 9.5% |
| Hans Müller | 314 | 9.4% |
| Lorenzo Cavalli | 288 | 8.6% |

**Observation**: Unusually uniform distribution suggests synthetic test data. Real usage typically follows Pareto principle (80/20).

### Production Recommendations

1. **PII**: Implement detection/masking pipeline
2. **Unicode**: Standardize UTF-8 encoding
3. **Validation**: Set minimum message length
4. **Monitoring**: Add anomaly detection

### Data Analysis

- Run analyze_data.py and it will generates `member_data_analysis.csv` with detailed   statistics.

## Acknowledgments

- Aurora for the take-home assignment
- Perplexity AI for the LLM API
- FastAPI for excellent framework
