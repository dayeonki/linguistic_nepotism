# Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG

Authors: Dayeon Ki, Marine Carpuat, Paul McNamee, Daniel Khashabi, Eugene Yang, Dawn Lawrie, Kevin Duh

This repository contains the code and dataset for our paper **Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG**.

<div align="center">
<img src="https://github.com/user-attachments/assets/b3415a65-ccac-4468-a291-07602cb95509" style="width: 15px;" alt="code"> <b><a href=https://github.com/dayeonki/linguistic_nepotism>Code</a></b> | <img src="https://github.com/user-attachments/assets/fc2ca3c2-3e78-4ca4-a208-448c0a6c7068" style="width: 15px;" alt="paper"> <b><a href=>Paper</a></b>
</div>


## Abstract
Multilingual Retrieval-Augmented Generation (mRAG) systems enable language models to answer knowledge-intensive queries with citation-supported responses across languages. 
While such systems have been proposed, an open questions is whether the mixture of different document languages impacts generation and citation in unintended ways. 
To investigate, we introduce a controlled methodology using model internals to measure language preference while holding other factors such as document relevance constant. 
Across eight languages and six open-weight models, we find that models preferentially cite English sources when queries are in English, with this bias amplified for lower-resource languages and for documents positioned mid-context. 
Crucially, we find that models sometimes trade-off document relevance for language preference, indicating that citation choices are not always driven by informativeness alone. 
Our findings shed light on how language models leverage multilingual context and influence citation behavior.


## Quick Links
- [Overview](#overview)
- [Dataset/Languages/Models](#dataset/languages/models)
- [(1) Generate Reference Reports](#(1)-generate-reference-reports)
- [(2) Filter for Supported Statements](#(2)-filter-for-supported-statements)
- [(3) Next Token Analysis](#(3)-next-token-analysis)
- [Logit Lens Analysis](#logit-lens-analysis)
- [ContextCite Analysis](#contextcite-analysis)
- [Effect of Query Language](#effect-of-query-language)
- [Relevance vs. Language](#relevance-vs.-language)


## Overview
We show both (i) synthetic data generation (in blue) and (ii) measurement method (in pink). 
Given an English query and its K relevant evidence documents, we first translate the documents into multiple languages (Step 1). 
We then generate a reference citation-supported report for each query using the query and evidence documents (Step 2). 
The report consists of sentence-level statements, each paired with a single citation ID. 
For each report, we retain only statements that are verified (Step 3). 
Language preference is detected when the next token prediction accuracy for the correct citation ID decreases as the language of the cited document is varied (Step 4).

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e45bc30-bc51-4fff-8db2-ca6c0357e084" width="3000">
</p>



## Dataset/Languages/Models

- **Dataset**
  - ELI5 (Explain Like I'm Five): English long-form QA dataset (#: 270 queries)
  - Need all relevant documents (human-picked) for answering the question
  - Directory: `data/eli5` (`/single_token`: single token, `/multi_token`: multi token, `/comet`: COMET-QE scores)
  - Requirements for choosing the appropriate dataset:
    - Assume K relevant documents per query is already retrieved (post-retrieval stage)
    - More than 2 relevant documents per query
    - Currently no mRAG dataset with a query + parallel documents in diff. languages, rely on machine translation

- **Languages**
  - 8 non-English languages varying resource level, language family, writing script
  - Arabic, Bengali, Spanish, French, Korean, Russian, Swahili, Chinese
  <img width="600" height="260" alt="Screenshot 2025-09-13 at 1 01 32 PM" src="https://github.com/user-attachments/assets/8f4ce566-1a88-4175-87b1-0bfc506c6e0a" />

- **Models**
  - Only consider open-weights LLMs with access to internal representations
  - Models with relatively large context window to fit document context
  - LLaMA-3 8B, 70B, Qwen-3 8B, 14B, Gemma-3 27B, Aya23 8B
  <img width="600" height="148" alt="Screenshot 2025-09-13 at 1 02 03 PM" src="https://github.com/user-attachments/assets/42ebc0b3-dca3-40c1-8a82-99c40eab3d30" />

- **Pre-processing functions**
  - `data/google_translate.py`: Code to translate English source to target languages with Google Translate API
  - `data/comet_quality.py`: Code to measure COMET-QE scores between source sentences and MT
  - `data/avg_comet.py`: Code to get average COMET-QE scores
  - `data/hf_download.py`: Code to download HuggingFace datasets
  - `data/preprocess.py`: Code to preprocess to pre-defined JSONL format
  - `data/make_single_token.py`: Code to ensure the citation ID tokens are single digit number (1-9)
  - `data/data_statistics.py`: Code to get dataset statistics
  - `data/get_query_answerability.py`: Code to get correctness label (True/False) for query-report
      - Results are in `code/result_answerability`.



## (1) Generate Reference Reports
- **Description:** Given user query + relevant K docs, generate "reference" evidence-supported reports using a strong model
    - Strong model is determined by the model receiving most votes from SciArena benchmark (report generation/citation benchmark)
- `code/get_report.py`: Code to generate gold reports using GPT o3 (for English query + relevant docs)
- `code/get_report_l.py`: Code to generate gold reports using GPT o3 (for non-English query + relevant docs)

## (2) Filter for Supported Claims
- **Description:** Use LLM-as-judge ensemble to ask for supportness of each claim & Use NLI entailment classifier to predict entailment relationship.
- Run in the order of `get_claims`, then `get_ensemble`.
    - `code/get_claims.py`: Code for collecting claims (for English query + relevant docs)
    - `code/get_claims_l.py`: Code for collecting claims (for non-English query + relevant docs)
    - `code/get_ensemble.py`: Code for combining judgments (for English query + relevant docs)
    - `code/get_ensemble_l.py`: Code for combining judgments (for non-English query + relevant docs)
- Generated reports per language are in `code/report/`.

## (3) Next Token Analysis
- **Description:** Given each claim + citation bracket, get next token probability of the document ID token in 2 ways: (1) Natural probability; without any constraint, what the model naturally outputs, (2) Constrained probability; restrict prediction to only relevant document ID tokens (e.g., 1,2,3).
- **Hypothesis:** If the accuracy of correct predictions of the document ID as the next token is higher when the document is provided in English compared to its non-English counterparts, this indicates a language preference toward citing English documents.
     - `code/main_run.py`: Code for getting next token probabilities
- Results are in `result/`.


## Logit Lens Analysis

Does the model settle on its initial choice and persist with it or does it initially favor English documents before shifting toward the correct target language citation? 
To understand how the language preference unfolds during generation, we employ logit lens, which maps intermediate state representations of LLMs into the vocabulary space, enabling the ability to track a model's token prediction across layers. 
For each statement, we check whether the top-1 token prediction at a given layer is (1) the correct citation ID (target language document), (2) an incorrect ID (English document), or (3) not a valid citation token.

- `analysis/logitlens.py`: Code for running logit lens
    - `analysis/logitlens_low_gpu.py`: Optimized version of the same code, need for running LLaMA-3 70B model
    - `analysis/logitlens_eli5.sh`: Script for running for ELI5 dataset
    - Reults are in `analysis/res_logitlens`.
- `analysis/visualize_logit_lens.py`: Code to generate visualizations for logit lens per layer
    - Results are in `analysis/vis_logitlens`.


## ContextCite Analysis

While citation accuracy explains corroborative attribution, which identifies sources that support a statement, we further analyze contributive attribution, which captures sources that causally influence the model’s generation.

- Use ContextCite (https://gradientscience.org/contextcite/) to measure internal-based attribution for citations. Original code is in `analysis/context-cite/`, directly cloned from original github.
    - `analysis/context-cite/contextcite_eli5.sh`: Script for running ContextCite
- `analysis/measure_contextcite.py`: Code to measure Hit @1,3 and Score @1,3 fpr each dataset, model, language.
    - Hit @k: Attributed text is part of the correct document (higher = more attribution to correct doc)
    - Score @k: ContextCite attribution score for the attributed part (when it's part of the correct document) (higher = more confidence)
- Results are in `analysis/res_contextcite/`.


## Effect of Query Language

To examine whether the English preference pattern persists when the query itself is in a language other than English, we translate the query and evidence documents into the query language and run 4 variants.

- `code/main_run_qlang_1.py`, `code/main_run_qlang_2.py`: Code for changing the query+claim language to non-English
    - Translated English query to each non-English target languages, generate gold report for each language using query + relevant documents in non-English.
    - Compare 4 variants varying in extent of English document in context.
        - `main_run_qlang_1.py` is for running All XX, All XX + correct En.
        - `main_run_qlang_2.py` is for running All En, All En + correct XX.
    - **Hypothesis:**
        - All En, All XX + correct En best → regardless of query language, having correct document in English is important.
        - All XX, All En + correct XX best → matching the language of the correct document to query language is important.
    - Results are in `code/result_qlang_1` and `code/result_qlang_2`.


## Relevance vs. Language

Between relevance and language, which exerts a stronger influence on model citation behavior?
For this experiment, we use a different RAG dataset, MIRACL, which is composed of short-form and long-form questions. We only use the English subset.
- Relevant + Irrelevant (distractor) documents given
- Directory: `data/miracl` (`/n1`: subset with 1 relevant doc, `/n1_comet`: COMET-QE scores)

- `code/main_run_miracl_n1.py`: Code for running for MIRACL, where documents in different languages contain non-overlapping content
    - Use 1 relevant + 1 distractor (irrelevant) document.
    - Two conditions are tested: (1) XX (relevant) - English (irrelevant) and (2) English (relevant) - XX (irrelevant)
        - Condition 1: If accuracy drops below baseline, indicates stronger language preference toward English → model is easily distracted by English irrelevant document.
        - Condition 2: If accuracy rises above baseline, indicates English preference → non-English distractor is less effective, making it easier to choose the English relevant document.
    - Baseline: English (relevant) - English (irrelevant)
    - Results are in `code/result_miracl_n1`.



## Citation
```
TBD
```
