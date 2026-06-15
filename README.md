

# Purpose 
This is a repo using code from Olmix (https://arxiv.org/abs/2602.12237, https://github.com/allenai/olmix)  
The pipeline looks like this:  
First, input paths to our nemotron .bin files, calculate tokens, and generate suggested training mixtures using olmix  
Second, use the training mixtures to generate slurm scripts for small scale training runs  
Thirdly, evaluate the different mixtures using BPB  
Finally train regression models on the mixtures and get a final suggested mix  

# Steps for computing optimal mixture  
Add all datasets that we are interested in, in the config.yaml file  
training data used as input can be found here: /scratch/project_465002530/preprocessed/oellm-v1-256k/catalogue/  

# Command to run to generate mixtures directly  
./generate_mixes.sh -- this will read the config.yaml and use olmix to generate suggested mixes, it will also create the launch_all_swarms.sh script  
the suggested mixes will be put into the /mixes directory  
Per the Olmix paper we generate unconstrained swarms to explore the mixture space, and later constrain the specific mixture to a realistic mix that takes data constraints into account

# Command to start run
./launch_all_swarms.sh - starts an array job of all mixes

# TODO  
test the fitting portion of a run based on metrics.csv, ratios.csv files - or run elsewhere
check that BPB will be added to oellm-cli in the near future, otherwise will have to compute bpb elsewhere to get the metrics
ensure all the datasets are available


# Datasets

### English
Datasets representing primarily natural language English text.

| Dataset | Path on LUMI | Scale |
| :--- | :--- | :--- |
| **DCLM-baseline** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/mlfoundations/dclm-baseline-1.0-10p-sample/dclm-10p-sample` | 10p |
| **Nemotron-CC-v1** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/20p-sample/high-actual-20p`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/20p-sample/medium-high-actual-20p`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/nemotron-cc/1.0/20p-sample/medium-actual-5p` | 20p |
| **HPLT4-CC** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/hplt/4.0/pre-clean/eng_Latn/hplt-4.0-pre-clean-eng_Latn-cc_text_document` | Full |
| **HPLT4-IA** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/hplt/4.0/pre-clean/eng_Latn/hplt-4.0-pre-clean-eng_Latn-ia_text_document` | Full |
| **HPLT4-AB** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/hplt/4.0/pre-clean/eng_Latn/hplt-4.0-pre-clean-eng_Latn-ab_text_document` | Full |
| **FinePDFs** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-10p-sample/eng_Latn` | 10p |
| **FinePDFs-edu** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceFW/finepdfs-edu-10p-sample/eng_Latn` | 10p |
| **FinePhrase** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/HuggingFaceFW/finephrase/finephrase_decomp` | Full |
| **olmo-mix-1124-wiki** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/allenai/OLMO-mix-10p-sample/Wiki-10p-sample` | 10p |
| **olmo-mix-1124-arxiv** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/allenai/OLMO-mix-10p-sample/arxiv-10p-sample` | 10p |
| **olmo-mix-1124-pes2o** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/allenai/OLMO-mix-10p-sample/pes2o-10p-sample` | 10p |
| **Nemotron-Pretraining-v1** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/nvidia/nemotron-pretraining-specialized-v1/nemotron-pretraining-specialized-v1` | Full |
| **Nemotron-Pretraining-v1.1** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/nvidia/nemotron-pretraining-specialized-v1.1/nemotron-pretraining-specialized-v1.1` | Full |
| **Nemotron-MIND** | `[PATH NEEDED]` | TBD |
| **MixtureVitae-v1** | `[PATH NEEDED]` | TBD |

---

### Code
Datasets representing primarily text in programming languages.

| Dataset | Path on LUMI | Scale |
| :--- | :--- | :--- |
| **Starcoder** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/starcoder-10p-sample` | 10p |
| **Swallow Code 2.0** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/swallow-code-v2/sampled-10p/stage5-auto-format/python-medium` | 10p |
| **Dolma3-dolmino-mix (Code)** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/allenai/dolma3_dolmino_mix-100B-1125/sampled-10p/stack_edu-fim/...` | 10p |
| **Stack 1.2** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/the-stack/1.2/data/sampled-10p/00_ALL_LANGUAGES/all-languages` | 10p |
| **common-pile Stack v2** | `/scratch/project_465002530/preprocessed/oellm-v1-256k/common-pile` | Full |
| **common-pile Stack v2-edu** | `/scratch/project_465002530/preprocessed/oellm-v1-256k/common-pile` | Full |

---

### Math
Datasets representing math content.

| Dataset | Path on LUMI | Scale |
| :--- | :--- | :--- |
| **FineMath** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/HuggingFaceTB/finemath-full/finemath-4plus/finemath-4plus_text_document` | Full |
| **MegaMath** | `/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-code/megamath-code_text_document`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-qa/megamath-qa_text_document`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-text-code-block/megamath-text-code-block_text_document`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-translated-code/megamath-translated-code_text_document`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-web/megamath-web_text_document`<br>`/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/megamath-web-pro/megamath-web-pro_text_document` | Full |
| **Swallow Math 2.0** | `[PATH NEEDED]` | TBD |
| **Dolma3-dolmino-mix (Math)**| `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/allenai/dolma3_dolmino_mix-100B-1125/sampled-10p/cranemath/...` | 10p |
| **OpenWebMath** | `[PATH NEEDED]` | TBD |

