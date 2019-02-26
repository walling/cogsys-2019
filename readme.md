# Implementing visio-acoustic associative memory model for fast mapping of word categories in small children

This repository contains the data, model code, and processing tools for the term paper. Paper online: <https://bjarke.me/2019-cogsys-paper>


## Setup

After cloning this repository, install the dependencies:

```bash
pipenv install
```


## Running

Run the four pre-experiments:

```bash
pipenv run python tools/3-experiment/1_pre_experiment.py
pipenv run python tools/3-experiment/2_pre_experiment.py
pipenv run python tools/3-experiment/3_pre_experiment.py
pipenv run python tools/3-experiment/4_pre_experiment.py
```

Analyze the results:

```bash
pipenv run python tools/3-experiment/5_main_experiment.py
```

The output can be found in the `data/3-experiment/` folder.


## Structure

This repository is organized according to the following file structure:

- `data/` — all data related to the research
  - `1-visual/` — visual stimuli data
    - `generated/`
      - `graph/` — some statistics for the generated data
      - `pattern/` — visualization of the generated data
    - `prototypes.csv` — the prototypes, not included in training set
    - `stimuli.csv` — the visual input stimuli
  - `2-acoustic/` — acoustic stimuli data
    - `original/`
      - `1-wordlist/` — complete and manually augmented word list
      - `2-audio/` — audio recordings of narrated words
    - `processed/`
      - `audio/` — automatically cut audio samples after running tools
      - `graph/` — some statistics for the cut audio samples after running tools
      - `wordlist/` — selected word list with 100 word senses (categories)
    - `stimuli.csv` — the acoustic input stimuli
  - `3-experiment/`
    - `graph/` — analyzed results and statistics for the experiments
    - `log/` — log protocols after running the experiments
- `lib/` — implementation code in Python
  - `data/` — code related to data processing and generation
  - `model/` — model implementation
- `paper/` — code for the paper (LaTeX)
  - `img/` — images used in the paper
- `tools/` — tools to automatically process data and run experiments
  - `1-visual/` — tools related to visual stimuli
    - `generate_patterns.py` — re-generate the visual dot patterns
  - `2-acoustic/` — tools related to acoustic stimuli
    - `1-wordlist/` — tools related to word lists
      - `process_wordlist.py` — re-generate the selected word list
    - `2-audio/` — tools related to audio recordings
      - `process_full_recordings.py` — (currently defunct) cut full recordings
      - `process_audio.py` — process audio files to generate stimuli data
  - `3-experiment/` — tools related to running experiments
    - `1_pre_experiment.py` — run 1st pre-experiment
    - `2_pre_experiment.py` — run 2nd pre-experiment
    - `3_pre_experiment.py` — run 3rd pre-experiment
    - `4_pre_experiment.py` — run 4th pre-experiment
    - `5_main_experiment.py` — (currently defunct) run main experiment
    - `6_analyze.py` — analyze the results of all experiments
    - `7_prepare_images.sh` — copy images used in the paper
- `Pipfile` — track dependencies to install
- `testall.py` — run all unit tests
