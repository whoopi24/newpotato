# `newpotato`: Open Information Extraction with Semantic Hypergraphs

This fork focused on supervised rule learning for OIE with Semantic Hypergraphs (SHs) by using main functionalities of `graphbrain` (https://github.com/graphbrain/graphbrain), developed by Menezes and Roth. In the original GitHub repository (https://github.com/adaamko/newpotato) you can find general information about `newpotato`.

## How to run the program

After you have successfully cloned this repository, install the program as described at the end of this ReadMe file and run `add_packages_newpotato.sh` to install specific package versions and additional packages. The best way to open this program is to use a docker container in VSCode.

Below is a list of commands which are useful to run the program:

1. Activate the parser: `uvicorn api.graphbrain_parser:app --port 7277` (always needs to be done first)

2. Parse LSOIE (train) data into SH structure, map to triplets and store in state file: `python newpotato/datasets/lsoie_gb.py -i lsoie_wiki_train.conll -s train.hitl`

3. Rule Learning:
    * Start graphbrain extractor: `python newpotato/term_client.py -x graphbrain`

    Add if necessary:
	* Load parsed sentences and triplets: `-l train.hitl`
	* Run rule learning: `-r`
	* Set number of rules, default=20: `-rc 100`
	* Save rules in a file (optional): `-p patterns.txt`

4. Evaluate an LSOIE dataset:
    * Start graphbrain extractor: `python newpotato/term_client.py -x graphbrain`

    Add test data set (two options - LSOIE data loader not needed):
	* Load original data set: `-e lsoie_wiki_dev.conll`
    * Load parsed sentences: `-l lsoie_wiki_dev.hitl`

    Add pattern set:
    * Load existing patterns generated during rule learning process with `-lp patterns.txt` or provide file name with manual input in console
	* Decide how many patterns you want to use for the extraction, default=20: `-rc 10`

5. Evaluate the WiRe57 dataset:
	* python eval_wire57_sent.py (provide patterns file, N, max_extr in console)
	* python ./WiRe57/wire_scorer.py

Control verbosity of the above commands with `-v` and `-d`.

Alternatively, you can open the interactive terminal client of `newpotato` with
`python newpotato/term_client.py -x graphbrain -i`.

## Details about `graphbrain`

In this section, we outline the key functionalities of `graphbrain` (https://github.com/graphbrain/graphbrain) that were used in this program. `graphbrain` is an open-source Python library designed for constructing, modifying, and analyzing hypergraphs. It intends to facilitate automated meaning extraction and text comprehension while supporting knowledge exploration and inference.

A fundamental component is `create_parser` (https://github.com/graphbrain/graphbrain/blob/master/graphbrain/parsers), which converts sentences into hyperedge structures. Since our supervised rule learning approach focuses on semantic hyperedges, we rely on several functions from the `Hyperedge` class (https://github.com/graphbrain/graphbrain/blob/master/graphbrain/hyperedge.pyx). These include:
* `hedge()`, which constructs a Hyperedge object from a list, tuple, or string,
* `mtype()`, which determines the main type of an edge after type inference,
* `argroles()`, which retrieves the argument roles of an edge, and
* `atoms()`, which lists all unique atoms contained in the edge.

Additionally, we make extensive use of functions related to the var functional pattern, such as `apply_variables()`, `contains_variable()`, and `all_variables()`(https://github.com/graphbrain/graphbrain/blob/master/graphbrain/patterns/variables.pyx). For the triplet extraction, we leverage the conjunction decomposition functionality (https://github.com/graphbrain/graphbrain/blob/master/graphbrain/utils/conjunctions.py), which breaks down complex hyperedges into their individual parts. Finally, we apply the `Matcher` class, specifically the `match_pattern(edge,pattern)` function, to match the decomposed edges with our learned rules. As previously mentioned, this section does not provide a complete list of all graphbrain features used in our work, but highlights the most relevant ones.

## Details about `newpotato`

In the following, we highlight the key components of `newpotato`. First, the class `NPTerminalClient` (https://github.com/adaamko/newpotato/blob/main/newpotato/term_client.py) provides an interactive UI and supports user interaction through the terminal. Its core functionalities outline the structure of the system, directing to the appropriate modules in other files. The most important features of our work, such as the learning process and the evaluation of rules, can be found in the functions `print_rules()` and `evaluate()`.

The client is connected to the `HITLManager` class (https://github.com/adaamko/newpotato/blob/main/newpotato/hitl.py), which manages the HITL process and stores the parsed graphs as well as the annotated or extracted triplets in a temporary state file. It is further linked to a specific `Extractor` class (https://github.com/adaamko/newpotato/blob/main/newpotato/extractors/extractor.py) - in our case, the `GraphbrainExtractor` (https://github.com/adaamko/newpotato/blob/main/newpotato/extractors/graphbrain_extractor.py). This class forms the core of our work with our implementation of a supervised OIE method. Triplets mapped during the rule learning process are of type `GraphbrainMappedTriplet`, a subclass of the datatype `Triplet` (https://github.com/adaamko/newpotato/blob/main/newpotato/datatypes.py). Text parsing is handled between the `GraphbrainParserClient` and the `GraphbrainParser` (https://github.com/adaamko/newpotato/blob/main/newpotato/extractors/graphbrain_parser.py; https://github.com/adaamko/newpotato/blob/main/api/graphbrain_parser.py). A parsed sentence is from type `GraphParse`, which is a Python dictionary object with specific entries. It is necessary to specify the type of extractor when starting the program, since `newpotato` also supports another type of extractor, called `GraphBasedExtractor()` (https://github.com/adaamko/newpotato/blob/main/newpotato/extractors/graph_extractor.py), which focuses on Universal Dependencies. 

In `newpotato`, `spaCy` is required for text parsing, in particular for coreference resolution. The following three `spaCy` models for English text are necessary to run `newpotato` properly: 
* `en_core_web_trf`,
* `en_core_web_lg`, and
* `en_core_web_sm`.

Finally, it is important to note that `newpotato` is still under development and should be considered an experimental system.


## Installation

### Pre-requisites
- Python 3.11
- Docker

### Development
#### Backend
The backend functionalities are implemented as a REST API built with FastAPI. The code is located in the `api/` directory.

#### Frontend
The frontend is built using Streamlit and is located in the `frontend/` directory.

#### Core Package
The core functionalities are in the `newpotato` package.

#### Running Locally
To run the project locally for development:

1. Install the dependencies:
    ```bash
    pip install -e .
    ```
2. Start the FastAPI server:
    ```bash
    uvicorn api.main:app --reload
    ```
3. Start the Streamlit app:
    ```bash
    streamlit run frontend/app.py
    ```

#### Running with Docker
The devlopment environment can be also used from .devcontainer in VSCode. It can be found under the .devcontainer folder.

### Production
The production environment is built using Docker Compose. To run the project in production:
```bash
docker-compose -f deploy/docker-compose.yml up
```