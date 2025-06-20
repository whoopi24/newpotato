import argparse
import json
import logging

logging.basicConfig(
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eval_wire57_sent.log", mode="w"),
    ],
)

from rich.console import Console

from newpotato.extractors.graphbrain_extractor import combine_triplets
from newpotato.hitl import HITLManager
from newpotato.modifications.oie_evaluation import information_extraction

console = Console()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    logging.getLogger().setLevel(logging.WARNING)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("initializing HITL session")
    hitl = HITLManager(extractor_type="graphbrain")

    if hitl.extractor.patterns is None:
        console.print("[bold cyan]Enter path to patterns file:[/bold cyan]")
        fn = input("> ")
        console.print("[bold cyan]Enter number of patterns:[/bold cyan]")
        p_cnt = input("> ")
        console.print(
            "[bold cyan]Enter maximal number of extractions per decomposed hyperedge:[/bold cyan]"
        )
        max_extr = input("> ")

        try:
            hitl.extractor.load_patterns(fn, N=int(p_cnt))
        except FileNotFoundError:
            console.print(f"[bold red]No such file or directory: {fn}[/bold red]")

    # Directories & files
    DIR = "WiRe57"
    EXTR_MANUAL = "WiRe57_343-manual-oie.json"
    EXTR_AFTER = fn.split(".")[0] + "_" + p_cnt + "_" + max_extr + ".json"

    manual = json.load(open("{}/{}".format(DIR, EXTR_MANUAL)))
    total, skipped = 0, 0
    extractions = {}

    for key in manual:
        for case in manual[key]:
            total += 1
            sentence = case["sent"]
            sen_idx = case["id"]
            skip = False

            # text parsing for new sentence
            text_to_graph = hitl.extractor.get_graphs(sentence, doc_id=str(sen_idx))
            if len(text_to_graph) > 1:
                logging.error(f"sentence split into two")
                logging.error(f"{sen_idx=}, {text_to_graph=}")
                logging.error("skipping")
                skipped += 1
                skip = True
                logging.error(f"{skipped=}, {total=}")

            logging.debug(f"{sentence=}, {text_to_graph=}")

            # generating triplets after conjunction decomposition (information_extraction)
            if not skip:
                try:
                    # rare problems with sentence modifications need this check
                    graph = text_to_graph[sentence]["main_edge"]
                except KeyError:
                    logging.error(f"sentence not found after parsing")
                    logging.error(f"{sen_idx=}, {sentence=}")
                    logging.error("skipping")
                    skipped += 1
                    logging.error(f"{skipped=}, {total=}")
                else:
                    atom2word = text_to_graph[sentence]["atom2word"]
                    logging.info("-" * 100)
                    logging.info(f"START of information extraction for {sen_idx=}:")
                    logging.info(f"{sen_idx=}, {sentence=}")
                    logging.info(f"{graph=}")
                    logging.info(f"{atom2word=}")
                    information_extraction(
                        extractions,
                        graph,
                        sen_idx,
                        atom2word,
                        hitl.extractor.patterns,
                        max_extr=max_extr,
                    )

    # take biggest set of overlapping matches per sentence and
    # add predictions to the results of the other OIE systems
    for k, v in extractions.items():
        newv = combine_triplets(v)
        extractions[k].append(newv[0])

    for k, v in extractions.items():
        print(k, v)
        extractions[k].append(v[0])

    # save predictions to json file
    with open("{}/{}".format(DIR, EXTR_AFTER), "w", encoding="utf-8") as f:
        json.dump(extractions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
