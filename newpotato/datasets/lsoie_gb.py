import argparse
import logging
import time

logging.basicConfig(
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # To print logs to the console
        logging.FileHandler("logs_mapping.log", mode="w"),  # To save logs to a file
    ],
)

from collections import defaultdict

import networkx as nx
from rich.console import Console
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

from newpotato.datatypes import Triplet
from newpotato.hitl import HITLManager

console = Console()


def load_and_map_lsoie(input_file, extractor):
    with open(input_file) as stream:
        total, skipped = 0, 0
        sent_cnt = 0
        last_sent = ""
        for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
            total += 1
            words = [t[1] for t in sen]
            sentence = " ".join(words)
            text_to_graph = extractor.get_graphs(sentence)

            # count sentences for statistics
            if sentence != last_sent:
                last_sent = sentence
                sent_cnt += 1
            if len(text_to_graph) > 1:
                logging.error(f"sentence split into two: {words}")
                # logging.error(f"{text_to_graph=}")
                logging.error("skipping sentence")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue

            # rare problems with sentence modifications need this check
            try:
                graph = text_to_graph[sentence]["main_edge"]
            except:
                logging.error(f"sentence not found after parsing")
                logging.error(f"{sen_idx=}, {sentence=}")
                logging.error("skipping sentence")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue
            logging.debug(f"{sen_idx=},{sentence=}, {graph=}")

            arg_dict = defaultdict(list)
            pred = []
            for i, tok in enumerate(sen):
                label = tok[7].split("-")[0]
                if label == "O":
                    continue
                elif label == "P":
                    pred.append(i)
                    continue
                arg_dict[label].append(i)

            pred = tuple(pred)
            args = [
                tuple(indices)
                for label, indices in sorted(
                    arg_dict.items(), key=lambda i: int(i[0][1:])
                )
            ]
            logging.debug(f"{pred=}, {args=}")

            triplet = Triplet(pred, args, toks=words)
            try:
                mapped_triplet = extractor.map_triplet(triplet, sentence, strict=True)
                # mapped_triplet is of class newpotato.extractors.graphbrain_extractor_PC.GraphbrainMappedTriplet
                # mapped_triplet has ".variables" attribute
            except (KeyError, nx.exception.NetworkXPointlessConcept):
                logging.error(f"error mapping triplet: {triplet=}, {words=}")
                logging.error("skipping triplet")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue

            if not mapped_triplet:
                logging.error(f"error mapping triplet: {triplet=}, {words=}")
                logging.error("skipping triplet")
                skipped += 1
                logging.error(f"{skipped=}, {total=}")
                continue

            logging.debug(f"{mapped_triplet.variables=}")
            print(f"Sentence count: {sent_cnt}")
            yield sentence, mapped_triplet

            # mapped_triplet.variables:
            # {'REL': dissolved/Pd.xso.<f-----/en,
            # 'ARG0': ('s/Bp.am/en (the/Md/en country/Cc.s/en) parliament/Cc.s/en),
            # 'ARG1': (+/B.mm/. ('s/Bp.am/en thailand/Cp.s/en (+/B.am/. prime/Cp.s/en minister/Cp.s/en)) (+/B.am/. yingluck/Cp.s/en shinawatra/Cp.s/en)),
            # 'ARG2': (earlier/M=/en today/Cc.s/en),
            # 'ARG3': formally/M/en}


def load_lsoie_to_hitl(input_file, hitl):
    for sen, triplet in load_and_map_lsoie(input_file, hitl.extractor):
        hitl.store_triplet(sen, triplet, True)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-s", "--state_file", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.getLogger().setLevel(logging.WARNING)
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("initializing HITL session")
    start = time.time()
    hitl = HITLManager(extractor_type="graphbrain")
    console.print(f"loading LSOIE data from {args.input_file}")
    load_lsoie_to_hitl(args.input_file, hitl)
    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)
    duration_min = (time.time() - start) / 60
    print(f"Duration of parsing and mapping process: {duration_min:.2f} minutes")


if __name__ == "__main__":
    main()
