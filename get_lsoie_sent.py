import networkx as nx
from rich.console import Console
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens


def main(input_file):
    output_file = input_file.split(".")[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        with open(input_file) as stream:
            last_sent = ""
            for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
                words = [t[1] for t in sen]
                sentence = " ".join(words)
                if sentence == last_sent:
                    continue
                f.write(f"{sen_idx}\t{sentence}\n")
                last_sent = sentence


if __name__ == "__main__":
    input_file = "lsoie_wiki_dev_small.conll"
    main(input_file)
