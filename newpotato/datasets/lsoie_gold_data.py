import argparse
import json
import logging
from collections import defaultdict

from rich.console import Console
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

console = Console()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-s", "--output_file", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # create dictionary with correctly ordered and named keys
    name_mapping = {
        "A0": "arg1",
        "P": "rel",
        "A1": "arg2",
        "A2": "arg3",
        "A3": "arg4",
        "A4": "arg5",
        "A5": "arg6",
    }

    if args.output_file:
        fn = args.output_file
    else:
        console.print(
            "[bold cyan]Enter path to save LSOIE gold data as .json file:[/bold cyan]"
        )
        fn = input("> ")

    with open(args.input_file) as stream:
        total, skipped = 0, 0
        last_sent = ""
        gold_list = list()
        for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
            total += 1
            words = [t[1] for t in sen]
            sentence = " ".join(words)
            # avoid new entry for same sentence
            # remark: sen_idx skipped -> from 0 to xy with gaps
            if sentence == last_sent:
                sen_idx = last_id
            else:
                if total > 1:
                    # combine arg3-arg6 to "arg3+"
                    keys_to_combine = ["arg3", "arg4", "arg5", "arg6"]
                    tuples_list = []
                    for tup in gold_dict["tuples"]:
                        arg3to6 = []
                        for key in keys_to_combine:
                            if key in tup:
                                arg3to6.append(tup[key])
                                # remove original key
                                del tup[key]
                        # add arg2 if missing
                        if "arg2" not in tup:
                            tup["arg2"] = {"words": [], "words_indexes": []}
                        # add combined keys or empty list
                        tup["arg3+"] = arg3to6
                        tuples_list.append(tup)

                    # Add to the list under a new key
                    gold_dict["tuples"] = tuples_list
                    gold_list.append(gold_dict)
                last_sent = sentence
                last_id = sen_idx
                gold_dict = {
                    "id": str(sen_idx),
                    "sent": sentence,
                    "tokens": words,
                    "tuples": list(),
                }

            temp = {k: defaultdict(list) for k in name_mapping.keys()}
            for i, tok in enumerate(sen):
                # 'text': '(The second)/(The second danger to a year of relatively healthy global economic growth)',
                # 'words': ['The', 'second'],
                # 'words_indexes': [0, 1],
                label = tok[7].split("-")[0]
                if label == "O":
                    continue
                temp[label]["words"].append(tok[1])
                temp[label]["words_indexes"].append(i)

            # TODO: add "text" entry with joining "words" list to string
            # for key in temp.keys():
            #     joined_text = " ".join(temp[key]["words"])
            #     temp[key]["text"] = joined_text

            args_dict = {
                name_mapping.get(key, key): value
                for key, value in temp.items()
                if value
            }
            logging.debug(f"{sentence=}, {args_dict=}")

            gold_dict["tuples"].append(args_dict)

        # save dictionary to json file
        save_dict = {args.input_file: gold_list}
        with open(fn, "w") as f:
            f.write(json.dumps(save_dict))


if __name__ == "__main__":
    main()
