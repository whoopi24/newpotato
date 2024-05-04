import pickle
from collections import defaultdict

import stanza
from tuw_nlp.text.utils import gen_tsv_sens

# stanza.download("en")


def main(input_stream, outfile):
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )
    stream = open(input_stream, "r", encoding="utf-8")
    mydict = {}
    iter = 0
    for sen_idx, sen in enumerate(gen_tsv_sens(stream)):
        iter += 1
        print(f"processing sentence {sen_idx}")
        parsed_doc = nlp(" ".join(t[1] for t in sen))
        parsed_sen = parsed_doc.sentences[0]
        args = defaultdict(list)
        sent = []
        pred = []
        for i, tok in enumerate(sen):
            sent.append(tok[1])
            label = tok[7].split("-")[0]
            if label == "O":
                continue
            elif label == "P":
                pred.append(i)
                continue
            args[label].append(i)

        mydict[sen_idx] = {"sent": sent, "pred": pred, "args": args}

        if iter == 20:
            break

    with open(outfile, "wb") as f:
        pickle.dump(mydict, f)
        print("dictionary saved successfully to file")

    # with open(outfile, "rb") as f:
    #     loaded_dict = pickle.load(f)


if __name__ == "__main__":
    input_stream = "lsoie_wiki_dev.conll"
    outfile = "sample.pkl"
    main(input_stream, outfile)
