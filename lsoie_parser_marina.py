import pickle
from collections import defaultdict

import stanza
from stanza.utils.conll import CoNLL
from tuw_nlp.text.utils import gen_tsv_sens

# stanza.download("en")


def main(input_stream, output_stream, outfile):
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )
    stream = open(input_stream, "r", encoding="utf-8")
    output = open(output_stream, "w", encoding="utf-8")
    mydict = {}
    for sen_idx, sen in enumerate(gen_tsv_sens(stream)):
        print(f"processing sentence {sen_idx}")
        # log = open(f"out/test{sen_idx}.log", "w")
        parsed_doc = nlp(" ".join(t[1] for t in sen))
        parsed_sen = parsed_doc.sentences[0]
        args = defaultdict(list)
        pred = []
        for i, tok in enumerate(sen):
            label = tok[7].split("-")[0]
            if label == "O":
                continue
            elif label == "P":
                pred.append(i + 1)
                continue
            args[label].append(i + 1)

        # log.write(f"test{sen_idx} pred: {pred}, args: {args}\n")
        # CoNLL.write_doc2conll(parsed_doc, f"out/test{sen_idx}.conll")
        # log.write(f"wrote parse to test{sen_idx}.conll\n")

        output.write(f"test{sen_idx} sent: {parsed_sen.text}\n")
        output.write(f"test{sen_idx} pred: {pred}, args: {args}\n")

        mydict[f"test{sen_idx}"] = parsed_sen

    with open(outfile, "wb") as f:
        pickle.dump(mydict, f)

    # with open(outfile, "rb") as f:
    #     loaded_dict = pickle.load(f)


if __name__ == "__main__":
    input_stream = "lsoie_wiki_dev.conll"
    output_stream = "output.log"
    outfile = "output.pkl"
    main(input_stream, output_stream, outfile)
