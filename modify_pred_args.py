import 


def modify_pred_args(sent, sen_graph):    

    # argwords = defaultdict()
    # predwords = []

    # predwords.append(tok[1].lower())
    # argwords[tok[1].lower()] = label


    # get tokens of original sentence
    all_toks_org = tuple(sent)
    # compare tokens of original sentence and spacy sentence
    all_toks = tuple(tok.text for tok in sen_graph["spacy_sentence"])
    words_to_i = defaultdict(set)
    for i, word in enumerate(all_toks):
        words_to_i[word.lower()].add(i)
    print(words_to_i)

    if len(all_toks_org) != len(all_toks):
        print("positions might have changed")

        add = 0
        idx_chg = []
        for i in range(0, len(all_toks_org)):
            j = i + add
            t1 = all_toks_org[i]
            t2 = all_toks[j]
            # print(t1, t2)
            if t1 == t2:
                idx_chg.append(0)
                continue
            elif t1 == all_toks[j + 1]:
                # print(t1, all_toks[j + 1])
                add += 1
                idx_chg.append(1)
            else:
                continue

        indices = [i for i, x in enumerate(idx_chg) if x == 1]
        # print(indices)
        for i in indices:
            add = 1
            # modify pred
            for pi in range(0, len(predlist)):
                p = predlist[pi]
                if p >= i:
                    predlist[pi] = p + 1

            # modify args
            for ai in range(0, len(args)):
                arglist = args[ai]
                add_idx = 0
                for aii in range(0, len(arglist)):
                    aii += add_idx
                    a = arglist[aii]
                    if a == i:
                        arglist.insert(aii, a)
                        arglist[aii + 1] = a + 1
                        add_idx = 1
                    elif a > i:
                        arglist[aii] = a + 1
                args[ai] = arglist

            add += 1

    pred = tuple(predlist)
    newargs = args
    args = []
    for l in newargs:
        args.append(tuple(l))
    print(pred, args)