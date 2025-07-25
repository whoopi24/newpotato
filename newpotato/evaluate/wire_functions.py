# functions from wirer_scorer.py
# https://github.com/rali-udem/WiRe57/blob/master/code/wire_scorer.py

# TODO:
# - Implement half points for part-misplaced words.
# - Deal with prepositions possibly being the first token of an arg, especially for arg2.
#   > It's fully ok for "any" prep to be last word of ref_rel or first_word of pred_arg

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


def eval_system(gold, predictions):
    results = {}
    # Get a manytuples-to-manytuples match-score for each sentence,
    # then gather the scores across sentences and compute the weighted-average
    for s, reference_tuples in gold.items():
        predicted_tuples = predictions.get(s, [])
        logger.info(f"### sen_idx={s} ###")
        results[s] = sentence_match(reference_tuples, predicted_tuples)
        logger.debug(f"{results[s]=}")
        # returns dict with
        # 'precision': [1.0, 8], 'recall': [0.6, 4],
        # 'precision_of_matches': [1.0], 'recall_of_matches': [0.6],
        # 'exact_match_precision': [0, 8], 'exact_match_recall': [0, 4]

    prec_num, prec_denom = 0, 0
    rec_num, rec_denom = 0, 0
    exactmatches_precnum, exactmatches_precdenom = 0, 0
    exactmatches_recnum, exactmatches_recdenom = 0, 0
    tot_prec_of_matches, tot_rec_of_matches = 0, 0
    for s in results.values():
        prec_num += s["precision"][0]
        prec_denom += s["precision"][1]
        rec_num += s["recall"][0]
        rec_denom += s["recall"][1]
        exactmatches_precnum += s["exact_match_precision"][0]
        exactmatches_precdenom += s["exact_match_precision"][1]
        exactmatches_recnum += s["exact_match_recall"][0]
        exactmatches_recdenom += s["exact_match_recall"][1]
        tot_prec_of_matches += sum(s["precision_of_matches"])
        tot_rec_of_matches += sum(s["recall_of_matches"])
    if prec_denom != exactmatches_precdenom:
        logger.debug(f"{prec_denom=}, {exactmatches_precdenom=}")
    if rec_denom != exactmatches_recdenom:
        logger.debug(f"{rec_denom=}, {exactmatches_recdenom=}")
    precision_scores = [v for s in results.values() for v in s["precision_of_matches"]]
    recall_scores = [v for s in results.values() for v in s["recall_of_matches"]]
    raw_match_scores = [precision_scores, recall_scores]
    matches = len(precision_scores)
    metrics = {
        "precision": prec_num / prec_denom,
        "recall": rec_num / rec_denom,
        "non-matches": exactmatches_precdenom - matches,
        "matches": matches,
        "precision_of_matches": tot_prec_of_matches / matches,
        "recall_of_matches": tot_rec_of_matches / matches,
        "exactmatches_precision": [exactmatches_precnum, exactmatches_precdenom],
        "exactmatches_recall": [exactmatches_recnum, exactmatches_recdenom],
    }
    return metrics, raw_match_scores


def f1(prec, rec):
    try:
        return 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        return 0


def sentence_match(gold, predicted):
    """For a given sentence, compute tuple-tuple matching scores, and gather them
    at the sentence level. Return scoring metrics."""
    score, maximum_score = 0, len(gold)
    exact_match_scores = [[None for _ in predicted] for __ in gold]
    scores = [[None for _ in predicted] for __ in gold]
    for i, gt in enumerate(gold):
        for j, pt in enumerate(predicted):
            exact_match_scores[i][j] = tuple_exact_match(pt, gt)
            scores[i][j] = tuple_match(pt, gt)  # this is a pair [prec,rec] or False

            logger.info(f"Matching gold tuple:")
            for key in gt:
                if key == "arg3+":
                    logger.info("arg3+:")
                    for arg in gt["arg3+"]:
                        logger.info(f"    - {' '.join(arg['words'])}")
                elif key in ["arg1", "rel", "arg2"]:
                    logger.info(f"{key}: {' '.join(gt[key]['words'])}")

            logger.info(f"\nwith predicted tuple:")
            for key, value in pt.items():
                if key == "arg3+":
                    logger.info("arg3+:")
                    for arg in value:
                        logger.info(f"    - {arg}")
                elif key in ["arg1", "rel", "arg2"]:
                    logger.info(f"{key}: {value}")
            logger.info("-" * 30)
            logger.info(f"exact match score: {exact_match_scores[i][j]}")
            logger.info(f"partial match score: {scores[i][j]}")
            logger.info("-" * 30)

    # logger.info(f"exact match scores: {exact_match_scores}")
    # logger.info(f"partial match scores: {scores}")
    scoring_metrics = aggregate_scores_greedily(scores)
    exact_match_summary = aggregate_exact_matches(exact_match_scores)
    scoring_metrics["exact_match_precision"] = exact_match_summary["precision"]
    scoring_metrics["exact_match_recall"] = exact_match_summary["recall"]
    return scoring_metrics


def aggregate_scores_greedily(scores):
    # Greedy match: pick the prediction/gold match with the best f1 and exclude
    # them both, until nothing left matches. Each input square is a [prec, rec]
    # pair. Returns precision and recall as score-and-denominator pairs.
    matches = []
    while True:
        max_s = 0
        gold, pred = None, None
        for i, gold_ss in enumerate(scores):
            logger.debug(f"{i=}, {gold_ss=}")
            if i in [m[0] for m in matches]:
                logger.debug(f"row already taken")
                # Those are already taken rows
                continue
            for j, pred_s in enumerate(scores[i]):
                logger.debug(f"{j=}, {pred_s=}")
                if j in [m[1] for m in matches]:
                    logger.debug(f"col already taken")
                    # Those are used columns
                    continue
                if pred_s and f1(*pred_s) > max_s:
                    max_s = f1(*pred_s)
                    gold = i
                    pred = j
                    logger.debug(f"new best score: {max_s}")
                    logger.debug(f"best gold: {i}, best pred: {j}")
        if max_s == 0:
            break
        # save indices of best match for this sentence e.g. [[2, 4]]
        matches.append([gold, pred])
    # Now that matches are determined, compute final scores.
    prec_scores = [scores[i][j][0] for i, j in matches]
    rec_scores = [scores[i][j][1] for i, j in matches]
    total_prec = sum(prec_scores)
    total_rec = sum(rec_scores)
    scoring_metrics = {
        "precision": [total_prec, len(scores[0])],
        "recall": [total_rec, len(scores)],
        "precision_of_matches": prec_scores,
        "recall_of_matches": rec_scores,
    }
    return scoring_metrics


def aggregate_exact_matches(match_matrix):
    # For this aggregation task, no predicted tuple can exact-match two gold
    # ones, so it's easy, look at lines and columns looking for OR-total booleans.
    recall = [
        sum([any(gold_matches) for gold_matches in match_matrix], 0),
        len(match_matrix),
    ]
    # ^ this is [3,5] for "3 out of 5", to be lumped together later.
    if len(match_matrix[0]) == 0:
        precision = [0, 0]  # N/A
    else:
        precision = [
            sum(
                [
                    any([g[i] for g in match_matrix])
                    for i in range(len(match_matrix[0]))
                ],
                0,
            ),
            len(match_matrix[0]),
        ]
    # f1 = 2 * precision * recall / (precision + recall)
    metrics = {"precision": precision, "recall": recall}
    return metrics


def tuple_exact_match(t, gt):
    """Without resolving coref and WITH the need to hallucinate humanly infered
    words, does the tuple match the reference ? Returns a boolean."""
    for part in ["arg1", "rel", "arg2"]:
        if not t[part] == " ".join(gt[part]["words"]):
            # This purposedly ignores that some of the gt words are 'inf'
            # print("Predicted '{}' is different from reference '{}'".format(t[part], ' '.join(gt[part]['words'])))
            return False
    if gt["arg3+"]:
        # check if "arg3+" is available in t
        if not t.get("arg3+", False):
            return False
        # added check if number of arg3+ arguments is equal
        elif len(t["arg3+"]) != len(gt["arg3+"]):
            return False
        for i, p in enumerate(gt["arg3+"]):
            # check if every arg3+ element from gt is available in t
            if t["arg3+"][i] != " ".join(p["words"]):
                return False
    return True


"""
Wire57 tuples are built like so:
t = {"attrib/spec?" : attrib,
     "arg1" : {'text' : arg1, 'words': arg1_w, "words_indexes" : arg1_ind,
               'dc_text' : arg1dc, 'decorefed_words' : arg1dc_w, 'decorefed_indexes' : arg1dc_ind},
     "rel" : {'text' : rel, 'words': rel_w, "words_indexes" : rel_ind},
     "arg2" : {'text' : arg2, 'words': arg2_w, "words_indexes" : arg2_ind,
               'dc_text' : arg2dc, 'decorefed_words' : arg2dc_w, 'decorefed_indexes' : arg2dc_ind},
     "arg3+" : [{'text' : a,
                 'words' : arg3dat['raw_w'][i], 'words_indexes' : arg3dat['raw_ind'][i],
                 'decorefed_words' : arg3dat['dc_w'][i],
                 'decorefed_indexes' : arg3dat['dc_ind'][i]}
                for i,a in enumerate(arg3s)]}
"""


def tuple_match(t, gt):
    """t is a predicted tuple, gt is the gold tuple. How well do they match ?
    Yields precision and recall scores, a pair of non-zero values, if it's a match, and False if it's not.
    """
    precision = [0, 0]  # 0 out of 0 predicted words match
    recall = [0, 0]  # 0 out of 0 reference words match
    # If, for each part, any word is the same as a reference word, then it's a match.
    for part in ["arg1", "rel", "arg2"]:
        logger.debug(f"{part=}")
        predicted_words = t[part].split()
        gold_words = gt[part]["words"]
        gold_indexes = gt[part]["words_indexes"]
        gold_num_realwords = sum([i != "inf" for i in gold_indexes], 0)
        logger.info(f"{gold_num_realwords=}")
        gold_is_fully_inferred = all([i == "inf" for i in gold_indexes])
        # added this info
        if gold_is_fully_inferred:
            logger.info("Gold tuple is fully inferred!")
        if not predicted_words:
            logger.info("No prediction available!")
            if gold_words and not gold_is_fully_inferred:
                return False
            else:
                continue
        matching_words = sum(1 for w in predicted_words if w in gold_words)
        if matching_words == 0 and not gold_is_fully_inferred:
            return False  # t <-> gt is not a match
        logger.info(f"{part}: {matching_words} matching words")
        precision[0] += matching_words
        precision[1] += len(predicted_words)
        # Currently this slightly penalises systems when the reference
        # reformulates the sentence words, because the reformulation doesn't
        # match the predicted word. It's a one-wrong-word penalty to precision,
        # to all systems that correctly extracted the reformulated word.
        recall[0] += matching_words
        recall[
            1
        ] += gold_num_realwords  # len(gold_words) would include inferred words, unfairly to systems

    if gt["arg3+"]:
        for i, a in enumerate(gt["arg3+"]):
            gold_words = a["words"]
            gold_num_realwords = sum([i != "inf" for i in a["words_indexes"]], 0)
            assert gold_num_realwords <= len(gold_words)
            recall[1] += gold_num_realwords
            # if gt has more arg3+ elements than t -> next if is passed
            # but number of reference words is updated for recall!
            # checks the number of entries in the arg3+ list for the predicted tuple
            if t.get("arg3+", False) and len(t["arg3+"]) > i:
                predicted_words = t["arg3+"][i].split()
                matching_words = sum(1 for w in predicted_words if w in gold_words)
                precision[0] += matching_words
                precision[1] += len(predicted_words)
                recall[0] += matching_words
            else:
                # 0 matching words and precision is unchanged
                pass
    prec = precision[0] / precision[1]
    rec = recall[0] / recall[1]
    return [prec, rec]


def split_tuples_by_extractor(gold, tuples):
    systems = sorted(list(set(t["extractor"] for st in tuples.values() for t in st)))
    predictions_by_OIE = {e: {} for e in systems}
    for s in gold:
        # skip sentences (s) without predictions (tuples[s])
        try:
            tt = tuples[s]
        except KeyError:
            continue
        for t in tuples[s]:
            if s in predictions_by_OIE[t["extractor"]]:
                predictions_by_OIE[t["extractor"]][s].append(t)
            else:
                predictions_by_OIE[t["extractor"]][s] = [t]
    return predictions_by_OIE


## additional functions from wirer_scorer.py
# def avg(l):
#     return sum(l) / len(l)

# def str_list(thing):
#     return "\n".join([str(s) for s in thing])

# def part_to_string(p):
#     return " ".join(p["words"])


# def gold_to_text(gt):
#     text = " ; ".join(
#         [
#             part_to_string(gt["arg1"]),
#             part_to_string(gt["rel"]),
#             part_to_string(gt["arg2"]),
#         ]
#     )
#     if gt["arg3+"]:
#         text += " ; " + " ; ".join(gt["arg3+"])
#     return text
