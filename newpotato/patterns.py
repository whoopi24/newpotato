import itertools
import logging
from collections import Counter
from functools import lru_cache
from typing import List, Union

import graphbrain.constants as const
from graphbrain import hedge
from graphbrain.hyperedge import Hyperedge
from graphbrain.hypergraph import Hypergraph
from graphbrain.learner.classifier import apply_curly_brackets
from graphbrain.learner.pattern_ops import all_variables, contains_variable
from graphbrain.semsim import semsim
from graphbrain.utils.lemmas import lemma
from graphbrain.utils.semsim import (
    SEMSIM_CTX_THRESHOLD_PREFIX,
    SEMSIM_CTX_TOK_POS_PREFIX,
    extract_pattern_words,
    extract_similarity_threshold,
    get_edge_word_part,
    get_semsim_ctx_thresholds,
    get_semsim_ctx_tok_poses,
    replace_edge_word_part,
)

logger = logging.getLogger(__name__)

FUNS = {"var", "atoms", "lemma", "any", "semsim", "semsim-fix", "semsim-ctx"}

argrole_order = {
    "m": -1,
    "s": 0,
    "p": 1,
    "a": 2,
    "c": 3,
    "o": 4,
    "i": 5,
    "t": 6,
    "j": 7,
    "x": 8,
    "r": 9,
    "?": 10,
}


def is_wildcard(atom):
    """Check if this atom defines a wildcard, i.e. if its root is a pattern matcher.
    (\*, ., ..., if it is surrounded by parenthesis or variable label starting with an uppercase letter)
    """
    if atom.atom:
        return atom.parens or atom[0][0] in {"*", "."} or atom[0][0].isupper()
    else:
        return False


def is_fun_pattern(edge):
    if edge.atom:
        return False
    return str(edge[0]) in FUNS


def is_pattern(edge):
    """Check if this edge defines a pattern, i.e. if it includes at least
    one pattern matcher.

    Pattern matcher are:
    - '\*', '.', '(\*)', '...'
    - variables (atom label starting with an uppercase letter)
    - argument role matcher (unordered argument roles surrounded by curly brackets)
    - functional patterns (var, atoms, lemma, ...)
    """
    if edge.atom:
        return is_wildcard(edge) or "{" in edge.argroles()
    elif is_fun_pattern(edge):
        return True
    else:
        return any(is_pattern(item) for item in edge)


def is_unordered_pattern(edge):
    """Check if this edge defines an unordered pattern, i.e. if it includes at least
    one instance of unordered argument roles surrounded by curly brackets.
    """
    if edge.atom:
        return "{" in edge.argroles()
    else:
        return any(is_unordered_pattern(item) for item in edge)


def is_full_pattern(edge):
    """Check if every atom is a pattern matcher.

    Pattern matcher are:
    '\*', '.', '(\*)', '...', variables (atom label starting with an
    uppercase letter) and functional patterns.
    """
    if edge.atom:
        return is_pattern(edge)
    else:
        return all(is_pattern(item) for item in edge)


def apply_vars(edge, variables):
    if edge.atom:
        if is_pattern(edge):
            varname = _varname(edge)
            if len(varname) > 0 and varname in variables:
                return variables[varname]
        return edge
    else:
        return hedge([apply_vars(subedge, variables) for subedge in edge])


def _matches_atomic_pattern(edge, atomic_pattern):
    ap_parts = atomic_pattern.parts()

    if len(ap_parts) == 0 or len(ap_parts[0]) == 0:
        return False

    # structural match
    struct_code = ap_parts[0][0]
    if struct_code == ".":
        if edge.not_atom:
            return False
    elif atomic_pattern.parens:
        if edge.atom:
            return False
    elif struct_code != "*" and not struct_code.isupper():
        if edge.not_atom:
            return False
        if edge.root() != atomic_pattern.root():
            return False

    # role match
    if len(ap_parts) > 1:
        pos = 1

        # type match
        ap_role = atomic_pattern.role()
        ap_type = ap_role[0]
        e_type = edge.type()
        n = len(ap_type)
        if len(e_type) < n or e_type[:n] != ap_type:
            return False

        e_atom = edge.inner_atom()

        if len(ap_role) > 1:
            e_role = e_atom.role()
            # check if edge role has enough parts to satisfy the wildcard
            # specification
            if len(e_role) < len(ap_role):
                return False

            # argroles match
            if ap_type[0] in {"B", "P"}:
                ap_argroles_parts = ap_role[1].split("-")
                if len(ap_argroles_parts) == 1:
                    ap_argroles_parts.append("")
                ap_negroles = ap_argroles_parts[1]

                # fixed order?
                ap_argroles_posopt = ap_argroles_parts[0]
                e_argroles = e_role[1]
                if len(ap_argroles_posopt) > 0 and ap_argroles_posopt[0] == "{":
                    ap_argroles_posopt = ap_argroles_posopt[1:-1]
                else:
                    ap_argroles_posopt = ap_argroles_posopt.replace(",", "")
                    if len(e_argroles) > len(ap_argroles_posopt):
                        return False
                    else:
                        return ap_argroles_posopt.startswith(e_argroles)

                ap_argroles_parts = ap_argroles_posopt.split(",")
                ap_posroles = ap_argroles_parts[0]
                ap_argroles = set(ap_posroles) | set(ap_negroles)
                for argrole in ap_argroles:
                    min_count = ap_posroles.count(argrole)
                    # if there are argrole exclusions
                    fixed = ap_negroles.count(argrole) > 0
                    count = e_argroles.count(argrole)
                    if count < min_count:
                        return False
                    # deal with exclusions
                    if fixed and count > min_count:
                        return False
                pos = 2

            # match rest of role
            while pos < len(ap_role):
                if e_role[pos] != ap_role[pos]:
                    return False
                pos += 1

    # match rest of atom
    if len(ap_parts) > 2:
        e_parts = e_atom.parts()
        # check if edge role has enough parts to satisfy the wildcard
        # specification
        if len(e_parts) < len(ap_parts):
            return False

        while pos < len(ap_parts):
            if e_parts[pos] != ap_parts[pos]:
                return False
            pos += 1

    return True


def _varname(atom):
    if not atom.atom:
        return ""
    label = atom.parts()[0]
    if len(label) == 0:
        return label
    elif label[0] in {"*", "."}:
        return label[1:]
    elif label[:3] == "...":
        return label[3:]
    elif label[0].isupper():
        return label
    else:
        return ""


# remove pattern functions from pattern, so that .argroles() works normally
def _defun_pattern_argroles(edge):
    if edge.atom:
        return edge

    if edge[0].argroles() != "":
        return edge

    if is_fun_pattern(edge):
        fun = edge[0].root()
        if fun == "atoms":
            for atom in edge.atoms():
                argroles = atom.argroles()
                if argroles != "":
                    return atom
            # if no atom with argroles is found, just return the first one
            return edge[1]
        else:
            return hedge([edge[0], _defun_pattern_argroles(edge[1])] + list(edge[2:]))
    else:
        return hedge([_defun_pattern_argroles(subedge) for subedge in edge])


def _atoms_and_tok_pos(edge, tok_pos):
    if edge.atom:
        return [edge], [tok_pos]
    atoms = []
    atoms_tok_pos = []
    for edge_item, tok_pos_item in zip(edge, tok_pos):
        _atoms, _atoms_tok_pos = _atoms_and_tok_pos(edge_item, tok_pos_item)
        for _atom, _atom_tok_pos in zip(_atoms, _atoms_tok_pos):
            if _atom not in atoms:
                atoms.append(_atom)
                atoms_tok_pos.append(_atom_tok_pos)
    return atoms, atoms_tok_pos


def _normalize_fun_patterns(pattern):
    if pattern.atom:
        return pattern

    pattern = hedge([_normalize_fun_patterns(subpattern) for subpattern in pattern])

    if is_fun_pattern(pattern):
        if str(pattern[0]) == "lemma":
            if is_fun_pattern(pattern[1]) and str(pattern[1][0]) == "any":
                new_pattern = ["any"]
                for alternative in pattern[1][1:]:
                    new_pattern.append(["lemma", alternative])
                return hedge(new_pattern)

    return pattern


def _edge_tok_pos(edge: Hyperedge, hg: Hypergraph = None) -> Union[Hyperedge, None]:
    if hg is None:
        logger.debug(f"No hypergraph given to retrieve 'tok_pos' attribute for edge")
        return None

    tok_pos_str: str = hg.get_str_attribute(edge, "tok_pos")
    # edge is not a root edge
    if not tok_pos_str:
        logger.debug(f"Edge has no 'tok_pos' string attribute: {edge}")
        return None

    try:
        tok_pos_hedge: Hyperedge = hedge(tok_pos_str)
    except ValueError:
        logger.warning(f"Edge has invalid 'tok_pos' attribute: {edge}")
        return None

    return tok_pos_hedge


def match_pattern(edge, pattern, curvars=None, hg=None, ref_edges=None):
    """Matches an edge to a pattern. This means that, if the edge fits the
    pattern, then a dictionary will be returned with the values for each
    pattern variable. If the pattern specifies no variables but the edge
    matches it, then an empty dictionary is returned. If the edge does
    not match the pattern, None is returned.

    Patterns are themselves edges. They can match families of edges
    by employing special atoms:

    -> '\*' represents a general wildcard (matches any entity)

    -> '.' represents an atomic wildcard (matches any atom)

    -> '(\*)' represents an edge wildcard (matches any edge)

    -> '...' at the end indicates an open-ended pattern.

    The wildcards ('\*', '.' and '(\*)') can be used to specify variables,
    for example '\*x', '(CLAIM)' or '.ACTOR'. In case of a match, these
    variables are assigned the hyperedge they correspond to. For example,

    (1) the edge: (is/Pd (my/Mp name/Cn) mary/Cp)
    applied to the pattern: (is/Pd (my/Mp name/Cn) \*NAME)
    produces the result: {'NAME', mary/Cp}

    (2) the edge: (is/Pd (my/Mp name/Cn) mary/Cp)
    applied to the pattern: (is/Pd (my/Mp name/Cn) (NAME))
    produces the result: {}

    (3) the edge: (is/Pd (my/Mp name/Cn) mary/Cp)
    applied to the pattern: (is/Pd . \*NAME)
    produces the result: None
    """
    edge_hedged: Hyperedge = hedge(edge)
    pattern_hedged: Hyperedge = hedge(pattern)
    pattern_hedged_normalized: Hyperedge = _normalize_fun_patterns(pattern_hedged)
    matcher = Matcher(
        edge_hedged,
        pattern_hedged_normalized,
        curvars=curvars,
        tok_pos=_edge_tok_pos(edge, hg),
        hg=hg,
    )

    # check for semsim_ctx matches if necessary
    if matcher.semsim_ctx and matcher.results:
        return _match_semsim_ctx(
            matcher, edge_hedged, pattern_hedged_normalized, ref_edges, hg
        )

    return matcher.results


def edge_matches_pattern(edge, pattern, hg=None, ref_edges=None):
    """Check if an edge matches a pattern.

    Patterns are themselves edges. They can match families of edges
    by employing special atoms:

    -> '\*' represents a general wildcard (matches any entity)

    -> '.' represents an atomic wildcard (matches any atom)

    -> '(\*)' represents an edge wildcard (matches any edge)

    -> '...' at the end indicates an open-ended pattern.

    The pattern can be any valid hyperedge, including the above special atoms.
    Examples: (is/Pd graphbrain/C .)
    (says/Pd * ...)
    """
    result = match_pattern(edge, pattern, hg=hg, ref_edges=ref_edges)
    return len(result) > 0


def _match_semsim_ctx(
    matcher: "Matcher",
    edge: Hyperedge,
    pattern: Hyperedge,
    ref_edges: list[Hyperedge],
    hg: Hypergraph,
):
    if not _edge_tok_pos(edge, hg):
        logger.error(f"Candidate edge has no 'tok_pos' attribute: {edge}")
        return []

    cand_edge_tok_poses: dict[int, Hyperedge] = get_semsim_ctx_tok_poses(
        matcher.results_with_special_vars
    )
    thresholds: dict[int, Union[float, None]] = get_semsim_ctx_thresholds(
        matcher.results_with_special_vars
    )

    ref_edges_tok_poses: list[dict[int, Hyperedge]] = _get_ref_edges_tok_poses(
        pattern, ref_edges, [_edge_tok_pos(ref_edge, hg) for ref_edge in ref_edges], hg
    )

    try:
        assert cand_edge_tok_poses.keys() == thresholds.keys() and all(
            cand_edge_tok_poses.keys() == ref_edge_tok_poses.keys()
            for ref_edge_tok_poses in ref_edges_tok_poses
        )
    except AssertionError:
        raise ValueError(
            f"Number of semsim-ctx for candidate edge and reference edges do not match"
        )

    for semsim_ctx_idx in cand_edge_tok_poses.keys():
        if not semsim(
            semsim_type="CTX",
            threshold=thresholds[semsim_ctx_idx],
            cand_edge=edge,
            ref_edges=ref_edges,
            cand_tok_pos=cand_edge_tok_poses[semsim_ctx_idx],
            ref_tok_poses=[
                ref_edge_tok_poses[semsim_ctx_idx]
                for ref_edge_tok_poses in ref_edges_tok_poses
            ],
            hg=hg,
        ):
            return []

    return matcher.results


# these methods need to be in this module to avoid circular imports
# store hypergraphs to avoid passing them as arguments and enable caching
# TODO: better caching, this is insensitive to changes in the hypergraph
_HG_STORE: dict[int, Hypergraph] = {}


#
def _get_ref_edges_tok_poses(
    pattern: Hyperedge,
    ref_edges: list[Hyperedge],
    root_tok_poses: list[Hyperedge],
    hg: Hypergraph,
) -> list[dict[int, Hyperedge]]:
    hg_id: int = id(hg)
    _HG_STORE[hg_id] = hg

    return _get_ref_edges_tok_poses_cached(
        pattern, tuple(ref_edges), tuple(root_tok_poses), hg_id
    )


@lru_cache(maxsize=None)
def _get_ref_edges_tok_poses_cached(
    pattern: Hyperedge,
    ref_edges: tuple[Hyperedge],
    root_tok_poses: tuple[Hyperedge],
    hg_id: int,
) -> list[dict[int, Hyperedge]]:
    try:
        hg: Hypergraph = _HG_STORE[hg_id]
    except KeyError:
        raise ValueError(f"Hypergraph with id '{hg_id}' not found")

    return [
        get_semsim_ctx_tok_poses(ref_matcher.results_with_special_vars)
        for ref_matcher in [
            Matcher(ref_edge, pattern, tok_pos=tok_pos, hg=hg)
            for ref_edge, tok_pos in zip(ref_edges, root_tok_poses)
        ]
    ]


def _generate_special_var_name(var_code, vars_):
    prefix = f"__{var_code}"
    var_count = len([var_name for var_name in vars_ if var_name.startswith(prefix)])
    return f"__{var_code}_{var_count}"


def _regular_var_count(vars_):
    return len([var_name for var_name in vars_ if not var_name.startswith("__")])


def _remove_special_vars(vars_):
    return {key: value for key, value in vars_.items() if not key.startswith("__")}


def _assign_edge_to_var(curvars, var_name, edge):
    new_edge = edge
    if var_name in curvars:
        cur_edge = curvars[var_name]
        if cur_edge.not_atom and str(cur_edge[0]) == const.list_or_matches_builder:
            new_edge = cur_edge + (edge,)
        else:
            new_edge = hedge((hedge(const.list_or_matches_builder), cur_edge, edge))
    return {var_name: new_edge}


class Matcher:
    def __init__(self, edge, pattern, curvars=None, tok_pos=None, hg=None):
        self.hg = hg
        self.semsim_ctx = False
        self.results_with_special_vars = self._match(
            edge, pattern, curvars=curvars, tok_pos=tok_pos
        )
        self.results = [
            _remove_special_vars(result) for result in self.results_with_special_vars
        ]

    def _match(self, edge, pattern, curvars=None, tok_pos=None):
        if curvars is None:
            curvars = {}

        # functional patterns
        if is_fun_pattern(pattern):
            return self._match_fun_pat(edge, pattern, curvars, tok_pos=tok_pos)

        # function pattern on edge can never match non-functional pattern
        if is_fun_pattern(edge):
            return []

        # atomic patterns
        if pattern.atom:
            if _matches_atomic_pattern(edge, pattern):
                variables = {}
                if is_pattern(pattern):
                    varname = _varname(pattern)
                    if len(varname) > 0:
                        # if varname in curvars and curvars[varname] != edge:
                        #     return []
                        variables[varname] = _assign_edge_to_var(
                            {**curvars, **variables}, varname, edge
                        )[varname]
                return [{**curvars, **variables}]
            else:
                return []

        min_len = len(pattern)
        max_len = min_len
        # open-ended?
        if pattern[-1].to_str() == "...":
            pattern = hedge(pattern[:-1])
            min_len -= 1
            max_len = float("inf")

        result = [{}]
        argroles_posopt = _defun_pattern_argroles(pattern)[0].argroles().split("-")[0]
        if len(argroles_posopt) > 0 and argroles_posopt[0] == "{":
            match_by_order = False
            argroles_posopt = argroles_posopt[1:-1]
        else:
            match_by_order = True
        argroles = argroles_posopt.split(",")[0]
        argroles_opt = argroles_posopt.replace(",", "")

        if len(argroles) > 0:
            min_len = 1 + len(argroles)
            max_len = float("inf")
        else:
            match_by_order = True

        if len(edge) < min_len or len(edge) > max_len:
            return []

        # match by order
        if match_by_order:
            for i, pitem in enumerate(pattern):
                eitem = edge[i]
                _result = []

                for variables in result:
                    if pitem.atom:
                        varname = _varname(pitem)
                        if _matches_atomic_pattern(eitem, pitem):  # elif
                            if len(varname) > 0 and varname[0].isupper():
                                variables[varname] = _assign_edge_to_var(
                                    {**curvars, **variables}, varname, eitem
                                )[varname]
                        else:
                            continue
                        _result.append(variables)
                    else:
                        tok_pos_item = None
                        if tok_pos is not None:
                            try:
                                assert len(tok_pos) > i
                            except AssertionError:
                                raise RuntimeError(
                                    f"Index '{i}' in tok_pos '{tok_pos}' is out of range"
                                )
                            tok_pos_item = tok_pos[i]
                        _result += self._match(
                            eitem, pitem, {**curvars, **variables}, tok_pos=tok_pos_item
                        )
                result = _result
        # match by argroles
        else:
            result = []
            # match connector first
            # TODO: avoid matching connector twice!
            ctok_pos = tok_pos[0] if tok_pos else None
            if self._match(edge[0], pattern[0], curvars, tok_pos=ctok_pos):
                role_counts = Counter(argroles_opt).most_common()
                unknown_roles = (len(pattern) - 1) - len(argroles_opt)
                if unknown_roles > 0:
                    role_counts.append(("*", unknown_roles))
                # add connector pseudo-argrole
                role_counts = [("X", 1)] + role_counts
                result = self._match_by_argroles(
                    edge,
                    pattern,
                    role_counts,
                    len(argroles),
                    curvars=curvars,
                    tok_pos=tok_pos,
                )

        unique_vars = []
        for variables in result:
            v = {**curvars, **variables}
            if v not in unique_vars:
                unique_vars.append(v)
        return unique_vars

    def _match_by_argroles(
        self,
        edge,
        pattern,
        role_counts,
        min_vars,
        matched=(),
        curvars=None,
        tok_pos=None,
    ):
        if curvars is None:
            curvars = {}

        if len(role_counts) == 0:
            return [curvars]

        argrole, n = role_counts[0]

        # match connector
        if argrole == "X":
            eitems = [edge[0]]
            pitems = [pattern[0]]
        # match any argrole
        elif argrole == "*":
            eitems = [e for e in edge if e not in matched]
            pitems = pattern[-n:]
        # match specific argrole
        else:
            eitems = edge.edges_with_argrole(argrole)
            pitems = _defun_pattern_argroles(pattern).edges_with_argrole(argrole)

        if len(eitems) < n:
            if _regular_var_count(curvars) >= min_vars:
                return [curvars]
            else:
                return []

        result = []

        if tok_pos:
            tok_pos_items = [
                tok_pos[i] for i, subedge in enumerate(edge) if subedge in eitems
            ]
            tok_pos_perms = tuple(itertools.permutations(tok_pos_items, r=n))

        for perm_n, perm in enumerate(tuple(itertools.permutations(eitems, r=n))):
            if tok_pos:
                tok_pos_perm = tok_pos_perms[perm_n]
            perm_result = [{}]
            for i, eitem in enumerate(perm):
                pitem = pitems[i]
                tok_pos_item = tok_pos_perm[i] if tok_pos else None
                item_result = []
                for variables in perm_result:
                    item_result += self._match(
                        eitem, pitem, {**curvars, **variables}, tok_pos=tok_pos_item
                    )
                perm_result = item_result
                if len(item_result) == 0:
                    break

            for variables in perm_result:
                result += self._match_by_argroles(
                    edge,
                    pattern,
                    role_counts[1:],
                    min_vars,
                    matched + perm,
                    {**curvars, **variables},
                    tok_pos=tok_pos,
                )

        return result

    def _match_atoms(
        self, atom_patterns, atoms, curvars, atoms_tok_pos=None, matched_atoms=None
    ) -> list:
        if matched_atoms is None:
            matched_atoms = []

        if len(atom_patterns) == 0:
            return [curvars]

        results = []
        atom_pattern = atom_patterns[0]

        for atom_pos, atom in enumerate(atoms):
            if atom not in matched_atoms:
                tok_pos = atoms_tok_pos[atom_pos] if atoms_tok_pos else None
                svars = self._match(atom, atom_pattern, curvars, tok_pos=tok_pos)
                for variables in svars:
                    results += self._match_atoms(
                        atom_patterns[1:],
                        atoms,
                        {**curvars, **variables},
                        atoms_tok_pos=atoms_tok_pos,
                        matched_atoms=matched_atoms + [atom],
                    )

        return results

    # TODO: deal with argroles
    def _match_lemma(self, lemma_pattern, edge, curvars):
        if self.hg is None:
            raise RuntimeError("Lemma pattern function requires hypergraph.")

        if edge.not_atom:
            return []

        _lemma = lemma(self.hg, edge, same_if_none=True)

        # add argroles to _lemma if needed
        ar = edge.argroles()
        if ar != "":
            parts = _lemma.parts()
            parts[1] = "{}.{}".format(parts[1], ar)
            _lemma = hedge("/".join(parts))

        if _matches_atomic_pattern(_lemma, lemma_pattern):
            return [curvars]

        return []

    def _match_semsim(
        self,
        pattern: Hyperedge,
        edge: Hyperedge,
        curvars: dict,
        # ) -> list[dict]:
    ) -> List[dict]:
        edge_word_part: str = get_edge_word_part(edge)
        if not edge_word_part:
            return []

        # can be one word (e.g. "say") or a list of words (e.g. ["say, tell, speak"])
        pattern_words_part: str = pattern[0].parts()[0]
        reference_words: list[str] = extract_pattern_words(pattern_words_part)

        threshold: float | None = extract_similarity_threshold(pattern)
        if not semsim(
            semsim_type="FIX",
            threshold=threshold,
            cand_word=edge_word_part,
            ref_words=reference_words,
        ):
            return []

        edge_with_pattern_word_part = replace_edge_word_part(edge, pattern_words_part)
        if _matches_atomic_pattern(edge_with_pattern_word_part, pattern[0]):
            return [curvars]

        return []

    def _match_fun_pat(self, edge, fun_pattern, curvars, tok_pos=None) -> list:
        fun = fun_pattern[0].root()
        if fun == "var":
            if len(fun_pattern) != 3:
                raise RuntimeError("var pattern function must have two arguments")
            pattern = fun_pattern[1]
            var_name = fun_pattern[2].root()
            if (
                edge.not_atom
                and str(edge[0]) == "var"
                and len(edge) == 3
                and str(edge[2]) == var_name
            ):
                this_var = _assign_edge_to_var(curvars, var_name, edge[1])
                return self._match(
                    edge[1], pattern, curvars={**curvars, **this_var}, tok_pos=tok_pos
                )
            else:
                this_var = _assign_edge_to_var(curvars, var_name, edge)
                return self._match(
                    edge, pattern, curvars={**curvars, **this_var}, tok_pos=tok_pos
                )
        elif fun == "atoms":
            if tok_pos:
                atoms, atoms_tok_pos = _atoms_and_tok_pos(edge, tok_pos)
            else:
                atoms = edge.atoms()
                atoms_tok_pos = None
            atom_patterns = fun_pattern[1:]
            return self._match_atoms(
                atom_patterns, atoms, curvars, atoms_tok_pos=atoms_tok_pos
            )
        elif fun == "lemma":
            return self._match_lemma(fun_pattern[1], edge, curvars)
        elif fun == "semsim" or fun == "semsim-fix":
            return self._match_semsim(
                fun_pattern[1:],
                edge,
                curvars,
                hg=self.hg,
            )
        elif fun == "semsim-ctx":
            self.semsim_ctx = True
            threshold = extract_similarity_threshold(fun_pattern[1:])
            special_vars = {
                _generate_special_var_name(SEMSIM_CTX_TOK_POS_PREFIX, curvars): tok_pos,
                _generate_special_var_name(
                    SEMSIM_CTX_THRESHOLD_PREFIX, curvars
                ): threshold,
            }
            return self._match(
                edge,
                fun_pattern[1],
                curvars={**curvars, **special_vars},
                tok_pos=tok_pos,
            )
        elif fun == "any":
            for pattern in fun_pattern[1:]:
                matches = self._match(edge, pattern, curvars=curvars, tok_pos=tok_pos)
                if len(matches) > 0:
                    return matches
            return []
        else:
            raise RuntimeError(f"Unknown pattern function: {fun}")


def _edge2pattern(edge, root=False, subtype=False):
    # print("edge2pattern-e: ", edge)
    if root and edge.atom:
        root_str = edge.root()
    ## START: MARINA
    elif contains_variable(edge):
        # print("var is contained")
        # ToDo: what happens when two variables inside one hyperedge?
        if edge.contains("REL", deep=True):
            root_str = "REL"
        elif edge.contains("ARG0", deep=True):
            root_str = "ARG0"
        elif edge.contains("ARG1", deep=True):
            root_str = "ARG1"
        elif edge.contains("ARG2", deep=True):
            root_str = "ARG2"
        elif edge.contains("ARG3", deep=True):
            root_str = "ARG3"
        elif edge.contains("ARG4", deep=True):
            root_str = "ARG4"
        elif edge.contains("ARG5", deep=True):
            root_str = "ARG5"
        else:
            # print("problem")
            root_str = "*"
    ## END: MARINA
    else:
        root_str = "*"
    if subtype:
        et = edge.type()
    else:
        et = edge.mtype()
    pattern = "{}/{}".format(root_str, et)
    ar = edge.argroles()
    if ar == "":
        # print("edge2pattern-p: ", hedge(pattern))
        return hedge(pattern)
    else:
        # print("edge2pattern-p: ", hedge("{}.{}".format(pattern, ar)))
        return hedge("{}.{}".format(pattern, ar))


def _inner_edge_matches_pattern(edge, pattern, hg=None):
    if edge.atom:
        return False
    for subedge in edge:
        if edge_matches_pattern(subedge, pattern, hg=hg):
            return True
    for subedge in edge:
        if _inner_edge_matches_pattern(subedge, pattern, hg=hg):
            return True
    return False


class PatternCounter:
    def __init__(
        self,
        depth=2,
        count_subedges=True,
        expansions=None,
        match_roots=None,
        match_subtypes=None,
    ):
        self.patterns = Counter()
        self.depth = depth
        self.count_subedges = count_subedges
        if expansions is None:
            self.expansions = {"*"}
        else:
            self.expansions = expansions
        if match_roots is None:
            self.match_roots = set()
        else:
            self.match_roots = match_roots
        if match_subtypes is None:
            self.match_subtypes = set()
        else:
            self.match_subtypes = match_subtypes

    def _matches_expansions(self, edge):
        for expansion in self.expansions:
            if edge_matches_pattern(edge, expansion):
                return True
        return False

    def _force_subtypes(self, edge):
        force_subtypes = False
        for st_pattern in self.match_subtypes:
            if edge_matches_pattern(edge, st_pattern):
                force_subtypes = True
        return force_subtypes

    def _force_root_expansion(self, edge):
        force_root = False
        force_expansion = False
        for root_pattern in self.match_roots:
            if edge_matches_pattern(edge, root_pattern):
                force_root = True
                force_expansion = True
            elif _inner_edge_matches_pattern(edge, root_pattern):
                force_expansion = True
        return force_root, force_expansion

    def _list2patterns(
        self,
        ledge,
        depth=1,
        force_expansion=False,
        force_root=False,
        force_subtypes=False,
    ):
        if depth > self.depth:
            return []

        # print("ledge: ", ledge)

        first = ledge[0]

        # print("first: ", first)

        f_force_subtypes = force_subtypes | self._force_subtypes(first)

        f_force_root, f_force_expansion = self._force_root_expansion(first)
        f_force_root |= force_root
        f_force_expansion |= force_expansion
        root = force_root | f_force_root

        if f_force_expansion and not first.atom:
            hpats = []
        else:
            hpats = [_edge2pattern(first, root=root, subtype=f_force_subtypes)]

        if not first.atom and (self._matches_expansions(first) or f_force_expansion):
            hpats += self._list2patterns(
                list(first),
                depth + 1,
                force_expansion=f_force_expansion,
                force_root=f_force_root,
                force_subtypes=f_force_subtypes,
            )
        # print("hpats: ", hpats)
        if len(ledge) == 1:
            print("length == 1")
            patterns = [[hpat] for hpat in hpats]
        else:
            print("length != 1 > recursion")
            patterns = []
            for pattern in self._list2patterns(
                ledge[1:],
                depth=depth,
                force_expansion=force_expansion,
                force_root=force_root,
                force_subtypes=force_subtypes,
            ):
                for hpat in hpats:
                    patterns.append([hpat] + pattern)
        return patterns

    def _edge2patterns(self, edge):
        force_subtypes = self._force_subtypes(edge)
        force_root, _ = self._force_root_expansion(edge)
        return [
            hedge(pattern)
            for pattern in self._list2patterns(
                list(edge.normalized()),
                force_subtypes=force_subtypes,
                force_root=force_root,
            )
        ]

    def count(self, edge, check_vars=True):
        edge = hedge(edge)
        print("edge: ", edge)
        vars = all_variables(edge)
        # print("variables: ", vars.keys())
        if edge.not_atom:
            if self._matches_expansions(edge):
                # print("matches expansions")
                for pattern in self._edge2patterns(edge):
                    if check_vars:
                        skip = False
                        atoms = pattern.atoms()
                        roots = {atom.root() for atom in atoms}
                        for var in vars:
                            if str(var) in roots:
                                continue
                            else:
                                skip = True
                                break
                        if not skip:
                            print("count: ", pattern)
                            self.patterns[hedge(apply_curly_brackets(pattern))] += 1
                    else:
                        self.patterns[hedge(apply_curly_brackets(pattern))] += 1
            if self.count_subedges:
                print("go for subedges")
                for subedge in edge:
                    self.count(subedge)
