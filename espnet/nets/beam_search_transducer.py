"""Search algorithms for transducer models."""

import numpy as np
import numba

import logging

from dataclasses import asdict
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[List[List[torch.Tensor]], List[torch.Tensor]]
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None


def greedy_search(decoder, h, recog_args, timer=None):
    """Greedy search implementation for transformer-transducer.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
        recog_args (Namespace): argument Namespace containing options

    Returns:
        hyp (list of dicts): 1-best decoding results

    """
    init_tensor = h.unsqueeze(0)
    dec_state = decoder.init_state(init_tensor)

    hyp = Hypothesis(score=0.0, yseq=[decoder.blank], dec_state=dec_state)

    cache = {}

    if timer:
        timer.tic("dec")
    y, state, _ = decoder.score(hyp, cache, init_tensor) # list of hyp , list of cache -> list of y, list of state
    # y: (bsz, dim), state: (bsz, number of dec layers, dim)
    if timer:
        timer.tic("utt total")
    for i, hi in enumerate(h):# first loop, frame sync, if h is batch : h.transpose(0, 1) (bsz, max_len, hdim-> (max_len, bsz, hdim)
        if timer:
            timer.tic("utt frame")

        ytu = torch.log_softmax(decoder.joint(hi, y[0]), dim=-1) # hi : (bsz, hdim), y: (bsz, dim), ytu: (bsz, odim)
        if timer:
            timer.toc("dec")
        logp, pred = torch.max(ytu, dim=-1)

        # insert new for loop handle batch hyp update
        #for i, (pred, logp, hyp, state) in enumrate(zip(preds, logps, hyp_batch, state_batch))
        if pred != decoder.blank:
            hyp.yseq.append(int(pred))
            hyp.score += float(logp)

            hyp.dec_state = state

            if timer:
                timer.tic("dec")
            y, state, _ = decoder.score(hyp, cache, init_tensor)
        if timer:
            timer.toc("utt frame")
    if timer:
        timer.toc("utt total")

    return [asdict(hyp)]

def greedy_search_batch(decoder, h, recog_args, timer=None): # batch h (bsz, max_len , hdim)
    """Greedy search implementation for transformer-transducer.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
        recog_args (Namespace): argument Namespace containing options

    Returns:
        hyp (list of dicts): 1-best decoding results

    """
    # init_tensor not used?
    # init_tensor = h.unsqueeze(0)
    # dec_state = decoder.init_state(init_tensor)
    init_tensor = h
    dec_state = decoder.init_state(init_tensor)

    # hyp list of batch size
    bsz= h.size(0)
    max_len  =h.size(1)
    hyp_batch = [Hypothesis(score=0.0, yseq=[decoder.blank], dec_state=dec_state) for _ in range(bsz)]  #list of hyp , equal batch size

    cache_batch = [{} for _ in range(bsz)]
    # seems cache is useless in greedy search mode, decoder inf execute in every token expanding loop, so no cache will be avaliable

    if timer:
        timer.tic("dec")
    #init_tensor not used in func
    y_batch, state_batch, _ = decoder.score_batch(hyp_batch, cache_batch, init_tensor) # list of hyp , list of cache -> list of y, list of state
    # y: (bsz, dim), state: (bsz, layers, max_len, dim)
    if timer:
        timer.tic("utt total")
    # for i, hi in enumerate(h):# first loop, frame sync, if h is batch : h.transpose(0, 1) (bsz, max_len, hdim-> (max_len, bsz, hdim)
    # processed frame numbers
    idx_batch= torch.zeros(bsz, dtype=torch.int)
    while  not (idx_batch ==max_len).all():
        for ib in range(bsz): # this loop should be optimized
            logp, pred=  .0, decoder.blank
            if idx_batch[ib] < max_len:
                ytu = torch.log_softmax(decoder.joint(h[ib, idx_batch[ib]].unsqueeze(0), y_batch[ib].unsqueeze(0)), dim=-1)
                logp, pred= torch.max(ytu, dim=-1)
                idx_batch[ib] += 1

                while idx_batch[ib] < max_len and pred == decoder.blank:

                    ytu = torch.log_softmax(decoder.joint(h[ib, idx_batch[ib]].unsqueeze(0), y_batch[ib].unsqueeze(0)), dim=-1)
                    logp, pred = torch.max(ytu, dim=-1)
                    idx_batch[ib] += 1

                if idx_batch[ib] < max_len: # escape while loop because of pred != blank, idx plus 1
                    idx_batch[ib] += 1

            # if do find the non blank output before reach end of seq
            if idx_batch[ib] < max_len or pred != decoder.blank:
                hyp_batch[ib].yseq.append(int(pred))
                hyp_batch[ib].score += float(logp)
                hyp_batch[ib].dec_state = state_batch[ib]

        y_batch, state_batch, _ = decoder.score_batch(hyp_batch, cache_batch, init_tensor)

    return [[asdict(hyp)] for hyp in hyp_batch]

def default_beam_search(decoder, h, recog_args, rnnlm=None, timer=None):
    """Beam search implementation.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)
    beam_k = min(beam, (decoder.odim - 1))

    nbest = recog_args.nbest
    normscore = recog_args.score_norm_transducer

    init_tensor = h.unsqueeze(0)
    blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

    dec_state = decoder.init_state(init_tensor)

    kept_hyps = [Hypothesis(score=0.0, yseq=[decoder.blank], dec_state=dec_state)]

    cache = {}
    if timer:
        timer.tic("utt total")

    for hi in h:
        if timer:
            timer.tic("utt frame")
        hyps = kept_hyps
        kept_hyps = []

        while True:
            max_hyp = max(hyps, key=lambda x: x.score)
            hyps.remove(max_hyp)

            if timer:
                timer.tic("dec")
            y, state, lm_tokens = decoder.score(max_hyp, cache, init_tensor)

            ytu = F.log_softmax(decoder.joint(hi, y[0]), dim=-1)
            if timer:
                timer.toc("dec")

            top_k = ytu[1:].topk(beam_k, dim=-1) # exclude the choice of blank, reason of plus 1 shows below

            ytu = (
                torch.cat((top_k[0], ytu[0:1])).cpu(), # force add blank as an optional choice
                torch.cat((top_k[1] + 1, blank_tensor)).cpu(),
            )

            if rnnlm:
                rnnlm_state, rnnlm_scores = rnnlm.predict(max_hyp.lm_state, lm_tokens)

            for logp, k in zip(*ytu):
                new_hyp = Hypothesis(
                    score=(max_hyp.score + float(logp)),
                    yseq=max_hyp.yseq[:],
                    dec_state=max_hyp.dec_state,
                    lm_state=max_hyp.lm_state,
                )

                if k == decoder.blank:
                    kept_hyps.append(new_hyp)
                else:
                    new_hyp.dec_state = state

                    new_hyp.yseq.append(int(k))

                    if rnnlm:
                        new_hyp.lm_state = rnnlm_state
                        new_hyp.score += recog_args.lm_weight * rnnlm_scores[0][k]

                    hyps.append(new_hyp)

            hyps_max = float(max(hyps, key=lambda x: x.score).score)
            kept_most_prob = sorted(
                [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                key=lambda x: x.score,
            )
            if len(kept_most_prob) >= beam:
                kept_hyps = kept_most_prob
                break
        if timer:
            timer.toc("utt frame")

    if normscore:
        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x.score / len(x.yseq), reverse=True
        )[:nbest]
    else:
        nbest_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)[:nbest]
    if timer: timer.toc("utt total")
    return [asdict(n) for n in nbest_hyps]
    # return nbest_hyps

# @numba.jit(nopython=True, parallel=True)
def time_sync_decoding(decoder, h, recog_args, rnnlm=None, timer=None):
    """Time synchronous beam search implementation.

    Based on https://ieeexplore.ieee.org/document/9053040

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)

    max_sym_exp = recog_args.max_sym_exp
    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    PreHypothesis = [Hypothesis(yseq=[decoder.blank], score=0.0, dec_state=decoder.select_state(beam_state, 0)) for _ in range(beam ** 2 * len(h) * max_sym_exp)]
    hyp_idx =  0

    B = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

            B[0].lm_state = init_lm_state(lm_model)

        lm_layers = len(lm_model.rnn)

    cache = {}

    if timer:
        timer.tic("utt total")

    for hi in h:
        if timer:
            timer.tic("utt frame")
        A = []
        C = B

        h_enc = hi.unsqueeze(0)

        for v in range(max_sym_exp):
            D = []

            if timer:
                timer.tic("dec")
            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                C, beam_state, cache, init_tensor, timer
            )

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1).cpu()
            beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)
            if timer:
                timer.toc("dec")

            seq_A = [h.yseq for h in A]

            if timer:
                timer.tic("blank hyp add")
            for i, hyp in enumerate(C):
                if hyp.yseq not in seq_A:
                    A.append(
                        Hypothesis(
                            score=(hyp.score + float(beam_logp[i, 0])),
                            yseq=hyp.yseq[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                        )
                    )
                else:
                    dict_pos = seq_A.index(hyp.yseq)

                    A[dict_pos].score = np.logaddexp(
                        A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                    )
            if timer:
                timer.toc("blank hyp add")

            if v < max_sym_exp -1:
                if rnnlm:
                    beam_lm_states = create_lm_batch_state(
                        [c.lm_state for c in C], lm_type, lm_layers
                    )

                    beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(C)
                    )

                if timer:
                    timer.tic("non blank hyp add")
                for i, hyp in enumerate(C):
                    if beam_logp[i, 0] > -1e-1:
                        continue # skip blank dominate condition
                    # logging.info("C loops : %d " % len(C) )
                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        if timer: timer.tic("add hyps")
                        new_hyp = PreHypothesis[hyp_idx]
                        new_hyp.score= hyp.score + float(logp)
                        new_hyp.yseq = hyp.yseq + [int(k)]
                        new_hyp.dec_state = decoder.select_state(beam_state, i)
                        new_hyp.lm_state = hyp.lm_state
                        hyp_idx += 1
                        # new_hyp = Hypothesis(
                        #     score=(hyp.score + float(logp)),
                        #     yseq=(hyp.yseq + [int(k)]),
                        #     dec_state=decoder.select_state(beam_state, i),
                        #     lm_state=hyp.lm_state,
                        # )


                        if rnnlm:
                            new_hyp.score += recog_args.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, lm_type, lm_layers
                            )

                        D.append(new_hyp)
                        if timer: timer.toc("add hyps")
                if timer:
                    timer.toc("non blank hyp add")

            if not D: break
            if timer: timer.tic("D sort")
            C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]
            if timer: timer.toc("D sort")

        if timer: timer.tic("A sort")
        B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
        if timer: timer.toc("A sort")
        if timer:
            timer.toc("utt frame")

    nbest_hyps = sorted(B, key=lambda x: x.score, reverse=True)[:nbest]
    if timer:
        timer.toc("utt total")

    return [asdict(n) for n in nbest_hyps]


def align_length_sync_decoding(decoder, h, recog_args, rnnlm=None, timer=None):
    """Alignment-length synchronous beam search implementation.

    Based on https://ieeexplore.ieee.org/document/9053040

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)

    h_length = int(h.size(0))
    u_max = min(recog_args.u_max, (h_length - 1))

    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    B = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]
    final = []

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

            B[0].lm_state = init_lm_state(lm_model)

        lm_layers = len(lm_model.rnn)

    cache = {}
    if timer:
        timer.tic("utt total")

    for i in range(h_length + u_max):
        if timer:
            timer.tic("utt frame")
        A = []

        B_ = []
        h_states = []
        for hyp in B:
            u = len(hyp.yseq) - 1
            t = i - u + 1

            if t > (h_length - 1):
                continue

            B_.append(hyp)
            h_states.append((t, h[t]))

        if B_:
            if timer:
                timer.tic("dec")
            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                B_, beam_state, cache, init_tensor
            )

            h_enc = torch.stack([h[1] for h in h_states])

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)
            if timer:
                timer.toc("dec")
            beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

            if rnnlm:
                beam_lm_states = create_lm_batch_state(
                    [b.lm_state for b in B_], lm_type, lm_layers
                )

                beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                    beam_lm_states, beam_lm_tokens, len(B_)
                )

            for i, hyp in enumerate(B_):
                new_hyp = Hypothesis(
                    score=(hyp.score + float(beam_logp[i, 0])),
                    yseq=hyp.yseq[:],
                    dec_state=hyp.dec_state,
                    lm_state=hyp.lm_state,
                )

                A.append(new_hyp)

                if h_states[i][0] == (h_length - 1):
                    final.append(new_hyp)

                for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(logp)),
                        yseq=(hyp.yseq[:] + [int(k)]),
                        dec_state=decoder.select_state(beam_state, i),
                        lm_state=hyp.lm_state,
                    )

                    if rnnlm:
                        new_hyp.score += recog_args.lm_weight * beam_lm_scores[i, k]

                        new_hyp.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )

                    A.append(new_hyp)

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
            B = recombine_hyps(B)
        if timer:
            timer.toc("utt frame")

    if final:
        nbest_hyps = sorted(final, key=lambda x: x.score, reverse=True)[:nbest]
    else:
        nbest_hyps = B[:nbest]
    if timer:
        timer.toc("utt total")

    return [asdict(n) for n in nbest_hyps]


def nsc_beam_search(decoder, h, recog_args, rnnlm=None, timer=None):
    """N-step constrained beam search implementation.

    Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
    Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
    until further modifications.

    Note: the algorithm is not in his "complete" form but works almost as
          intended.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)
    beam_k = min(beam, (decoder.odim - 1))

    nstep = recog_args.nstep
    prefix_alpha = recog_args.prefix_alpha

    nbest = recog_args.nbest

    cache = {}

    init_tensor = h.unsqueeze(0)
    blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    init_tokens = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]

    beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
        init_tokens, beam_state, cache, init_tensor
    )

    state = decoder.select_state(beam_state, 0)

    if rnnlm:
        beam_lm_states, beam_lm_scores = rnnlm.buff_predict(None, beam_lm_tokens, 1)

        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

        lm_layers = len(lm_model.rnn)

        lm_state = select_lm_state(beam_lm_states, 0, lm_type, lm_layers)
        lm_scores = beam_lm_scores[0]
    else:
        lm_state = None
        lm_scores = None

    kept_hyps = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=state,
            y=[beam_y[0]],
            lm_state=lm_state,
            lm_scores=lm_scores,
        )
    ]

    if timer:
        timer.tic("utt total")
    for hi in h:
        if timer:
            timer.tic("utt frame")
        hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
        kept_hyps = []

        h_enc = hi.unsqueeze(0)

        for j in range(len(hyps) - 1):
            for i in range((j + 1), len(hyps)):
                if (
                    is_prefix(hyps[j].yseq, hyps[i].yseq)
                    and (len(hyps[j].yseq) - len(hyps[i].yseq)) <= prefix_alpha
                ):
                    next_id = len(hyps[i].yseq)

                    ytu = F.log_softmax(decoder.joint(hi, hyps[i].y[-1]), dim=0)

                    curr_score = hyps[i].score + float(ytu[hyps[j].yseq[next_id]])

                    for k in range(next_id, (len(hyps[j].yseq) - 1)):
                        ytu = F.log_softmax(decoder.joint(hi, hyps[j].y[k]), dim=0)

                        curr_score += float(ytu[hyps[j].yseq[k + 1]])

                    hyps[j].score = np.logaddexp(hyps[j].score, curr_score)

        S = []
        V = []
        for n in range(nstep):
            beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)
            beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

            if rnnlm:
                beam_lm_scores = torch.stack([hyp.lm_scores for hyp in hyps])

            for i, hyp in enumerate(hyps):
                i_topk = (
                    torch.cat((beam_topk[0][i], beam_logp[i, 0:1])),
                    torch.cat((beam_topk[1][i] + 1, blank_tensor)),
                )

                for logp, k in zip(*i_topk):
                    new_hyp = Hypothesis(
                        yseq=hyp.yseq[:],
                        score=(hyp.score + float(logp)),
                        y=hyp.y[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        lm_scores=hyp.lm_scores,
                    )

                    if k == decoder.blank:
                        S.append(new_hyp)
                    else:
                        new_hyp.yseq.append(int(k))

                        if rnnlm:
                            new_hyp.score += recog_args.lm_weight * float(
                                beam_lm_scores[i, k]
                            )

                        V.append(new_hyp)

            V = sorted(V, key=lambda x: x.score, reverse=True)
            V = substract(V, hyps)[:beam]

            l_state = [v.dec_state for v in V]
            l_tokens = [v.yseq for v in V]

            beam_state = decoder.create_batch_states(beam_state, l_state, l_tokens)
            if timer:
                timer.tic("dec")
            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                V, beam_state, cache, init_tensor
            )
            if timer:
                timer.toc("dec")

            if rnnlm:
                beam_lm_states = create_lm_batch_state(
                    [v.lm_state for v in V], lm_type, lm_layers
                )
                beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                    beam_lm_states, beam_lm_tokens, len(V)
                )

            if n < (nstep - 1):
                for i, v in enumerate(V):
                    v.y.append(beam_y[i])

                    v.dec_state = decoder.select_state(beam_state, i)

                    if rnnlm:
                        v.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )
                        v.lm_scores = beam_lm_scores[i]

                hyps = V[:]
            else:
                beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)

                for i, v in enumerate(V):
                    if nstep != 1:
                        v.score += float(beam_logp[i, 0])

                    v.y.append(beam_y[i])

                    v.dec_state = decoder.select_state(beam_state, i)

                    if rnnlm:
                        v.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )
                        v.lm_scores = beam_lm_scores[i]

        kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]
        if timer:
            timer.toc("utt frame")

    nbest_hyps = sorted(kept_hyps, key=lambda x: (x.score / len(x.yseq)), reverse=True)[
        :nbest
    ]
    if timer:
        timer.toc("utt total")
    return [asdict(n) for n in nbest_hyps]


def search_interface(decoder, h, recog_args, rnnlm, timer=None):
    """Select and run search algorithms.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    if hasattr(decoder, "att"):
        decoder.att[0].reset()

    if recog_args.beam_size <= 1:
        nbest_hyps = greedy_search(decoder, h, recog_args, timer)
    elif recog_args.search_type == "default":
        nbest_hyps = default_beam_search(decoder, h, recog_args, rnnlm, timer)
    elif recog_args.search_type == "nsc":
        nbest_hyps = nsc_beam_search(decoder, h, recog_args, rnnlm, timer)
    elif recog_args.search_type == "tsd":
        nbest_hyps = time_sync_decoding(decoder, h, recog_args, rnnlm, timer)
    elif recog_args.search_type == "alsd":
        nbest_hyps = align_length_sync_decoding(decoder, h, recog_args, rnnlm, timer)
    else:
        raise NotImplementedError

    return nbest_hyps
