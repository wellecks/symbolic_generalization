import sys
import sympy as sp
import torch
import os
import json
import numpy as np
from tqdm import tqdm


def load_env(repo_path, model_path='', cpu=False):
    sys.path.append(repo_path)
    from src.utils import AttrDict
    from src.envs import build_env
    from src.model import build_modules

    from src.utils import to_cuda
    from src.envs.sympy_utils import simplify

    # Using parameters from the authors' notebook
    params = {
        # environment parameters
        'env_name': 'char_sp',
        'int_base': 10,
        'balanced': False,
        'positive': True,
        'precision': 10,
        'n_variables': 1,
        'n_coefficients': 0,
        'leaf_probs': '0.75,0,0.25,0',
        'max_len': 512,
        'max_int': 5,
        'max_ops': 15,
        'max_ops_G': 15,
        'clean_prefix_expr': True,
        'rewrite_functions': '',
        'tasks': 'prim_fwd',
        'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',

        # model parameters
        'cpu': cpu,
        'emb_dim': 1024,
        'n_enc_layers': 6,
        'n_dec_layers': 6,
        'n_heads': 8,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': model_path
    }
    params = AttrDict(params)
    env = build_env(params)
    modules = build_modules(env, params)
    encoder = modules['encoder']
    decoder = modules['decoder']
    return env, encoder, decoder


def parse(func_infix, env):
    from src.envs.char_sp import InvalidPrefixExpression, UnknownSymPyOperator
    try:
        func = sp.sympify(func_infix)
        func_prefix = env.sympy_to_prefix(func)
    except (InvalidPrefixExpression, UnknownSymPyOperator):
        return None
    return func_prefix


def infix_to_sympy(infix, env):
    from src.envs.char_sp import InvalidPrefixExpression, UnknownSymPyOperator, ValueErrorExpression
    if infix is None:
        return None
    try:
        out = sp.simplify(env.infix_to_sympy(infix))
    except (InvalidPrefixExpression, UnknownSymPyOperator, ValueErrorExpression, OverflowError):
        # If the authors' `infix_to_sympy` code fails, fall back to calling `sympify`
        try:
            out = sp.sympify(infix)
        except:
            return None
    return out


def generate(input_prefix, top_n, env, encoder, decoder, device, beam_size=10, include_logp=True):
    x1_prefix = env.clean_prefix(['sub', 'derivative', 'f', 'x', 'x'] + input_prefix)
    x1 = torch.LongTensor(
        [env.eos_index] +
        [env.word2id[w] for w in x1_prefix] +
        [env.eos_index],
    ).view(-1, 1).to(device)
    len1 = torch.LongTensor([len(x1)]).to(device)
    x1, len1 = (x1, len1)

    with torch.no_grad():
        encoded = encoder('fwd', x=x1, lengths=len1, causal=False).transpose(0, 1)

    with torch.no_grad():
        _, _, beam = decoder.generate_beam(
            encoded, len1,
            beam_size=beam_size,
            length_penalty=1.0,
            early_stopping=1,
            max_len=200
        )
        assert len(beam) == 1
        hypotheses = beam[0].hyp
        assert len(hypotheses) == beam_size
        if include_logp:
            gen_len = len1.new(beam_size)
            best = []
            for i, (score, sent) in enumerate(beam[0].hyp):
                gen_len[i] = len(sent) + 1  # +1 for the <EOS> symbol
                best.append(sent)
            generated = len1.new(gen_len.max().item(), beam_size).fill_(decoder.pad_index)
            for i, hypo in enumerate(best):
                generated[:gen_len[i] - 1, i] = hypo
                generated[gen_len[i] - 1, i] = decoder.eos_index
            encoded = encoded.expand(beam_size, encoded.size(1), encoded.size(2))
            len1 = len1.expand(beam_size, -1).view(-1)
            log_ps = score_generated(decoder, generated, gen_len, encoded, len1)
            for i, (score, sent) in enumerate(hypotheses):
                hypotheses[i] = (score, sent, log_ps[i].item())

    hyps = []
    for i, hyp in enumerate(sorted(hypotheses, key=lambda item: item[0], reverse=True)[:top_n]):
        score = hyp[0]
        sent = hyp[1]
        log_p = hyp[2] if include_logp else 0.0
        ids = sent[1:].tolist()  # decoded token IDs
        hyp_toks = [env.id2word[wid] for wid in ids]  # convert to prefix
        try:
            hyp_infix = env.prefix_to_infix(hyp_toks)  # convert to infix
        except:
            hyp_infix = None  # Model produced an unparseable output.
        hyps.append({
            'hyp_infix': hyp_infix,
            'hyp_toks': hyp_toks,
            'score': score,
            'log_p': log_p,
            'rank': i,
            'success': None  # This will be filled in during verification.
        })
    result = {
        'hyps': hyps
    }
    return result


def score_generated(decoder, generated, gen_len, src_enc, src_len):
    # input batch
    bs = len(src_len)
    assert src_enc.size(0) == bs

    # positions
    positions = src_len.new(generated.size(0)).long()
    positions = torch.arange(generated.size(0), out=positions).unsqueeze(1).expand(generated.size(0), bs)

    tensor = decoder.forward(
        'fwd',
        x=generated,
        lengths=gen_len,
        positions=positions,
        causal=True,
        src_enc=src_enc,
        src_len=src_len,
    )
    log_ps = decoder.proj(tensor).log_softmax(-1)
    mask = torch.arange(log_ps.size(0)).expand(len(gen_len), log_ps.size(0)).to(log_ps.device) < gen_len.unsqueeze(1)
    mask = mask.transpose(0, 1).contiguous().float()
    mask = mask[1:]
    generated = generated[1:]
    log_ps = log_ps[:-1]
    T, B, V = log_ps.size()
    log_ps = log_ps.view(-1, V).gather(1, generated.view(-1, 1)).view(T, B)
    log_ps = (log_ps*mask).sum(0)
    return log_ps


def generate_sample(input_prefix, top_n, env, encoder, decoder, device, temperature=1.0):
    x1_prefix = env.clean_prefix(['sub', 'derivative', 'f', 'x', 'x'] + input_prefix)
    x1 = torch.LongTensor(
        [env.eos_index] +
        [env.word2id[w] for w in x1_prefix] +
        [env.eos_index],
    ).view(-1, 1).to(device)
    len1 = torch.LongTensor([len(x1)]).to(device)
    x1, len1 = (x1, len1)

    with torch.no_grad():
        encoded = encoder('fwd', x=x1, lengths=len1, causal=False).transpose(0, 1)
        encoded = encoded.expand(top_n, encoded.size(1), encoded.size(2))
        len1 = len1.expand(top_n, -1).view(-1)

    with torch.no_grad():
        generated, gen_len = decoder.generate(
            encoded, len1,
            sample_temperature=temperature,
            max_len=200
        )
        scores = score_generated(decoder, generated, gen_len, encoded, len1)
        hypotheses = [(scores[i].item(), generated[1:gen_len[i]-1, i]) for i in range(generated.size(1))]

    hyps = []
    for i, (score, sent) in enumerate(sorted(hypotheses, key=lambda item: item[0], reverse=True)[:top_n]):
        ids = sent.tolist()
        hyp_toks = [env.id2word[wid] for wid in ids]  # convert to prefix
        try:
            hyp_infix = env.prefix_to_infix(hyp_toks)  # convert to infix
        except:
            hyp_infix = None  # Model produced an unparseable output.
        hyps.append({
            'hyp_infix': hyp_infix,
            'hyp_toks': hyp_toks,
            'score': score,
            'log_p': score,
            'rank': i,
            'success': None  # This will be filled in during verification.
        })
    result = {
        'hyps': hyps
    }
    return result


def run_sample(f_, env, encoder, decoder, top_n, device, temperature=1.0):
    input_prefix = parse(
        func_infix=f_,
        env=env
    )
    if input_prefix is None:  # Unable to parse, skip.
        return None
    results = generate_sample(
        input_prefix=input_prefix,
        top_n=top_n,
        env=env,
        encoder=encoder,
        decoder=decoder,
        device=device,
        temperature=temperature
    )
    return input_prefix, results


def run(f_, env, encoder, decoder, top_n, device, beam_size=10, include_logp=True):
    input_prefix = parse(
        func_infix=f_,
        env=env
    )
    if input_prefix is None:  # Unable to parse, skip.
        return None
    results = generate(
        input_prefix=input_prefix,
        top_n=top_n,
        env=env,
        encoder=encoder,
        decoder=decoder,
        device=device,
        beam_size=beam_size,
        include_logp=include_logp
    )
    return input_prefix, results


def save(results, output_dir, expr_name):
    os.makedirs(output_dir, exist_ok=True)
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    filename = os.path.join(
        output_dir, '%s.json' % (expr_name)
    )
    with open(filename, 'w') as f:
        json.dump(results, f, default=np_encoder)
    print("Wrote to %s" % filename)


def run_and_check(problems, env, encoder, decoder, device, top_n, seconds=10, beam_size=10, **kwargs):
    from src.utils import TimeoutError, timeout
    x = env.local_dict['x']
    for problem in tqdm(problems, total=len(problems)):
        problem['hyp'] = None
        problem['hyp_derivative'] = None
        problem['difference'] = 1
        problem['ndifference'] = 1
        problem['correct'] = False
        problem['cancelled'] = False

        @timeout(seconds)
        def _run():
            try:
                out = run(
                    str(problem['x']),
                    env, encoder, decoder, top_n, device,
                    beam_size=beam_size
                )
                if out is not None:  # problem parsed successfully
                    input_prefix, results = out
                    for hyp_result in results['hyps']:
                        hyp = hyp_result['hyp_infix']
                        hyp_sp = infix_to_sympy(hyp, env)
                        if hyp_sp is None:  # model produced an unparseable output; mark as incorrect.
                            problem['difference'] = -1
                            problem['ndifference'] = -1
                        else:
                            problem['hyp'] = hyp_sp
                            problem['hyp_derivative'] = sp.simplify(hyp_sp.diff(x))
                            problem['difference'] = sp.simplify(sp.sympify(str(problem['x'] - problem['hyp_derivative'])))
                            problem['ndifference'] = sp.nsimplify(problem['difference'], tolerance=0.001)

                        # if we found a successful result, stop early
                        correct = problem['difference'] == 0 or problem['ndifference'] == 0
                        if correct:
                            problem['correct'] = True
                            break
            # We (generously) mark it as successful if there was a timeout or exception
            except (TimeoutError, Exception) as e:
                problem['difference'] = 0
                problem['ndifference'] = 0
                problem['correct'] = True
                problem['cancelled'] = True
        _run()

    return problems

