"""Compositionality with simple primitives.
Uses successful simple primitives, e.g. computed and verified with
`robustness.py` and `verify.py`.
"""
import argparse
import torch
import numpy as np
import random
import json
from tqdm import tqdm, trange
import admath.utils as utils


def random_tuples(lst, k, n):
    sample = set()
    while len(sample) < n:
        candidate = tuple([random.choice(lst) for _ in range(k)])
        if candidate not in sample:
            sample.add(candidate)
    return list(sample)


def expr_simple_comp_addcoeff(generations, env, encoder, decoder, top_n, device, n=1000, ks=[2,3,4], temperature=None):
    primitives = [
        generation['input_raw'] for generation in generations
        if generation['expr'] == 'simple_coeff1' and generation['hyps'][0]['success']
    ]
    output = []
    beam_size = 10 if top_n <= 10 else top_n
    for operator in ['+']:
        for k in ks:
            lst = list(range(len(primitives)))
            idxs = random_tuples(lst, k, n)
            for idx in tqdm(idxs, total=len(idxs)):
                fs = [primitives[i] for i in idx]
                f_ = operator.join(fs)
                if temperature is None:  # beam search
                    input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
                else:  # sampling
                    input_prefix, results = utils.run_sample(
                        f_, env, encoder, decoder, top_n, device, temperature=temperature
                    )
                output.append({
                    'metadata': {
                        'params': ('+',),
                        'primitives': fs,
                        'top-n': top_n,
                        'decoding_params': {
                            'temperature': temperature,
                            'beam_size': None if temperature is None else beam_size
                        },
                    },
                    'hyps': results['hyps'],
                    'input_raw': f_,
                    'input_prefix': input_prefix,
                    'expr': 'simple_comp_addcoeff_%d' % k
                })
    return output


def expr_simple_comp_addexp(generations, env, encoder, decoder, top_n, device, n=1000, ks=[2,3,4], temperature=None):
    primitives = [
        generation['input_raw'] for generation in generations
        if (generation['expr'] in {'simple_expincr', 'simple_expdecr'}) and generation['hyps'][0]['success']
    ]
    output = []
    beam_size = 10 if top_n <= 10 else top_n
    for operator in ['+']:
        for k in ks:
            lst = list(range(len(primitives)))
            idxs = random_tuples(lst, k, n)
            for idx in tqdm(idxs, total=len(idxs)):
                fs = [primitives[i] for i in idx]
                f_ = operator.join(fs)
                if temperature is None:  # beam search
                    input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
                else:  # sampling
                    input_prefix, results = utils.run_sample(
                        f_, env, encoder, decoder, top_n, device, temperature=temperature
                    )
                output.append({
                    'metadata': {
                        'params': (k,),
                        'primitives': fs,
                        'top-n': top_n,
                        'decoding_params': {
                            'temperature': temperature,
                            'beam_size': None if temperature is None else beam_size
                        },
                    },
                    'hyps': results['hyps'],
                    'input_raw': f_,
                    'input_prefix': input_prefix,
                    'expr': 'simple_comp_addexp_%d' % k
                })
    return output


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generations-file',
        default=[
            './output/simple_expdecr_verified_sympy.json',
            './output/simple_expincr_verified_sympy.json',
            './output/simple_coeff1_verified_sympy.json'
        ],
        nargs='+'
    )
    parser.add_argument(
        '--output-dir',
        default='./output'
    )
    parser.add_argument(
        '--model-path',
        default='/path/to/fwd_bwd_ibp.pth'
    )
    parser.add_argument(
        '--symbolic-math-repo-path',
        default='/path/to/SymbolicMathematics'
    )
    parser.add_argument('--expr-name', default='simple')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--top-n', type=int, default=10)

    args = parser.parse_args()
    print(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env, encoder, decoder = utils.load_env(
        repo_path=args.symbolic_math_repo_path,
        model_path=args.model_path
    )

    generations = []
    for f in args.generations_file:
        generations.extend(json.load(open(f)))

    all_out = []
    out = expr_simple_comp_addcoeff(generations, env, encoder, decoder, args.top_n, device, n=1000)
    utils.save(out, args.output_dir, 'simple_comp_addcoeff')
    all_out.extend(out)

    out = expr_simple_comp_addexp(generations, env, encoder, decoder, args.top_n, device, n=1000)
    utils.save(out, args.output_dir, 'simple_comp_addexp')
    all_out.extend(out)

    utils.save(all_out, args.output_dir, 'simple_comp_all')



if __name__ == '__main__':
    cli_main()
