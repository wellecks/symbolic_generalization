import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import admath.utils as utils
from admath.compositionality import random_tuples


def expr_simple_coeff1_range(env, encoder, decoder, top_n, device, N):
    F0 = [
        '%d*ln(%d*x)',
        '%d*exp(%d*x)',
        '%d*sin(%d*x)',
        '%d*cos(%d*x)',
        '%d*tan(%d*x)',
        '%d*x'
    ]
    ranges = [(1, 100), (100, 200), (200, 300), (300, 400), (400, 500), (1000, 2000), (10000, 11000)]
    output = []
    beam_size = 10 if top_n <= 10 else top_n
    for start, end in ranges:
        coeffs = random_tuples(list(range(start, end)), 2, N)
        for a1, a2 in tqdm(coeffs, total=len(coeffs)):
            for f in F0:
                if f == '%d*x':
                    f_ = f % (a1)
                else:
                    f_ = f % (a1, a2)

                input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
                output.append({
                    'metadata': {
                        'params': (a1, a2, start, end),
                        'primitive': f,
                        'top-n': top_n
                    },
                    'hyps': results['hyps'],
                    'input_raw': f_,
                    'input_prefix': input_prefix,
                    'expr': 'simple_coeff1_range'
                })
    return output

F0 = [
    '%d*ln(%d*x)',
    '%d*exp(%d*x)',
    '%d*sin(%d*x)',
    '%d*cos(%d*x)',
    '%d*tan(%d*x)',
    '%d*x',
    '%d*x**42',
]
def expr_simple_coeff1(env, encoder, decoder, top_n, device, N, templates=F0, temperature=None):
    ranges = [(1, 100)]
    output = []
    beam_size = 10 if top_n <= 10 else top_n
    for start, end in ranges:
        for f in templates:
            if f == '%d*x' or f == '%d*x**42':
                coeffs = random.sample(list(range(start, end+1)), min(end-start, N))
            else:
                coeffs = random_tuples(list(range(start, end)), 2, N)
            for coeff in tqdm(coeffs, total=len(coeffs)):
                if f == '%d*x' or f == '%d*x**42':
                    f_ = f % coeff
                else:
                    f_ = f % (coeff[0], coeff[1])

                if temperature is None:  # beam search
                    input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
                else:  # sampling
                    input_prefix, results = utils.run_sample(
                        f_, env, encoder, decoder, top_n, device, temperature=temperature
                    )
                output.append({
                    'metadata': {
                        'params': {'coeffs': coeff, 'range': (start, end)},
                        'decoding_params': {
                            'temperature': temperature,
                            'beam_size': None if temperature is None else beam_size
                        },
                        'primitive': f,
                        'top-n': top_n
                    },
                    'hyps': results['hyps'],
                    'input_raw': f_,
                    'input_prefix': input_prefix,
                    'expr': 'simple_coeff1'
                })
    return output


def expr_simple_coeff2(env, encoder, decoder, top_n, device):
    f = '32*cos(%d*x)'
    coeffs = list(range(2 ** 9, 2 ** 10))

    output = []
    beam_size = 10 if top_n <= 10 else top_n
    for a2 in tqdm(coeffs, total=len(coeffs)):
        f_ = f % (a2)
        input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
        output.append({
            'metadata': {
                'params': (a2,),
                'primitive': f,
                'top-n': top_n
            },
            'hyps': results['hyps'],
            'input_raw': f_,
            'input_prefix': input_prefix,
            'expr': 'simple_coeff2'
        })
    return output


def expr_simple_expincr(env, encoder, decoder, top_n, device):
    output = []
    f = 'x**(%d)'
    xs = list(range(1, 1001)) + list(range(10000, 11000))
    beam_size = 10 if top_n <= 10 else top_n
    for n1 in tqdm(xs):
        f_ = f % (n1)
        input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
        output.append({
            'metadata': {
                'params': (n1,),
                'primitive': f,
                'top-n': top_n
            },
            'hyps': results['hyps'],
            'input_raw': f_,
            'input_prefix': input_prefix,
            'expr': 'simple_expincr'
        })
    return output


def expr_simple_expdecr(env, encoder, decoder, top_n, device):
    output = []
    f = 'x**(1/%d)'
    xs = list(range(1, 1001)) + list(range(10000, 11000))
    beam_size = 10 if top_n <= 10 else top_n
    for n1 in tqdm(xs):
        f_ = f % (n1)
        input_prefix, results = utils.run(f_, env, encoder, decoder, top_n, device, beam_size=beam_size)
        output.append({
            'metadata': {
                'params': (n1,),
                'primitive': f,
                'top-n': top_n
            },
            'hyps': results['hyps'],
            'input_raw': f_,
            'input_prefix': input_prefix,
            'expr': 'simple_expdecr'
        })
    return output


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exprs', nargs='+',
        choices=[
            'simple_coeff1_range', 'simple_coeff1', 'simple_coeff2', 'simple_expincr', 'simple_expdecr', 'all',
        ],
        default=['all'],
    )
    parser.add_argument(
        '--output-dir',
        default='./output'
    )
    parser.add_argument(
        '--model-path',
        default='/path/to/symbolic_mathematics/fwd_bwd_ibp.pth'
    )
    parser.add_argument(
        '--symbolic-math-repo-path',
        default='/path/to/SymbolicMathematics'
    )
    parser.add_argument('--N', type=int, default=500)
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

    out_all = []
    if 'all' in args.exprs or 'simple_coeff1_range' in args.exprs:
        out = expr_simple_coeff1_range(env, encoder, decoder, args.top_n, device, N=args.N)
        utils.save(out, args.output_dir, 'simple_coeff1_range')
        out_all.extend(out)

    if 'all' in args.exprs or 'simple_coeff1' in args.exprs:
        out = expr_simple_coeff1(env, encoder, decoder, args.top_n, device, N=args.N)
        utils.save(out, args.output_dir, 'simple_coeff1')
        out_all.extend(out)

    if 'all' in args.exprs or 'simple_coeff2' in args.exprs:
        out = expr_simple_coeff2(env, encoder, decoder, args.top_n, device)
        utils.save(out, args.output_dir, 'simple_coeff2')
        out_all.extend(out)

    if 'all' in args.exprs or 'simple_expincr' in args.exprs:
        out = expr_simple_expincr(env, encoder, decoder, args.top_n, device)
        utils.save(out, args.output_dir, 'simple_expincr')
        out_all.extend(out)

    if 'all' in args.exprs or 'simple_expdecr' in args.exprs:
        out = expr_simple_expdecr(env, encoder, decoder, args.top_n, device)
        utils.save(out, args.output_dir, 'simple_expdecr')
        out_all.extend(out)

    if 'all' in args.exprs:
        utils.save(out_all, args.output_dir, 'simple_robustness_all')


if __name__ == '__main__':
    cli_main()
