import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import admath.utils as utils
from admath.robustness import expr_simple_coeff1


def expr_simple_coeff1_nonzero(env, encoder, decoder, top_n, device, N, temperature):
    return expr_simple_coeff1(
        env, encoder, decoder, top_n, device, N,
        templates=['%d*exp(%d*x)', '%d*sin(%d*x)', '%d*cos(%d*x)', '%d*tan(%d*x)'],
        temperature=temperature
    )


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--expr', choices=['robustness'],
    )
    parser.add_argument(
        '--decoder', choices=['beam', 'sample', 'all'],
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
        default='/home/seanw/projects/SymbolicMathematics'
    )
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--expr-name', default='simple_robustness_largebeam_and_sample')
    parser.add_argument('--seed', type=int, default=43)

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

    if args.expr == 'robustness':
        expr_ = expr_simple_coeff1_nonzero

    if args.decoder in {'beam', 'all'}:
        print("Running beam 500")
        out = expr_(
            env, encoder, decoder,
            top_n=500,
            device=device,
            N=args.N,
            temperature=None
        )
        utils.save(out, args.output_dir, 'simple_%s_largebeam_and_sample__beam500' % args.expr)

    if args.decoder in {'sample', 'all'}:
        for temperature in [0.6, 0.8, 1.0]:
            # sampling
            print("Running sample %.1f" % temperature)
            out = expr_(
                env, encoder, decoder,
                top_n=500,
                device=device,
                N=args.N,
                temperature=temperature
            )
            utils.save(out, args.output_dir, 'simple_%s_largebeam_and_sample__sample%.1f' % (args.expr, temperature))

    print('=== Done.')

if __name__ == '__main__':
    cli_main()

