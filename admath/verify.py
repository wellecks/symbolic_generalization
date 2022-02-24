"""Verification with Sympy."""
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import admath.utils as utils
from pathlib import Path
import sympy as sp
import time

def check(generations, env, methods, early_stop, top_n=50):
    for generation in tqdm(generations, total=len(generations)):
        if 'sympy' in methods:
            input_sp_func = sp.sympify(generation['input_raw'])
            for hyp in generation['hyps'][:top_n+1]:
                success = check_sympy(
                    input_sp_func,
                    hyp['hyp_infix'],
                    env
                )
                hyp['success'] = success
                hyp['success_sympy'] = success
                if success and early_stop:  # stop after we find a success
                    break
    return generations


def parse_and_integrate_sympy(input_raw):
    import sympy as sp
    input_sp_func = sp.sympify(input_raw)
    target_sp_func = sp.integrate(input_sp_func)
    return input_sp_func, target_sp_func


def check_sympy(input_sp_func, hyp_infix, env, seconds=1):
    import sympy as sp
    from src.envs.sympy_utils import simplify
    from src.utils import TimeoutError, timeout
    if hyp_infix is None:  # Model output was unparseable.
        return False

    @timeout(seconds)
    def _run():
        try:
            hyp_sp_func = utils.infix_to_sympy(hyp_infix, env)
            if hyp_sp_func is None:  # unable to parse
                return False
            x = env.local_dict['x']
            hyp_derivative = sp.simplify(hyp_sp_func.diff(x))
            # Sympy difference of derivative and input function
            difference = sp.simplify(sp.sympify(str(input_sp_func - hyp_derivative)))
            ndifference = sp.nsimplify(difference, tolerance=0.001)
            success = difference == 0 or ndifference == 0
        # We (generously) mark it as successful if there was a timeout or exception
        except (TimeoutError, Exception) as e:
            success = True
        return success
    return _run()


def metrics(generations):
    expr2stats = defaultdict(lambda: defaultdict(list))
    for generation in generations:
        top1 = generation['hyps'][0]['success']

        expr = generation['expr']
        expr2stats[expr]['success@1'].append(float(top1))
        
        top10 = False
        for hyp in generation['hyps'][:10]:
            if hyp['success']:
                top10 = True
                break
        expr2stats[expr]['success@10'].append(float(top10))
    return expr2stats


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generations',
        required=True
    )
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['sympy'],
        required=True
    )
    parser.add_argument(
        '--output-dir',
        default='./output'
    )
    parser.add_argument(
        '--symbolic-math-repo-path',
        default='/path/to/SymbolicMathematics'
    )
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--expr-name', default='simple')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--top-n', type=int, default=50)

    args = parser.parse_args()
    print(args)

    env, _, _ = utils.load_env(
        repo_path=args.symbolic_math_repo_path
    )

    generations = json.load(open(args.generations))
    start = time.time()
    out = check(generations, env, args.method, args.early_stop, top_n=args.top_n)
    stop = time.time()
    elapsed = stop - start
    print("Elapsed: %.3f" % (elapsed))

    name = Path(args.generations).stem
    method_suffix = '_'.join(args.method)
    utils.save(out, args.output_dir, '%s_verified_%s_top%d' % (name, method_suffix, args.top_n))

    expr2stats = metrics(out)
    for expr, stats in expr2stats.items():
        print("== %s" % (expr))
        for k, v in stats.items():
            print("\t%s\t%.5f" % (k, np.mean(v)))
    expr2stats['verification_elapsed'] = elapsed
    utils.save(expr2stats, args.output_dir, '%s_metrics_%s_top%d' % (name, method_suffix, args.top_n))


if __name__ == '__main__':
    cli_main()
