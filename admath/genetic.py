"""Simple genetic algorithm for finding difficult problems.

`part, inpart` functions from: https://www.gandreoliva.org/papers/sympy-expressions.html
"""

import os
import random
import numpy as np
import sympy as sp
import torch
from tqdm import trange, tqdm
import admath.utils as utils
import argparse
from collections import defaultdict
import transformers
from sklearn.cluster import KMeans
transformers.logging.set_verbosity_error()


# --- settings
OPS = {
    'base': {
        sp.core.add.Add,
        sp.core.power.Pow,
        sp.core.mul.Mul
    },
    'trig': {
        sp.core.add.Add,
        sp.core.mul.Mul,
        sp.functions.cos,
        sp.functions.sin,
        sp.functions.tan,
    }
}

MUTATION_PARAMS = {
    'robustness': {
        'internal': {
            'symbol': 0.0,
            'constant': 1.0,
            'add_arg': 0.0,
            'op': 0.0
        },
        'leaf': {
            'symbol': 0.0,
            'constant': 1.0,
            'op': 0.0
        },
        'vmin': -100,
        'vmax': 100,
    },
    'distance': {
        'internal': {
            'symbol': 0.25,
            'constant': 0.25,
            'add_arg': 0.25,
            'op': 0.25
        },
        'leaf': {
            'symbol': 0.33,
            'constant': 0.33,
            'op': 0.34
        },
        'vmin': -100,
        'vmax': 100
    },
    'uniform': {
        'internal': {
            'symbol': 0.25,
            'constant': 0.25,
            'add_arg': 0.25,
            'op': 0.25
        },
        'leaf': {
            'symbol': 0.33,
            'constant': 0.33,
            'op': 0.34
        },
        'vmin': -1000,
        'vmax': 1000
    },
}

# --- utilities
def find_depth(expr):
    if len(expr.args) == 0:
        return 1
    return 1 + max([find_depth(x) for x in expr.args])


def random_path(expr, max_path_length):
    path = []
    curr = expr

    while max_path_length > 0:
        if len(curr.args) == 0:  # no children
            break
        child_id = np.random.choice(len(curr.args))
        path.append(child_id)
        curr = curr.args[child_id]
        max_path_length -= 1
    return path


def part(expr, address):
    for num in address:
        expr = expr.args[num]
    return expr


def inpart(expr, repl, address):
    if len(address) == 1:
        largs = list(expr.args)
        largs[address[0]] = repl
        return expr.func(*largs)
    else:
        largs = list(expr.args)
        largs[address[0]] = inpart(expr.args[address[0]],repl,address[1:])
        new = expr.func(*largs)
    return new


def random_op(cmin=0, cmax=10, emin=1, emax=5):
    f_ = '%d %s x ** (%d)' % (
        np.random.randint(cmin, cmax),
        random.choice(['*', '**', '/']),
        np.random.randint(emin, emax)
    )
    new_node = sp.sympify(f_, evaluate=False)
    return new_node


# --- mutations
def mutate(
        expr,
        depth,
        mutation_params,
        ops,
):
    leaf_mutate_probs = mutation_params['leaf']
    internal_mutate_probs = mutation_params['internal']
    vmin = mutation_params['vmin']
    vmax = mutation_params['vmax']
    if depth == 1:
        vi = expr
    else:
        path = random_path(expr, np.random.randint(1, depth))
        vi = part(expr, path)

    if isinstance(vi, sp.functions.cos) or isinstance(vi, sp.functions.sin) or isinstance(vi, sp.functions.tan):
        return expr

    if len(vi.args) == 0:  # leaf
        choices = list(leaf_mutate_probs.keys())
        probs = list(leaf_mutate_probs.values())
        choice = np.random.choice(choices, p=probs)

        # Replace the node with a constant
        if choice == 'constant':
            vnew = sp.core.numbers.Integer(np.random.randint(vmin, vmax))
        if choice == 'symbol':
            # Replace the node with a symbol with a random multiplier
            vnew = sp.sympify('%d*x' % np.random.randint(vmin, vmax), evaluate=False)
        if choice == 'op':
            # Replace the node with a random simple operation
            vnew = random_op(cmin=vmin, cmax=vmax, emin=1, emax=3)
    else:  # internal node
        choices = list(internal_mutate_probs.keys())
        probs = list(internal_mutate_probs.values())
        choice = np.random.choice(choices, p=probs)

        if choice == 'symbol':
            vnew = sp.sympify('x', evaluate=False)
        if choice == 'constant':
            vnew = sp.core.numbers.Integer(np.random.randint(1000))
        if choice == 'op':
            other_ops = list(OPS['base'].difference(set([type(vi)])))
            new_op = other_ops[np.random.choice(len(other_ops))]
            if len(vi.args) == 1:
                new_args = [vi, vi.args[0]]
            else:
                new_args = vi.args
            vnew = new_op(*new_args)
        if choice == 'add_arg':
            new_arg = random_op(cmin=vmin, cmax=vmax, emin=1, emax=3)
            new_args = vi.args + (new_arg,)
            if len(vi.args) == 1:
                new_args = [sp.core.add.Add(*new_args)]
            vnew = vi.func(*new_args)

    if depth == 1:
        expr_new = vnew
    else:
        expr_new = inpart(expr, vnew, path)
    return expr_new


def _validate(x):
    x_str = str(x)
    if 'zoo' in x_str:  # odd case that happened
        return False

    # prevent very large integers
    def _reasonable_int(expr):
        if len(expr.args) == 0 and isinstance(expr, sp.core.numbers.Integer):
            return -1e7 < int(expr) < 1e7
        elif len(expr.args) == 0:
            return True
        return min([_reasonable_int(x) for x in expr.args])

    def _reasonable_length(x_str):
        return len(x_str) < 100

    if not _reasonable_int(x):
        return False

    if not _reasonable_length(x_str):
        return False

    return True


def mutate_generation(generation, mutation_params, ops, n=100):
    from src.utils import TimeoutError, timeout

    problems = []
    for problem in tqdm(generation, total=len(generation)):
        expr = problem['x']
        depth = find_depth(expr)
        for _ in range(n):
            @timeout(1)
            def _mutate():
                try:
                    x = sp.simplify(mutate(expr, depth, mutation_params, ops))
                    if _validate(x):
                        problems.append({
                            'x': x,
                        })
                except TimeoutError:
                    pass
            _mutate()
    return problems


def run_and_check(problems, env, encoder, decoder, device, fitness_type, seconds=2, **kwargs):
    from src.utils import TimeoutError, timeout
    x = env.local_dict['x']
    for problem in tqdm(problems, total=len(problems)):
        problem['hyp'] = None
        problem['hyp_infix'] = None
        problem['hyp_derivative'] = None
        problem['difference'] = 0
        problem['ndifference'] = 0

        @timeout(seconds)
        def _run_single():
            try:
                out = utils.run(
                    str(problem['x']),
                    env, encoder, decoder, 1, device
                )
                if out is not None:  # problem parsed successfully
                    input_prefix, results = out
                    problem['hyp_toks'] = results['hyps'][0]['hyp_toks']
                    hyp = results['hyps'][0]['hyp_infix']
                    hyp_sp = utils.infix_to_sympy(hyp, env)
                    if hyp_sp is None:  # model produced an unparseable output; mark as incorrect.
                        problem['difference'] = -1
                        problem['ndifference'] = -1
                    else:
                        problem['hyp'] = hyp_sp
                        problem['hyp_derivative'] = sp.simplify(hyp_sp.diff(x))
                        problem['difference'] = sp.simplify(sp.sympify(str(problem['x'] - problem['hyp_derivative'])))
                        problem['ndifference'] = sp.nsimplify(problem['difference'], tolerance=0.001)
            # We (generously) mark it as successful if there was a timeout
            except TimeoutError as e:
                problem['difference'] = 0
                problem['ndifference'] = 0
        _run_single()

        correct = problem['difference'] == 0 or problem['ndifference'] == 0
        problem_string = str(problem['x'])
        if fitness_type == 'length_penalty':
            fitness = _fitness_length_penalty(correct, problem_string)
        elif fitness_type == 'target_length':
            fitness = _fitness_target_length(correct, problem_string, kwargs['target_length'])
        elif fitness_type == 'trig_length_penalty':
            fitness = _fitness_length_penalty(correct, problem_string, string_filter=['sin', 'cos', 'tan'])
        elif fitness_type == 'target_problems':
            fitness = _fitness_target_problems(
                correct,
                problem_string,
                target_problem_strings=kwargs['target_problem_strings'],
                model=kwargs['model'],
                tokenizer=kwargs['tokenizer']
            )
        else:
            raise NotImplementedError(fitness_type)

        problem['fitness'] = fitness
    return problems


def _fitness_target_length(correct, problem_string, target_length):
    denom = np.abs(len(problem_string) - target_length)
    if denom == 0:
        denom = 1
    else:
        denom += 1  # ensures penalty when the distance is off by 1
    length_penalty = 1.0 / denom
    fitness = float(not correct) * length_penalty
    return fitness


def _fitness_length_penalty(correct, problem_string, string_filter=[]):
    if len(string_filter) > 0:
        found = False
        for token in string_filter:
            if token in problem_string:
                found = True
        if not found:
            return 0.0
    length_penalty = 1.0 / len(problem_string)
    fitness = float(not correct) * length_penalty
    return fitness


def _fitness_target_problems(correct, problem_string, target_problem_strings, model, tokenizer):
    if correct:
        return 0.0
    problems = [problem_string] + target_problem_strings
    model_inputs = tokenizer(problems, padding="longest", return_tensors="pt")
    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
    with torch.no_grad():
        out = model(**model_inputs)
        embeddings = out.pooler_output

        problem_emb = embeddings[:1]
        target_embs = embeddings[1:]
        sims = torch.cosine_similarity(problem_emb, target_embs)
        fitness = sims.max()
    return fitness


def _format_for_archive(problem, generation_num):
    return {
        'x': str(problem['x']),
        'fitness': problem['fitness'],
        'generation_num': generation_num,
        'hyp': str(problem['hyp']),
        'hyp_derivative': str(problem['hyp_derivative']),
        'difference': str(problem['difference']),
        'ndifference': str(problem['ndifference']),
    }


def _select_fitness(generation, seed_size, **kwargs):
    # Select next seed set as the highest-fitness novel problems.
    seed = sorted(
        filter(lambda x: x['fitness'] >= kwargs['tau'] and x['novel'], generation),
        key=lambda x: -x['fitness']
    )[:seed_size]
    return seed


def _select_kmeans(generation, seed_size, model, tokenizer, **kwargs):
    # filter by novelty
    generation = [x for x in generation if x['novel']]

    # Select highest-fitness problems from k different clusters
    problems = [str(x['x']) for x in generation]
    problem2item = {str(x['x']): x for x in generation}
    model_inputs = tokenizer(problems, padding="longest", return_tensors="pt")
    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
    with torch.no_grad():
        out = model(**model_inputs)
        embeddings = out.pooler_output

    num_clusters = min(len(generation), kwargs['num_clusters'])
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings.cpu().numpy())
    cluster_assignment = clustering_model.labels_
    clustered_problems = [[] for i in range(num_clusters)]
    for id_, cluster_id in enumerate(cluster_assignment):
        clustered_problems[cluster_id].append(problems[id_])

    seed = []
    seed_per_cluster = seed_size // kwargs['num_clusters']
    for i, cluster in enumerate(clustered_problems):
        items = [problem2item[problem] for problem in cluster]
        ranked = sorted(items, key=lambda x: -x['fitness'])
        seed.extend(ranked[:seed_per_cluster])
    return seed


def _initial_seed(basis):
    # Initial generation
    x = sp.sympify('x', evaluate=False)
    if basis == 'polynomial':
        seed = [{'x': sp.sympify('1')}, {'x': x}, {'x': x + 1}, {'x': x ** 2 + x + 1}]
    elif basis == 'polynomial_robustness':
        seed = [
            {'x': sp.sympify('1')},
            {'x': 2 * x},
            {'x': 2 / x},
            {'x': 2 * x + 1},
            {'x': 2 / x + 1},
            {'x': 2 * x ** 2 + 2 * x + 1},
            {'x': 2 * x ** 2 + 2 / x + 1},
            {'x': 2 * x ** 3 + 2 * x ** 2 + 1},
            {'x': 2 * x ** 42 + 2 * x ** 3 + 2 * x ** 2 + 1},
        ]
    elif basis == 'x':
        seed = [{'x': x}]
    elif basis == 'trig_robustness':
        seed = [
            {'x': 17 * sp.cos(83 * x)},
            {'x': 34 * sp.sin(77 * x)},
            {'x': 17 * sp.cos(83 * x) + 1},
            {'x': 34 * sp.sin(77 * x) + 1},
            {'x': 2 * sp.cos(2 * x) + 2 * x},
            {'x': 2 * sp.sin(2 * x) + 2 * x},
            {'x': 2 * sp.cos(2 * x) + 2 * x + 1},
            {'x': 2 * sp.sin(2 * x) + 2 * x + 1},
            {'x': 2 * sp.sin(2 * x) * sp.cos(2 * x)},
        ]
    elif basis == 'trig':
        seed = [
            {'x': 2 * sp.cos(2 * x)},
            {'x': 2 * sp.sin(2 * x)},
            {'x': 2 * sp.cos(2 * x) + 1},
            {'x': 2 * sp.sin(2 * x) + 1},
            {'x': 2 * sp.cos(2 * x) + 2 * x},
            {'x': 2 * sp.sin(2 * x) + 2 * x},
            {'x': 2 * sp.cos(2 * x) + 2 * x + 1},
            {'x': 2 * sp.sin(2 * x) + 2 * x + 1},
            {'x': 2 * sp.sin(2 * x) * sp.cos(2 * x)},
        ]
    elif basis == 'target':
        seed = [
            {'x': sp.S(x)} for x in _get_target_problems()
        ]
    else:
        raise NotImplementedError()
    return seed


def _get_distance_model():
    model = transformers.AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
    model.cuda()
    return model, tokenizer


def _get_target_problems():
    problems = ['-x - 3 + sqrt(3)',
     'x + (sqrt(2)*sqrt(x)/4 + 6*x + 3/x)**2',
     '-x**2 + x + log(4)*tan(x)',
     '4*x**5',
     'x/(x*(2*x + (x**2 + x)/x) + x)',
     'sqrt(3*x + 3) - 2',
     'x**2*exp(-2*x)',
     'tan(exp(2))/(18*x)',
     '-3*x**3 + x + 10/x',
     'x/(log(cos(5)) + 1/(2*x**2))']
    return problems


def genetic_algorithm(
        env, encoder, decoder, device, output_dir, mutation_params, fitness_type, ops,
        seed_selection='kmeans',
        basis='polynomial',
        tau=0.1,
        archive_size_terminate=1000,
        generation_size=200,
        seed_size=20,
        max_generations=100,
        target_length=None
):
    # Scibert model used for equation distance
    model, tokenizer = _get_distance_model()
    target_problems = _get_target_problems()

    # Stores novel, high-fitness problems.
    archive = {}

    # Run the algorithm until we generate enough difficult problems.
    seed = _initial_seed(basis)
    generation_num = 1
    stats = defaultdict(list)
    seed0 = seed
    while len(archive) < archive_size_terminate:
        print("=== Starting generation %d" % generation_num)

        # Mutate the seed set to obtain `generation_size` problems.
        n = generation_size // len(seed)
        generation = mutate_generation(seed, mutation_params, ops, n=n)

        # Evaluate fitness of the generation (via integrating and checking the prediction).
        generation = run_and_check(
            generation, env, encoder, decoder, device, fitness_type,
            target_length=target_length,
            target_problem_strings=target_problems,
            model=model,
            tokenizer=tokenizer
        )

        # Add novel, high-fitness problems to the archive.
        for p in generation:
            p['novel'] = False
            if p['fitness'] >= tau:
                key = str(p['x'])
                if key not in archive:
                    archive[key] = _format_for_archive(p, generation_num)
                    p['novel'] = True

        # Select next seed set.
        if seed_selection == 'fitness':
            seed = _select_fitness(generation, seed_size, tau=tau)
        elif seed_selection == 'kmeans':
            seed = _select_kmeans(generation, seed_size, model, tokenizer, num_clusters=10)
        else:
            raise NotImplementedError(seed_selection)
        seed = seed + seed0

        # --- logging, etc.
        stats['problems_found'].append(len([p for p in generation if p['fitness'] > 0]))
        stats['fitness_found'].append(len([p for p in generation if p['fitness'] >= tau]))
        novel_found = [p for p in generation if p['fitness'] >= 0 and p['novel']]
        stats['novel_found'].append(len(novel_found))
        stats['novel_found_length'].append(np.mean([len(str(p['x'])) for p in novel_found]))
        stats['seed_size'].append(len(seed))
        stats['archive_size'].append(len(archive))
        stats['avg_length'].append([len(key) for key in archive])
        print("\t%d problems found\t%d meeting fitness threshold\t%d novel\t%.3f length" % (
            stats['problems_found'][-1],
            stats['fitness_found'][-1],
            stats['novel_found'][-1],
            stats['novel_found_length'][-1]
        ))
        print("\tSeed size %d\tArchive size %d" % (len(seed), len(archive)))
        utils.save({'archive': archive, 'stats': stats}, output_dir, 'genetic_gen%d' % generation_num)

        generation_num += 1
        if generation_num == max_generations:
            break

        if len(seed) == 0:
            print("Ending early: no seed functions")
            break

    return archive


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-size-terminate', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--seed-size', type=int, default=100)
    parser.add_argument('--generation-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--ops', default='base', choices=['base', 'trig'])
    parser.add_argument('--seed-selection', default='kmeans', choices=['fitness', 'kmeans'])
    parser.add_argument('--basis', default='polynomial', choices=['polynomial', 'x', 'polynomial_robustness', 'trig', 'trig_robustness', 'target'])
    parser.add_argument('--mutation-params', default='uniform', choices=['uniform', 'robustness', 'distance'])
    parser.add_argument(
        '--fitness-type',
        default='length_penalty',
        choices=['length_penalty', 'trig_length_penalty', 'target_length', 'target_nodes', 'target_problems']
    )
    parser.add_argument('--target-length', default=None, type=int)
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
        default='/path/to/SymbolicMathematics',
        help="Lample and Charton code repo"
    )
    parser.add_argument('--expr-name', default='genetic')

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

    genetic_algorithm(
        env, encoder, decoder, device, args.output_dir, MUTATION_PARAMS[args.mutation_params], args.fitness_type, OPS[args.ops],
        tau=args.tau,
        archive_size_terminate=args.archive_size_terminate,
        generation_size=args.generation_size,
        seed_size=args.seed_size,
        seed_selection=args.seed_selection,
        basis=args.basis,
        max_generations=100,
        target_length=args.target_length
    )


if __name__ == '__main__':
    cli_main()
