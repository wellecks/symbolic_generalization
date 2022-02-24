## Symbolic Brittleness in Sequence Models: on Systematic Generalization in Symbolic Mathematics

[Symbolic Brittleness in Sequence Models: on Systematic Generalization in Symbolic Mathematics](https://arxiv.org/pdf/2109.13986.pdf)\
Sean Welleck, Peter West, Jize Cao, Yejin Choi\
AAAI 2022

This repo contains code for **automated** (SAGGA) and **rule-based** (robustness, compositionality) failure discovery and verification.

Please cite our work if you found the resources in this repository useful:
```
@inproceedings{welleck2022symbolic,
    title={Symbolic Brittleness in Sequence Models: on Systematic Generalization in Symbolic Mathematics},
    author={Sean Welleck and Peter West and Jize Cao and Yejin Choi},
    booktitle={AAAI},
    year={2022},
    url={https://arxiv.org/pdf/2109.13986.pdf}
}
```

## Setup

#### Library code

We use utilities from the implementation of  [[Lample \& Charton](https://arxiv.org/abs/1912.01412)].
1. Clone their [[repo](https://github.com/facebookresearch/SymbolicMathematics/tree/4596d070e1a9a1c2239c923d7d68fda577c8c007)]. The repo filepath is then provided as a `--symbolic-math-repo-path` command line argument in the scripts below.

Then setup our code by running:
```bash
python setup.py develop
```

#### Data and models

We use pretrained models and data from [[Lample \& Charton](https://arxiv.org/abs/1912.01412)]. \
From their [[repo](https://github.com/facebookresearch/SymbolicMathematics/tree/4596d070e1a9a1c2239c923d7d68fda577c8c007)]:
1. Download and unpack the `FWD + BWD + IBP` model [[link](https://dl.fbaipublicfiles.com/SymbolicMathematics/models/fwd_bwd_ibp.pth)].\
The model path is provided as a `--model-path` command line argument in the scripts below.



For experiments involving validation primitives:

2. Download the Integration `FWD`, `BWD`, `IBP` datasets. 

## SAGGA
#### Robustness
```bash
python admath/genetic.py --basis polynomial_robustness --mutation-params robustness

python admath/genetic.py --basis trig_robustness --mutation-params robustness 
```

#### Robustness - explicit distance
```bash
python admath/genetic.py --fitness-type target_problems --basis target --mutation-params distance
```


#### General / exploits
```bash
python admath/genetic.py --basis polynomial --fitness-type length_penalty

python admath/genetic.py --basis trig --ops trig --fitness-type trig_length_penalty
```


#### Target length
```bash
python admath/genetic.py --fitness-type target_length --target-length 10 --tau 0.05 --seed-size 50 --generation-size 300 --archive-size-terminate 5000

python admath/genetic.py --fitness-type target_length --target-length 20 --tau 0.05 --seed-size 50 --generation-size 300 --archive-size-terminate 5000

python admath/genetic.py --fitness-type target_length --target-length 40 --tau 0.05 --seed-size 50 --generation-size 300 --archive-size-terminate 5000
```


## Simple Primitives

#### Robustness
Generate
```bash
python admath/robustness.py --output-dir ./output

==> Wrote to ./output/simple_robustness_all.json
```

Check and verify
```bash
python admath/verify.py \
--generations ./output/simple_robustness_all_generations.json \ 
--method sympy \
--early-stop \    
--output-dir ./output

==> Wrote to ./output/simple_robustness_all_verified_sympy_sagemath.json
```

#### Compositionality
Compositionality uses verified functions from the Robustness experiment via the `--generations-file` parameter.
```bash
python admath/compositionality.py \
--generations-file ./output/simple_robustness_all_verified_sympy.json

==> Wrote to ./output/simple_comp_all.json
```

Then do the "check and verify" step above.


## Validation Primitives - Robustness and Compositionality

See `notebooks/`.


