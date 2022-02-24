from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='admath',
    version='1.0.0',
    description='adversarial mathematics',
    packages=['admath'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)