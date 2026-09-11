from setuptools import setup, find_packages

setup(
    name='champollion_V1',
    version='0.0.1',
    packages=find_packages(
        exclude=['tests*']),
    license='CeCILL license version 2',
    description='Deep learning models for cortical folding analysis',
    long_description=open('README.rst').read(),
    install_requires=['numpy',
                      'pandas',
                      'scipy',
                      'torch',
                      'torchvision',
                      'tensorboard',
                      'hydra-core',
                      'hydra-joblib-launcher',
                      'OmegaConf',
                      'pytorch-lightning==2.1.3',
                      'sparse',
                      ],
    url='https://github.com/neurospin/champollion',
    author='Julien Laval, Joël Chavas, Barthélémy Drabczuk Antoine Dufournet, Aymeric Gaudin',
    author_email='julien.laval@cea.fr, joel.chavas@cea.fr, antoine.dufournet@cea.fr, aymeric.gaudin@cea.fr'
)
