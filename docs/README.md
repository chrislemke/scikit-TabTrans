![TabTransformer](https://github.com/chrislemke/scikit-TabTrans/blob/main/docs/assets/images/logo.png)
# scikit-TabTrans
**TabTransformer ready for Scikit learn**

[![DeployPackage](https://github.com/chrislemke/scikit-tabtrans/actions/workflows/deploy-package.yml/badge.svg)](https://github.com/chrislemke/scikit-tabtrans/actions/workflows/deploy-package.yml)
[![Testing](https://github.com/chrislemke/scikit-tabtrans/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/chrislemke/scikit-tabtrans/actions/workflows/testing.yml)

[![pypi](https://img.shields.io/pypi/v/scikit-tabtrans)](https://pypi.org/project/scikit-tabtrans/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-tabtrans)](https://pypistats.org/packages/scikit-tabtrans)
[![python version](https://img.shields.io/pypi/pyversions/scikit-tabtrans?logo=python&logoColor=yellow)](https://www.python.org/downloads/)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://invia-flights.github.io/scikit-tabtrans/)
[![license](https://img.shields.io/github/license/chrislemke/scikit-tabtrans)](https://github.com/chrislemke/scikit-tabtrans/blob/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://github.com/PyCQA/isort)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)

The idea behind this project is to provide the [TabTransformer](https://arxiv.org/pdf/2012.06678.pdf) as a
[Scikit-Learn](https://scikit-learn.org/stable/) model. So it can be e.g. used as a part of a [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html). We use the [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) implementation of the TabTransformer. Checkout [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) it's a great project!
