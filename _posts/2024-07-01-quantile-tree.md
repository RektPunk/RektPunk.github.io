---
layout: post
title: MQBoost; Quantile estimator preserving monotonicity among quantiles
description: LightGBM, XGBoost 기반의 quantile regressor 패키지 개발
tags: 'Quantile-regression LightGBM XGBoost'
categories: [Machine learning, Quantile regression]
date: '2024-07-01'
---

### TL;DR
이번 글에서는 앞선 두 모델 [LightGBM 글](../mqr-lgb), [XGBoost 글](../mqr-xgboost)을 하나로 묶어 패키지화 한 내용을 다룬다.
코드는 [RektPunk/mqboost](https://github.com/RektPunk/mqboost)에 작성해두었다.

### How?
두 모델은 서로 많은 것을 공유한다. Data 준비하는 과정 [(`utils.py`)](https://github.com/RektPunk/mqboost/blob/main/mqboost/utils.py), objective function [(`objective.py`)](https://github.com/RektPunk/mqboost/blob/main/mqboost/objective.py)은 완벽하게 같고, constraints 할당하는 방법, train 과정이 미묘하게 다르다. 그래서 고민 끝에 먼저 [`abtract.py`](https://github.com/RektPunk/mqboost/blob/main/mqboost/abstract.py)에 먼저 parent class를 만들어서 `_model_name` 이라는 입력을 받도록 했다. `_model_name`은 enum 처리로 "lightgbm", "xgboost" 중 하나를 입력으로 받도록 했다.


```python
class _ModelName(str, Enum):
    lightgbm: str = "lightgbm"
    xgboost: str = "xgboost"

class MonotoneQuantileRegressor:
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
        _model_name: _ModelName,
    ):
        self._model_name = _model_name
        self.x_train, self.y_train = prepare_train(x, y, alphas)
        self.fobj = partial(check_loss_grad_hess, alphas=alphas)
        self.feval = partial(check_loss_eval, alphas=alphas)
        self.dataset = TRAIN_DATASET_FUNCS.get(self._model_name)(
            data=self.x_train, label=self.y_train
        )
```
`TRAIN_DATASET_FUNCS`은 딕셔너리로 아래처럼 처리하여 모델에 따라 데이터 형태를 선택하게끔 했다.
```python
TRAIN_DATASET_FUNCS: Dict[str, Union[lgb.Dataset, xgb.DMatrix]] = {
    "lightgbm": lgb.Dataset,
    "xgboost": xgb.DMatrix,
}
```
다음으로 train 에서는 `params` 만 update 해주도록 구성하고, 각각의 constraints type에 맞도록 변경해줬다. predict는 입력값을 각 모델마다 필요한 형태로 변환하여 모델을 거치고 예측값을 출력하도록 구성했다.
```python
MONOTONE_CONSTRAINTS_TYPE: Dict[str, Union[list, tuple]] = {
    "lightgbm": list,
    "xgboost": tuple,
}

PREDICT_DATASET_FUNCS: Dict[str, Union[Callable, xgb.DMatrix]] = {
    "lightgbm": lambda x: x,
    "xgboost": xgb.DMatrix,
}


class MonotoneQuantileRegressor:
    ...
    def train(self, params: Dict[str, Any]):
        self._params = params.copy()
        monotone_constraints_str: str = "monotone_constraints"
        if monotone_constraints_str in self._params:
            _monotone_constraints = list(self._params[monotone_constraints_str])
            _monotone_constraints.append(1)
            self._params[monotone_constraints_str] = MONOTONE_CONSTRAINTS_TYPE.get(
                self._model_name
            )(_monotone_constraints)
        else:
            self._params.update(
                {
                    monotone_constraints_str: MONOTONE_CONSTRAINTS_TYPE.get(
                        self._model_name
                    )([1 if "_tau" == col else 0 for col in self.x_train.columns])
                }
            )

    def predict(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ) -> np.ndarray:
        alphas = alpha_validate(alphas)
        _x = prepare_x(x, alphas)
        _x = PREDICT_DATASET_FUNCS.get(self._model_name)(_x)
        _pred = self.model.predict(_x)
        _pred = _pred.reshape(len(alphas), len(x))
        return _pred
```
이제, 얘를 상속 받아 lgb, xgb 만 적용해주면 모델을 쉽게 구성할 수 있다. `__init__` 에서만 `_model_name`을 할당해주고, `train` 에서만 살짝 다르게 적용해주면 끝이다. 참고로 LightGBM 버전 4.0.0 이상에서는 custom objective function을 `params`에 넣어야 적용이 된다. 이전에는 `lgb.train`에서 `fobj`로 주던게 변경 되었나보다.
해당 부분 수정해서 아래처럼 작성해줬다.
```python
class QuantileRegressorLgb(MonotoneQuantileRegressor):
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            _model_name="lightgbm",
        )

    def train(self, params: Dict[str, Any]) -> lgb.basic.Booster:
        super().train(params=params)
        self._params.update({"objective": self.fobj})
        self.model = lgb.train(
            train_set=self.dataset,
            params=self._params,
            feval=self.feval,
        )
        return self.model


class QuantileRegressorXgb(MonotoneQuantileRegressor):
    def __init__(
        self,
        x: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: Union[List[float], float],
    ):
        super().__init__(
            x=x,
            y=y,
            alphas=alphas,
            _model_name="xgboost",
        )

    def train(self, params: Dict[str, Any]) -> xgb.Booster:
        super().train(params=params)
        self.model = xgb.train(
            dtrain=self.dataset,
            verbose_eval=False,
            params=self._params,
            obj=self.fobj,
        )
        return self.model
```
`predict`는 미리 처리해둔 덕분에 따로 신경쓰지 않아도 된다. 추후에 다른 tree를 추가할 때도 구성이 크게 다르지 않으면 쉽게 붙일 수 있을 지도..?

### Upload to pip
구성된 모델을 예전 연구실 동료들과 공유하려다 보니 불편해서 `pip`에 업로드 하기로 결정했다. 먼저 `setup.py` 를 아래와 같이 구성해준다.
```python
from setuptools import setup, find_packages

setup(
    name="quantile-tree",
    version="0.0.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
    ],
    author="RektPunk",
    author_email="rektpunk@gmail.com",
    description="Monotone quantile regressor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RektPunk/monotone-quantile-tree",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
```
다음으로는 다음 명령어를 통해 build 해주고, 
```bash
pip install setuptools wheel
python setup.py sdist bdist_wheel
```
twine을 통해 업로드해주면 끝이다.
```bash
pip install twine
twine upload dist/*
```
[pip 링크](https://pypi.org/project/quantile-tree/)에 업로드 확인하고 설치까지 확인하면 완료!
```bash
pip install quantile-tree
```

마지막으로 직접 배포는 귀찮으니 github action을 통해 태그를 push하면 배포하도록 구성했다.
```yaml
name: Publish Python Package

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
      python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install setuptools wheel twine

    - name: Build package
      run: |
        python setup.py sdist bdist_wheel

    - name: Publish to PyPI
      env:
      TWINE_USERNAME: __token__
      TWINE_PASSWORD: ${ { secrets.PYPI_TOKEN } }
      run: |
        twine upload dist/*
```

### Conclusion
1년전..?에 생각해서 만들어둔 로직인데 최근까지도 비슷한 방법론이나 paper을 찾아보기 어려운 것 같다. 때때로 흥미가 생기면 다른 tree 알고리즘도 구경하면서 적용가능하면 찾아서 업데이트할 계획이다.

### Update
- 목적함수를 변경했다. MM 알고리즘에 사용되는 Approximated huber loss이다. 얘가 성능이 더 괜찮다고 한다.
- Publish 로직을 poetry 도입으로 변경했다.
