---
layout: post
title: 본격 Imbalanced 분류 모델 개발기
description: LightGBM+Focal loss 적용해보기
tags: 'LightGBM'
categories: [Machine learning, Imbalanced]
date: '2024-10-28'
---

### TL;DR
LightGBM에 Focal loss를 적용하여 binary, multiclass classification 문제에 사용가능한 모델을 만들었다. Github 레포는 [Imbalance-LightGBM](https://github.com/RektPunk/Imbalance-LightGBM) 이고, LightGBM과 마찬가지로 `train`, `cv`을 지원하고, `scikit-learn` API처럼 `ImbalancedLGBMClassifier` class도 지원한다.

### How?
처음에는 Imbalanced 분류 문제에 관심이 생겨서 조사를 해봤다. 엄청나게 많은 글들, 코드 예시들이 있었지만 정작 구현된 코드는 비슷비슷한 코드고, 특히 많은 자료가 gradient와 hessian을 수치적으로 구하고 있는 점이 건드리고 싶은 욕구를 자극했다. 추가로, `LightGBM` 기반의 패키지로 만들어둔건 없었다. 특히, multiclass 의 경우에는 더더욱 예시조차 찾아보기 힘들었다. One-vs-the-rest (OVR) 관련된 얘기는 있었지만, 글쎄, 클래스 개수만큼 fitting 하는 것 으로는 다소 충분치 않다는 생각이 들었고, 목적함수 조절을 통해 한번의 fitting으로 구현이 가능하지 않을까 하는 생각으로 만들어봤다. Focal loss에 관한 글들은 엄청나게 많으니 원리나 해석 등은 하진 않겠다. 

모델을 건들 때, 제일 어려운 부분은 아무래도 목적함수 부분이다. 운좋게도, 검색을 통해 발견한 `XGBoost`에서 custom objective를 통해서 구현한 오픈소스[Imbalanced-XGBoost](https://github.com/jhwjhw0123/Imbalance-XGBoost)가 있었고, 논문까지 작성되어 있어서 이해와 접근이 쉬웠다. 해당 프로젝트의 [objective](https://github.com/jhwjhw0123/Imbalance-XGBoost/blob/6d82dc83266a32c8579a3e8ea172378085ad1711/imxgboost/focal_loss.py#L21-L44)를 따와서 `_focal_grad_hess` 함수를 만들었다.

```python
def _safe_power(num_base: np.ndarray, num_pow: float) -> np.ndarray:
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _safe_log(array: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    """Safe log."""
    return np.log(np.clip(array, min_value, None))


def _focal_grad_hess(
    y_true: np.ndarray, pred_prob: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return focal grad hess."""
    prob_product = pred_prob * (1 - pred_prob)
    true_diff_pred = y_true + ((-1) ** y_true) * pred_prob
    focal_grad_term = pred_prob + y_true - 1
    focal_log_term = 1 - y_true - ((-1) ** y_true) * pred_prob
    focal_grad_base = y_true + ((-1) ** y_true) * pred_prob
    grad = gamma * focal_grad_term * _safe_power(true_diff_pred, gamma) * _safe_log(focal_log_term) + \
        ((-1) ** y_true) * _safe_power(focal_grad_base, (gamma + 1))

    hess_term1 = _safe_power(true_diff_pred, gamma) + \
        gamma * ((-1) ** y_true) * focal_grad_term * _safe_power(true_diff_pred, (gamma - 1))
    hess_term2 = ((-1) ** y_true) * focal_grad_term * _safe_power(true_diff_pred, gamma) / focal_log_term
    hess = (hess_term1 * _safe_log(focal_log_term) - hess_term2) * gamma + \
        (gamma + 1) * _safe_power(focal_grad_base, gamma)) * prob_product
    return grad, hess
```


남은 건 이 목적함수를 이용하여 binary, multiclass에 맞게 수행하도록 구성해주는 작업이다. 먼저 binary의 경우 아래처럼 logit 함수 (`scipy.special.expit`)를 적용해서 focal objective를 계산했다.
```python
from scipy.special import expit

def _binary_focal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for sklearn API."""
    pred_prob = expit(y_pred)
    return _focal_grad_hess(y_true=y_true, pred_prob=pred_prob, gamma=gamma)
```

multiclass의 경우, `scipy.special.softmax` 함수와 y 값을 onehot encoding하여 objective를 계산했다. 이렇게 계산하게 되면 OVR 접근 방법과 optimization problem은 동치이면서 fitting은 한번만 해도 되지 않을까..? 물론, 수식적으로 증명해보진 않았다.
```python
from scipy.special import softmax

def _multiclass_focal_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for sklearn API."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true.astype(int)]
    return _focal_grad_hess(y_true=y_true_onehot, pred_prob=pred_prob, gamma=gamma)
```

제일 어려운 목적함수 부분을 처리했으니 이젠 사소한 부분만 남았다. `params` 를 setting 해주고, binary, multiclass 에 따라서 objective를 params에 넣어주면 된다. 그걸 수행하는 함수를 아래와 같이 구성했다.

```python
def set_params(params: dict[str, Any], train_set: Dataset) -> dict[str, Any]:
    """Set params and objective in params."""
    ...
    _params.update({"objective": fobj})
    return _params
```

남은건 `train`, `cv`를 `LightGBM`에서 따와서 그대로 구현하는 방법이다. 문제는 custom objective에 대해서 score 형태로만 결과를 predict 해준다는 건데, 그걸 방지하기 위해 `lgb.Dataset`을 상속받아 `ImbalancedBooster`을 만들어주고 `predict` method를 변경해준다.

```python
class ImbalancedBooster(lgb.Booster):
    def predict(
        self,
        data: lgb.basic._LGBM_PredictDataType,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        _predict = super().predict(
            data=data,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
            **kwargs,
        )
        if (
            raw_score
            or pred_leaf
            or pred_contrib
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if len(_predict.shape) == 1:
            return expit(_predict)
        else:
            return softmax(_predict, axis=1)
```

그런 다음, `train` 함수는 `lgb.train`을 통해 custom objective로 계산된 `lgb.Booster` 객체 대신 위에서 정의한 새로운 객체를 `model_to_string` method를 통해 `ImbalancedBooster` 객체로 바꿔준다.

```python
def train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    num_boost_round: int = 100,
    valid_sets: list[lgb.Dataset] = None,
    valid_names: list[str] = None,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    keep_training_booster: bool = False,
    callbacks: list[Callable] | None = None,
) -> ImbalancedBooster:
    _params = set_params(params=params, train_set=train_set)
    _booster = lgb.train(
        params=_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        init_model=init_model,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
    )
    _booster_str = _booster.model_to_string()
    return ImbalancedBooster(model_str=_booster_str)
```
이렇게 하면 `train`을 통해 학습된 모델이 predict를 할 때, score로 출력되는 값을 비교적 보기 편하게, 확률의 형태로 출력해 줄 수 있다. 

`cv`는 별다른 처리없이 똑같이 구성하면 잘 동작함을 확인했다. 마찬가지로, 최소 공수로 `ImbalancedLGBMClassifier`를 만들기 위해 `LGBMClassifier`의 `__is_multiclass` private method 뿐만 아니라 다양한 꼼수(?)를 적용해 아마도 400줄 이내의 코드로 다양한 기능을 수행하는 모델을 만들 수 있었다. ~~끗~~
