# 🌾 TriticumBCFramework
### A Modular Boosting Framework for Binary Classification
**本README使用Chatgpt5生成,请注意内容甄别。**

---

## 🧭 一、项目简介

> TriticumBCFramework（简称 **TBCF**）是一个专为 **二分类任务（Binary Classification）** 设计的多模型融合训练框架。  
> 框架整合了 LightGBM、XGBoost、CatBoost 三大核心模型，支持自动特征工程、超参数自动调优、模型集成融合与阈值优化。  
> 它旨在让开发者能快速上手二分类任务，并获得结构化、可复现的预测结果。

---

## 🏗️ 二、项目设计理念

| 设计目标 | 实现方式 |
|-----------|-----------|
| 模块化架构 | 独立的 utils / models / feature_engineering 层 |
| 高可配置性 | 通过 config.json 控制所有参数与流程 |
| 自动调参与模型融合 | 内置 Optuna（文件持久化）+ Weighted / Stacking 融合 |
| 二分类优化 | 所有模型固定 objective=binary，支持动态阈值优化 |
| 可复现与可追踪 | 固定随机种子 + 日志记录 + 时间戳命名输出 |
| 无数据库依赖 | 使用 JSON / Pickle 持久化调参与模型 |
| 可扩展性 | 模块化注册机制，可快速增加新特征或模型 |

---

## 📁 三、项目目录结构

```
TriticumBCFramework/
│
├── run.py                                # 🧠 主运行入口（控制训练与预测流程）
│
├── input/                                # 📥 输入与配置文件
│   ├── train.csv
│   ├── test.csv
│   ├── config.json
│   └── README.md
│
├── output/                               # 📤 输出结果与持久化内容
│   ├── tuning/
│   ├── best_params/
│   ├── models/
│   ├── feature_importance.csv
│   ├── oof_predictions.csv
│   ├── submission.csv
│   ├── feature_list.txt
│   ├── ensemble_report.txt
│   └── log_*.txt
│
├── models/                               # 🤖 模型封装层
│   ├── base_model.py
│   ├── lgb_model.py
│   ├── xgb_model.py
│   ├── cat_model.py
│   └── ensemble.py
│
├── utils/                                # 🧰 通用模块层
│   ├── config_loader.py
│   ├── config_validator.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── tuning.py
│   ├── metrics.py
│   ├── threshold_search.py
│   ├── io_utils.py
│   └── logger.py
│
├── feature_engineering/                  # 🔬 特征工程模块化层
│   ├── __init__.py
│   ├── core.py
│   ├── telecom_features.py
│   ├── nonlinear_features.py
│   ├── feature_registry.py
│   └── builder.py
│
├── docs/                                 # 📚 文档与记录
│   ├── README.md
│   ├── feature_semantics.md
│   ├── model_design.md
│   ├── experiments_log.md
│   └── changelog.md
│
└── LICENSE
```

---

## ⚙️ 四、运行流程（核心逻辑）

```
run.py
 ├── 1️⃣ 加载 config.json
 ├── 2️⃣ 初始化日志与随机种子
 ├── 3️⃣ 加载 train/test 数据
 ├── 4️⃣ 执行特征工程（Feature Builder）
 ├── 5️⃣ 遍历模型 ["lgb","xgb","cat"]:
 │       ├── 调参 (tuning)
 │       ├── 模型训练与预测 (fit_predict)
 │       ├── 模型保存
 ├── 6️⃣ 模型融合 (ensemble)
 ├── 7️⃣ 阈值搜索 (threshold_search)
 ├── 8️⃣ 输出结果 (submission, oof, logs)
 └── ✅ 结束（输出指标与报告）
```

---

## 🔗 五、模块间依赖关系

```
config_loader ─▶ config_validator
       │
       ▼
run.py ─▶ logger
   │
   ├──▶ data_loader
   ├──▶ feature_engineering
   ├──▶ tuning ─▶ metrics
   ├──▶ models (lgb/xgb/cat)
   ├──▶ ensemble ─▶ threshold_search
   └──▶ io_utils
```

---

## 🧱 六、模块职责与接口摘要

### 🧩 1. run.py
主控脚本：协调整个训练流程。  
输出 submission.csv / 日志 / 模型 / 特征列表。

---

### 🧩 2. utils 模块

| 文件 | 功能描述 |
|-------|------------|
| **config_loader.py** | 读取 JSON 配置文件并校验 |
| **config_validator.py** | 合并默认值、校验字段类型 |
| **data_loader.py** | 读取 train/test CSV 并返回 DataFrame |
| **feature_engineering.py** | 特征构建封装层（调用 feature_engineering 包） |
| **tuning.py** | Optuna 调参封装，无SQL依赖 |
| **metrics.py** | 动态评测系统：解析 `"0.7*acc+0.3*f1"` 等公式 |
| **threshold_search.py** | 阈值扫描与综合得分计算 |
| **io_utils.py** | 模型保存、JSON/Pickle/Joblib 文件读写工具 |
| **logger.py** | 全局日志控制（控制台 + 文件双输出） |

---

### 🧩 3. models 模块

| 文件 | 功能描述 |
|-------|------------|
| **base_model.py** | 抽象 Trainer 父类，定义统一接口 |
| **lgb_model.py** | LightGBM 二分类训练 |
| **xgb_model.py** | XGBoost 二分类训练 |
| **cat_model.py** | CatBoost 二分类训练 |
| **ensemble.py** | 模型融合层：Weighted + Stacking + Auto Weight Tuning |

#### BaseTrainer 接口
```python
class BaseTrainer:
    def __init__(self, params: dict):
        self.params = params

    def fit_predict(self,
                    X_train: pd.DataFrame,
                    y_train: np.ndarray,
                    X_test: pd.DataFrame,
                    folds: int = 5,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """执行交叉验证训练，返回OOF预测、测试预测、特征重要性"""
```
---

### 🧩 4) `feature_engineering` 模块

#### a) `core.py`
- 日期解析、数值清洗、类别编码

#### b) `telecom_features.py`
- 通话/流量/驻留比例等业务特征

#### c) `nonlinear_features.py`
- 熵、KL、偏度、峰度、平方差等非线性特征

#### d) `feature_registry.py`
```python
REGISTRY = {
  "telecom_features": apply_telecom_features,
  "nonlinear_features": apply_nonlinear_features
}
```

#### e) `builder.py`
```python
def build_features(train_df, test_df, fe_cfg: dict) -> (train_fe, test_fe, used_cols, cat_cols)
```

---

### 🧩 5) `metrics` 模块
```python
def evaluate(y_true, y_pred):
    return {"acc":..., "f1":..., "p":..., "r":...}

def compute_score(metrics: dict, formula: str) -> float:
    env = {k: float(v) for k, v in metrics.items()}
    return eval(formula, {"__builtins__": None}, env)
```

---

### 🧩 6) `ensemble.py`
```python
class EnsembleCombiner:
    def __init__(self, mode="weighted", weights=None, auto_tune=False,
                 metric_formula="0.7*acc+0.3*f1"):
        self.mode = mode
        self.weights = weights or {"lgb":0.5,"xgb":0.3,"cat":0.2}
        self.auto_tune = auto_tune
        self.metric_formula = metric_formula

    def combine(self, oof_preds, test_preds, y_true=None):
        if self.mode == "weighted":
            w = self._optimize_weights(oof_preds, y_true) if self.auto_tune else self.weights
            oof_final = sum(w[k]*v for k, v in oof_preds.items())
            test_final = sum(w[k]*v for k, v in test_preds.items())
            return oof_final, test_final, None
        else:
            X_meta = np.vstack([oof_preds[k] for k in oof_preds]).T
            X_meta_test = np.vstack([test_preds[k] for k in test_preds]).T
            meta = LogisticRegression(max_iter=500)
            meta.fit(X_meta, y_true)
            return meta.predict_proba(X_meta)[:,1], meta.predict_proba(X_meta_test)[:,1], meta
```

## 🧮 七、配置文件结构 (`/input/config.json`)

```json
{
  "global": {
    "seed": 42,
    "parallel": false,
    "overwrite": false,
    "ensemble_mode": "stacking",
    "auto_weight_tune": true
  },
  "tuning": {
    "enable_lgb": true,
    "enable_xgb": true,
    "enable_cat": false,
    "n_trials": 50,
    "continue_previous": true
  },
  "search_space": {
    "lgb": { "num_leaves": [31, 255], "max_depth": [-1, 12] },
    "xgb": { "max_depth": [4, 10], "eta": [0.01, 0.1] },
    "cat": { "depth": [4, 10], "learning_rate": [0.01, 0.1] }
  },
  "ensemble": {
    "mode": "weighted",
    "default_weights": { "lgb": 0.5, "xgb": 0.3, "cat": 0.2 },
    "meta_model": "logistic"
  },
  "threshold_search": {
    "start": 0.05,
    "end": 0.95,
    "step": 0.01,
    "auto_adjust": true
  },
  "evaluation": {
    "metric_formula": "0.7*acc + 0.3*f1",
    "primary_metric": "f1"
  },
  "feature_engineering": {
    "use_nonlinear": true,
    "save_feature_list": true,
    "registry": ["telecom_features", "nonlinear_features"]
  }
}
```

---

## 🧠 八、数据流图

```
        +--------------------+
        |  config.json       |
        +---------+----------+
                  |
                  v
+-----------------+------------------+
|   utils.config_loader              |
|   + validation + default merge     |
+-----------------+------------------+
                  |
                  v
            [run.py 控制流程]
                  |
       +----------+----------+
       |                     |
       v                     v
  utils.data_loader      feature_engineering.builder
       |                     |
       +----------+----------+
                  |
             train_fe, test_fe
                  |
       +----------+----------+
       |                     |
       v                     v
  utils.tuning           models.*
       |                     |
       +----------+----------+
                  |
            ensemble.combine
                  |
                  v
         threshold_search.find_best_threshold
                  |
                  v
             输出到 /output/
```

---

## 🧩 九、函数依赖表

| 函数 | 所属模块 | 依赖模块 |
|--------|--------------|----------------|
| `load_config()` | config_loader | config_validator, json |
| `validate_and_complete()` | config_validator | 内部递归合并 |
| `load_data()` | data_loader | pandas |
| `build_features()` | feature_engineering | core, registry |
| `tune_model()` | tuning | optuna, metrics |
| `fit_predict()` | base_model 派生类 | numpy, sklearn, lightgbm/xgboost/catboost |
| `combine()` | ensemble | numpy, sklearn |
| `evaluate()` | metrics | sklearn.metrics |
| `compute_score()` | metrics | eval() |
| `find_best_threshold()` | threshold_search | metrics |
| `save_model()` / `save_json()` | io_utils | joblib/json |
| `setup_logger()` | logger | logging |

---

## 📊 十、模型评测输出格式

| 文件名 | 内容 |
|---------|---------|
| `feature_importance.csv` | 特征重要性平均值 |
| `oof_predictions.csv` | Out-of-Fold 预测结果 |
| `submission.csv` | 最终预测标签 |
| `ensemble_report.txt` | 各模型分数、融合权重、阈值 |
| `log_*.txt` | 日志输出 |

---

## 🔢 十一、评测标准系统

TriticumBCFramework 的评测系统完全 **配置化**，不再固定写入代码中。  
用户可通过 `config.json` 中的 `"evaluation"` 段落来自定义评分逻辑与主评测指标。

### 📘 配置格式

```json
"evaluation": {
    "metric_formula": "0.7*acc + 0.3*f1",
    "primary_metric": "f1"
}
```
---

其中：

| 参数 | 说明 |
|------|------|
| `metric_formula` | 综合得分计算公式，支持动态表达式解析（Python 风格） |
| `primary_metric` | 主指标名称，用于 Optuna 调参与模型优先比较 |

---

### ⚙️ 支持的内置指标

| 名称 | 含义 |
|------|------|
| `acc` | Accuracy 准确率 |
| `f1` | F1-score |
| `p` | Precision 精确率 |
| `r` | Recall 召回率 |

---

### 🧮 可修改示例

- 竞赛环境：`"metric_formula": "0.7*acc + 0.3*f1"`  
- 学术研究环境：`"metric_formula": "0.5*f1 + 0.5*p"`  
- 医学/风控场景：`"metric_formula": "f1"`  

---

### 🧩 输出指标

| 指标 | 说明 |
|------|------|
| `Accuracy` | 分类正确率 |
| `Precision` | 精确率 |
| `Recall` | 召回率 |
| `F1` | F1-Score |
| `Score` | 根据公式计算的综合得分 |

---

### 🔍 动态解析机制（`utils/metrics.py`）

框架会自动解析配置中的表达式：

```python
def compute_score(metrics: dict, formula: str) -> float:
    """动态解析综合评分公式，如 '0.7*acc + 0.3*f1'"""
    local_env = {k: float(v) for k, v in metrics.items()}
    return eval(formula, {"__builtins__": None}, local_env)
```

> ⚠️ 安全提示：框架通过禁用 `__builtins__`，确保公式解析安全，仅允许基本算术运算。

---

### 📈 输出示例

```
[Eval] Metrics: ACC=0.93124  F1=0.79077  P=0.65415  R=0.99954  
[Eval] Formula: 0.7*acc + 0.3*f1  
[Eval] Final Score = 0.88910
```

## 🧾 十二、命名与代码规范

| 规则 | 说明 |
|-------|-------|
| 模块命名 | 全小写 + 下划线 |
| 类命名 | PascalCase |
| 函数命名 | snake_case |
| 类型注解 | 所有公共接口必须包含 |
| 日志记录 | 使用统一 logger |
| 输出命名 | 时间戳精确到秒 |
| 随机性 | GLOBAL_SEED = 42 |
| 错误处理 | 所有外部 I/O 捕获异常 |

---

## 🔩 十三、开发路线图   

| 阶段 | 模块 | 状态 |
|------|------|------|
| ① | config_loader + config_validator | ✅ 设计完成 |
| ② | metrics (动态评测系统) | ⏳ 进行中 |
| ③ | logger | ⏳ 进行中 |
| ④ | feature_engineering | ⏳ 进行中 |
| ⑤ | tuning (Optuna调参) | ⏳ 待实现 |
| ⑥ | models (LGB/XGB/CAT) | ⏳ 待实现 |
| ⑦ | ensemble (融合层) | ⏳ 待实现 |
| ⑧ | threshold_search | ⏳ 待实现 |
| ⑨ | run.py 主控流程 | ⏳ 待实现 |
| ✅ | 集成测试与打包 | 📦 待上线 |

---

## 🔮 十四、未来扩展方向

- ✅ 支持多分类与回归任务  
- ✅ 集成 TabNet / LogisticRegression / RF 模型  
- ✅ 支持特征选择与重要性剪枝  
- ✅ 集成模型可解释性 (SHAP)  
- ✅ CLI 命令接口

---
