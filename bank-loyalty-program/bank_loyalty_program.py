# Databricks notebook source
# MAGIC %md
# MAGIC # Uplift modeling

# COMMAND ----------

# MAGIC %md
# MAGIC U nastavku predstavljen je primer modela uzdizanja (uplift modeling) koriscenjem meta-modela uz pomoc biblioteka *causalml* i *scikit-uplift*.

# COMMAND ----------

#instaliranje potrebnih paketa

# COMMAND ----------

pip install causalml

# COMMAND ----------

pip install scikit-uplift

# COMMAND ----------

from causalml.inference.meta.base import BaseLearner
from causalml.inference.meta import(
    BaseSClassifier,
    BaseTClassifier,
    BaseXClassifier,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklift.metrics import uplift_by_percentile, uplift_curve
from sklift.viz import (
    plot_qini_curve,
    plot_uplift_by_percentile,
    plot_uplift_curve,
)
plt.style.use("bmh")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = "svg"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ucitavanje podataka

# COMMAND ----------

clients_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/clients.csv",parse_dates=["first_issue_date", "first_redeem_date"])
products_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/products.csv")
uplift_sample_submission_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/uplift_sample_submission.csv")
uplift_test_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/uplift_test.csv")
uplift_train_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/uplift_train.csv")

# COMMAND ----------

pur1 = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/purchases_1.csv")
pur2 = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/purchases_2.csv")
pur3 = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/purchases_3.csv")
pur4 = pd.read_csv("/dbfs/FileStore/shared_uploads/aleksandrab.popovic@raiffeisenbank.rs/purchases_4.csv")

all_dfs = [pur1, pur2, pur3, pur4]

purchases_df = pd.concat(all_dfs).reset_index(drop=True)

# COMMAND ----------

purchases_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Podaci o klijentima:

# COMMAND ----------

clients_df.info()

# COMMAND ----------

uplift_train_df.info()

# COMMAND ----------

purchases_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA
# MAGIC Let's start with exploring the data first.

# COMMAND ----------

fig, ax = plt.subplots()

uplift_train_df \
  .groupby(["treatment_flg", "target"], as_index=False) \
  .size() \
  .assign(
    share=lambda x: x["size"] / x["size"].sum()
  ) \
  .pipe((sns.barplot, "data"), x="target", y="share", hue="treatment_flg", ax=ax)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y :.0%}"))
ax.set(title="Targer by treatment distribution");

# COMMAND ----------

# MAGIC %md
# MAGIC We have approximately 50/50 split of the treatment_flg but the target is not balanced (approx. 40/60).

# COMMAND ----------

# Now we examine the clients data. First we count the unique number of client_id.
assert clients_df.shape[0] == clients_df["client_id"].nunique()
assert uplift_train_df.shape[0] == uplift_train_df["client_id"].nunique()

print(f"""
clients_id
----------
clients_df: {clients_df["client_id"].nunique()}
uplift_train_df: {uplift_train_df["client_id"].nunique()}
""")

# COMMAND ----------

# We have more client_id in the clients_df. Next we merge the data by client_id.

raw_data_df = pd.merge(
    left=clients_df, right=uplift_train_df, on="client_id", how="outer"
)

assert raw_data_df.shape[0] == clients_df.shape[0]
assert raw_data_df.shape[0] == raw_data_df["client_id"].nunique()

# COMMAND ----------

# We continue by taking a look into the gender feature.

#Warning: Including gender-like variables in ML models can induce undesirable biases. We do keep this feature just because we want to compare the techniques with the original example.

g = (
    raw_data_df.query("target.notnull()")
    .groupby(["treatment_flg", "target", "gender"], as_index=False)
    .agg(count=("client_id", "count"))
    .pipe(lambda d:
        sns.catplot(data=d,
        x="target",
        y="count",
        hue="gender",
        col="treatment_flg",
        kind="bar")
    )
)

# COMMAND ----------

# Now we plot the age distribution. Note however, we need to remove some outliers:
# reasonable age range
good_age_mask = "10 < age < 100"

print( f"""
Rows with age outliers: 
{1 - clients_df.query(good_age_mask).shape[0] / clients_df.shape[0]: 0.2%}
""")

# COMMAND ----------

raw_data_df.query("(target.notnull()) and (10 < age < 100)").pipe(
    (sns.displot, "data"), x="age", hue="gender", col="treatment_flg", kind="kde"
);

# COMMAND ----------

# We continue by studying the time variables. Note that the variable first_redeem_date has missing values. Let us see target and treatment distribution over these missing values.

g = (
    raw_data_df.assign(
        first_redeem_date_is_null=lambda x: x["first_redeem_date"].isna()
    )
    .groupby(
      ["treatment_flg", "target", "first_redeem_date_is_null"], as_index=False
    )
    .agg(count=("client_id", "count"))
    .pipe(
        (sns.catplot, "data"),
        x="target",
        y="count",
        hue="first_redeem_date_is_null",
        col="treatment_flg",
        kind="bar",
    )
)

# COMMAND ----------

# We do not see any pattern at first glance. Let us see the development the client counts over first_issue_date.

fig, ax = plt.subplots()

raw_data_df \
  .assign(first_issue_date=lambda x: x["first_issue_date"].dt.date) \
  .groupby(
    ["first_issue_date"], as_index=False
  ) \
  .agg(count=("client_id", "count")) \
  .pipe(
    (sns.lineplot, "data"),
    x="first_issue_date",
    y="count",
    label="first_issue_date",
    ax=ax,
)

raw_data_df \
  .query("first_redeem_date.isnull()") \
  .assign(
    first_issue_date=lambda x: x["first_issue_date"].dt.date
  ) \
  .groupby(["first_issue_date"], as_index=False) \
  .agg(count=("client_id", "count")) \
  .pipe(
    (sns.lineplot, "data"),
    x="first_issue_date",
    y="count",
    label="first_issue_date (first_redeem_date null)",
    ax=ax,
);

# COMMAND ----------

# There seems to be missing values along the whole time period.

print(f"""
rows share with missing values:
{raw_data_df.query("first_redeem_date.isnull()").shape[0] / raw_data_df.shape[0]: 0.2%}
""")

# COMMAND ----------

#From this initial EDA there is no hint of the source of these missing values, i.e. they are at random (or maybe we are missing some information or context of the data?).

#We now plot the client counts over first_issue_date and first_redeem_date:

fig, ax = plt.subplots()

raw_data_df \
  .assign(first_issue_date=lambda x: x["first_issue_date"].dt.date) \
  .groupby(["first_issue_date"], as_index=False) \
  .agg(count=("client_id", "count")) \
  .pipe((sns.lineplot, "data"),
    x="first_issue_date",
    y="count",
    label="first_issue_date",
    ax=ax,
)

raw_data_df \
  .assign(first_redeem_date=lambda x: x["first_redeem_date"].dt.date) \
  .groupby(["first_redeem_date"], as_index=False) \
  .agg(count=("client_id", "count")) \
  .pipe((sns.lineplot, "data"),
    x="first_redeem_date",
    y="count",
    label="first_redeem_date",
    ax=ax,
)
ax.set(xlabel="date", ylabel="count")

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, in order to enrich our models, we calculate some simple summary metrics from the purchase data:

# COMMAND ----------

client_purchases_summary_df = (
    purchases_df.groupby(["client_id"], as_index=False)
    .agg(
        n_transactions=("transaction_id", "count"),
        n_products=("product_id", "nunique"),
        n_stores=("store_id", "nunique"),
        last_transaction_date=("transaction_datetime", "max"),
        express_points_received=("express_points_received", np.sum),
        express_points_spent=("express_points_spent", np.sum),
        regular_points_spent=("regular_points_spent", np.sum),
        mean_product_quantity=("product_quantity", np.mean),
    )
    .assign(
      last_transaction_date=lambda x: pd.to_datetime(x["last_transaction_date"])
    )
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Warning: We are using time-dependent features like last_transaction_date = ("transaction_datetime", "max"), which have to be treated carefully when doing out-of-sample validation. Below we will do a train-test split by randomly selecting a fraction of the data bases on the client_id. Nevertheless, to have a faithful out-of-sample evaluation metrics we might want to compute these features on each split otherwise we would be leaking information. For the sake of this toy-example we will not do this.

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare data for Uplift modeling

# COMMAND ----------

# MAGIC %md
# MAGIC In this section we prepare the data for uplift modeling. We will use the features from the original example (with minor modifications) plus some purchase features.

# COMMAND ----------

# add purchase features
raw_data_ext_df = raw_data_df \
  .copy() \
  .merge(
    right=client_purchases_summary_df,
    on="client_id",
    how="left"
)

# COMMAND ----------

transformation_map = {
    "first_issue_time": lambda x: (x["first_issue_date"] - pd.Timestamp("2017-01-01")).dt.days,
    "first_issue_time_weekday": lambda x: x["first_issue_date"].dt.weekday,
    "first_issue_time_month": lambda x: x["first_issue_date"].dt.month,
    "first_redeem_time": lambda x: (x["first_redeem_date"] - pd.Timestamp("2017-01-01")).dt.days,
    "issue_redeem_delay": lambda x: (x["first_redeem_time"] - x["first_issue_time"]),
    "last_transaction_time": lambda x: (x["last_transaction_date"] - pd.Timestamp("2017-01-01")).dt.days,
}

data_df = (
    raw_data_ext_df.copy()
    .query("target.notnull()")
    .query(good_age_mask)
    .set_index("client_id")
    .assign(**transformation_map)
    .sort_values("first_issue_time")
    .drop(
        columns=[
            "first_issue_date",
            "first_redeem_date",
            "last_transaction_date",
        ]
    )
)

data_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC We can now show a pair-plot of the main features of the original example:

# COMMAND ----------

sns.pairplot(
    data=data_df[
        [
            "first_issue_time",
            "first_redeem_time",
            "issue_redeem_delay",
        ]
    ],
    kind="hist",
    height=2.5,
    aspect=1.5,
)

# COMMAND ----------

# Now we do a simple train-validation split of the data.

target_col = "target"
treatment_col = "treatment_flg"

y = data_df[target_col]
w = data_df[treatment_col]
x = data_df.drop(columns=[treatment_col, target_col])

idx_train, idx_val = train_test_split(
    data_df.index,
    test_size=0.3,
    random_state=42,
    stratify=(y.astype(str) + "_" + w.astype(str)),
)

x_train = x.loc[idx_train]
x_val = x.loc[idx_val]

w_train = w.loc[idx_train]
w_val = w.loc[idx_val]

y_train = y.loc[idx_train]
y_val = y.loc[idx_val]

# COMMAND ----------

# Let us encode the gender as an ordinal categorical variable.

ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(x_train[["gender"]])

x_train_transformed = x_train.assign(
    gender=lambda x: ordinal_encoder.transform(x[["gender"]])
)

x_val_transformed = x_val.assign(
    gender=lambda x: ordinal_encoder.transform(x[["gender"]])
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Propensity Score Model

# COMMAND ----------

# MAGIC %md
# MAGIC The propensity score are defined as 
# MAGIC p
# MAGIC (
# MAGIC X
# MAGIC i
# MAGIC )
# MAGIC =
# MAGIC P
# MAGIC (
# MAGIC W
# MAGIC i
# MAGIC =
# MAGIC 1
# MAGIC |
# MAGIC X
# MAGIC i
# MAGIC )
# MAGIC , that is, the probability of having a treatment given the covariates. If the treatment assignment is at random these scores should be concentrated around 0.5. For a nice introduction to the subject you can see “Propensity Score Matching: A Non-experimental Approach to Causal Inference” by Michael Johns, PyData New York 2019. We ser scikit-learn’s HistGradientBoostingClassifier. For this model we need to explicitly indicate the categorical variables:

# COMMAND ----------

categorical_features = ["gender"]

hgc_params = {
    "categorical_features": np.argwhere(
        [col in categorical_features for col in x_train_transformed.columns]
    ).flatten()
}

# COMMAND ----------

propensity_model = HistGradientBoostingClassifier(**hgc_params)

propensity_model.fit(X=x_train_transformed, y=w_train)
p_train = propensity_model.predict_proba(X=x_train_transformed)
p_val = propensity_model.predict_proba(X=x_val_transformed)

p_train = pd.Series(p_train[:, 0], index=idx_train)
p_val = pd.Series(p_val[:, 0], index=idx_val)

# COMMAND ----------

fig, ax = plt.subplots()
sns.kdeplot(x=p_train, label="train", ax=ax)
sns.kdeplot(x=p_val, label="val", ax=ax)
ax.legend()
ax.set(
    title="Propensity Score Predictions Distribution",
    xlabel="propensity score",
    ylabel="density",
);

# COMMAND ----------

print(f"""
Share of predictions with |p - 0.5| > 0.2 (train) {p_train[abs(p_train - 0.5) > 0.2].size / p_train.size : 0.2%}
Share of predictions with |p - 0.5| > 0.2 (val) {p_val[abs(p_val - 0.5) > 0.2].size / p_val.size : 0.2%}
""")

# COMMAND ----------

from sklearn.inspection import permutation_importance

pi = permutation_importance(
    estimator=propensity_model, X=x_train_transformed, y=w_train
)

fig, ax = plt.subplots(figsize=(8, 8))

idx = pi["importances_mean"].argsort()[::-1]

sns.barplot(
    x=pi["importances_mean"][idx],
    y=x_train_transformed.columns[idx],
    color="C4",
    ax=ax
)
ax.set(title="Permutation importance propensity score model");

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Container
# MAGIC We now define a convenient data structure for the uplift models input data.

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class DataIn:
    x: pd.DataFrame
    x_transformed: np.array
    y: pd.Series
    treatment: pd.Series
    p: pd.Series


data_train = DataIn(
    x=x_train,
    x_transformed=x_train_transformed,
    y=y_train,
    treatment=w_train,
    p=p_train,
)

data_val = DataIn(
    x=x_val,
    x_transformed=x_val_transformed,
    y=y_val,
    treatment=w_val,
    p=p_val
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Models
# MAGIC Now that we have a better understanding of the data we can start modeling. We use some of the meta-learners from causalml. For more details please see the causalml documentation.

# COMMAND ----------

# MAGIC %md
# MAGIC S learner

# COMMAND ----------

s_learner = BaseSClassifier(learner=HistGradientBoostingClassifier(**hgc_params))

s_ate = s_learner.estimate_ate(
    X=data_train.x_transformed, treatment=data_train.treatment, y=data_train.y
)

# COMMAND ----------

s_learner.models[1]

# COMMAND ----------

# MAGIC %md
# MAGIC T learner

# COMMAND ----------

t_learner = BaseTClassifier(learner=HistGradientBoostingClassifier(**hgc_params))

t_ate_lwr, t_ate, t_ate_upr = t_learner.estimate_ate(
    X=data_train.x_transformed, treatment=data_train.treatment, y=data_train.y
)

# COMMAND ----------

t_learner.models_c[1]  # control group
t_learner.models_t[1]  # treatment group

# COMMAND ----------

# MAGIC %md
# MAGIC X learner

# COMMAND ----------

x_learner = BaseXClassifier(
    outcome_learner=HistGradientBoostingClassifier(**hgc_params),
    effect_learner=HistGradientBoostingRegressor(**hgc_params),
)

x_ate_lwr, x_ate, x_ate_upr = x_learner.estimate_ate(
    X=data_train.x_transformed,
    treatment=data_train.treatment,
    y=data_train.y,
    p=data_train.p,
)

# COMMAND ----------

# step 1
x_learner.models_mu_c[1]  # control group
x_learner.models_mu_t[1]  # treatment group
# step 3
x_learner.models_tau_c[1]  # control group
x_learner.models_tau_t[1]  # treatment group

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))

pd.DataFrame(
    data={
        "model": ["s_learner", "t_learner", "x_learner"],
        "ate": np.array([s_ate, t_ate, x_ate]).flatten(),
    },
).pipe((sns.barplot, "data"), x="model", y="ate", ax=ax)
ax.set(title="ATE Estimation (Train)");

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions & Diagnostics
# MAGIC Next, now that we have fitted meta-learner models, we generate predictions in the training and validations sets.

# COMMAND ----------

@dataclass
class DataOut:
    meta_learner_name: str
    meta_learner: BaseLearner
    y_pred: np.array


# in-sample predictions
data_out_train_s = DataOut(
    meta_learner_name="S-Learner",
    meta_learner=s_learner,
    y_pred=s_learner.predict(
        X=data_train.x_transformed, treatment=data_train.treatment
    ),
)
data_out_train_t = DataOut(
    meta_learner_name="T-Learner",
    meta_learner=t_learner,
    y_pred=t_learner.predict(
        X=data_train.x_transformed, treatment=data_train.treatment
    ),
)
data_out_train_x = DataOut(
    meta_learner_name="X-Learner",
    meta_learner=x_learner,
    y_pred=x_learner.predict(
        X=data_train.x_transformed, treatment=data_train.treatment, p=data_train.p
    ),
)
# out-of-sample predictions
data_out_val_s = DataOut(
    meta_learner_name="S-Learner",
    meta_learner=s_learner,
    y_pred=s_learner.predict(
      X=data_val.x_transformed, treatment=data_val.treatment
    ),
)
data_out_val_t = DataOut(
    meta_learner_name="T-Learner",
    meta_learner=t_learner,
    y_pred=t_learner.predict(
      X=data_val.x_transformed, treatment=data_val.treatment
    ),
)
data_out_val_x = DataOut(
    meta_learner_name="X-Learner",
    meta_learner=x_learner,
    y_pred=x_learner.predict(
        X=data_val.x_transformed, treatment=data_val.treatment, p=data_val.p
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC Perfect uplift prediction

# COMMAND ----------

def perfect_uplift_model(data: DataIn):
    # control Responders
    cr_num = np.sum((data.y == 1) & (data.treatment == 0))
    # treated Non-Responders
    tn_num = np.sum((data.y == 0) & (data.treatment == 1))

    # compute perfect uplift curve
    summand = data.y if cr_num > tn_num else data.treatment
    return 2 * (data.y == data.treatment) + summand


perfect_uplift_train = perfect_uplift_model(data=data_train)
perfect_uplift_val = perfect_uplift_model(data=data_val)

# COMMAND ----------

# MAGIC %md
# MAGIC We can compare the sorted predictions of the models against the perfect one.

# COMMAND ----------

fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 8), sharex=True, layout="constrained"
)
sns.lineplot(
    x=range(data_train.y.size),
    y=np.sort(a=perfect_uplift_train)[::-1],
    color="C3",
    label="perfect model",
    ax=ax[0],
)
sns.lineplot(
    x=range(data_train.y.size),
    y=np.sort(a=data_out_train_s.y_pred.flatten())[::-1],
    color="C0",
    label="S Learner",
    ax=ax[1],
)
sns.lineplot(
    x=range(data_train.y.size),
    y=np.sort(a=data_out_train_t.y_pred.flatten())[::-1],
    color="C1",
    label="T Learner",
    ax=ax[1],
)
sns.lineplot(
    x=range(data_train.y.size),
    y=np.sort(a=data_out_train_x.y_pred.flatten())[::-1],
    color="C2",
    label="X Learner",
    ax=ax[1],
)
ax[1].set(xlabel="Number treated")
fig.suptitle("np.sort(a=uplift_prediction)[::-1] (train)");

# COMMAND ----------

# MAGIC %md
# MAGIC Uplift by percentile
# MAGIC 1. Sort uplift predictions by decreasing order.
# MAGIC 2. Predict uplift for both treated and control observations
# MAGIC 3. Compute the average prediction per percentile in both groups.
# MAGIC 4. The difference between those averages is taken for each percentile.
# MAGIC This difference gives an idea of the uplift gain per percentile. One can compute this using the uplift_by_percentile function (from sklift.metrics). Let us see how the data looks for the S learner.

# COMMAND ----------

uplift_by_percentile_df = uplift_by_percentile(
    y_true=data_train.y,
    uplift=data_out_train_s.y_pred.flatten(),
    treatment=data_train.treatment,
    strategy="overall",
    total=True,
)

uplift_by_percentile_df

# COMMAND ----------

# MAGIC %md
# MAGIC A well performing model would have large values in the first percentiles and decreasing values for larger ones. Now we can generate the plots:

# COMMAND ----------

train_pred = [data_out_train_s, data_out_train_t, data_out_train_x]

for data_out_train in train_pred:
    ax = plot_uplift_by_percentile(
        y_true=data_train.y,
        uplift=data_out_train.y_pred.flatten(),
        treatment=data_train.treatment,
        strategy="overall",
        kind="line",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    fig.suptitle(
      f"In-sample predictions ({data_out_train.meta_learner_name})", y=1.1
    )

# COMMAND ----------

val_pred = [data_out_val_s, data_out_val_t, data_out_val_x]

for data_out_val in val_pred:
    ax = plot_uplift_by_percentile(
        y_true=data_val.y,
        uplift=data_out_val.y_pred.flatten(),
        treatment=data_val.treatment,
        strategy="overall",
        kind="line",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    fig.suptitle(
      f"Out-of-sample predictions ({data_out_val.meta_learner_name})", y=1.1
    )

# COMMAND ----------

# Here is the uplift by percentile table for the perfect model (train):

uplift_by_percentile_df = uplift_by_percentile(
    y_true=data_train.y,
    uplift=perfect_uplift_train,
    treatment=data_train.treatment,
    strategy="overall",
    total=False,
)

uplift_by_percentile_df

# COMMAND ----------

# MAGIC %md
# MAGIC Cumulative gain chart

# COMMAND ----------

uplift_by_percentile_df = uplift_by_percentile(
    y_true=data_train.y,
    uplift=data_out_train_s.y_pred.flatten(),
    treatment=data_train.treatment,
    strategy="overall",
    total=False,
)


def compute_response_absolutes(df: pd.DataFrame) -> pd.DataFrame:
    df["responses_treatment"] = df["n_treatment"] * df["response_rate_treatment"]
    df["responses_control"] = df["n_control"] * df["response_rate_control"]
    return df


def compute_cumulative_response_rates(df: pd.DataFrame) -> pd.DataFrame:
    df["n_treatment_cumsum"] = df["n_treatment"].cumsum()
    df["n_control_cumsum"] = df["n_control"].cumsum()
    df["responses_treatment_cumsum"] = df["responses_treatment"].cumsum()
    df["responses_control_cumsum"] = df["responses_control"].cumsum()
    df["response_rate_treatment_cumsum"] = (
        df["responses_treatment_cumsum"] / df["n_treatment_cumsum"]
    )
    df["response_rate_control_cumsum"] = (
        df["responses_control_cumsum"] / df["n_control_cumsum"]
    )
    return df


def compute_cumulative_gain(df: pd.DataFrame) -> pd.DataFrame:
    df["uplift_cumsum"] = (
        df["response_rate_treatment_cumsum"] - df["response_rate_control_cumsum"]
    )
    df["cum_gain"] = df["uplift_cumsum"] * (
        df["n_treatment_cumsum"] + df["n_control_cumsum"]
    )
    return df


fig, ax = plt.subplots()

uplift_by_percentile_df \
  .pipe(compute_response_absolutes) \
  .pipe(compute_cumulative_response_rates) \
  .pipe(compute_cumulative_gain) \
  .plot(y="cum_gain", kind="line", marker="o", ax=ax)
ax.legend().remove()
ax.set(
    title="Cumulative gain by percentile - S Learned (train)",
    ylabel="cumulative gain"
);

# COMMAND ----------

# MAGIC %md
# MAGIC Uplift curve

# COMMAND ----------

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), layout="constrained")
# in-sample
for i, data_out_train in enumerate(train_pred):
    ax = axes[i, 0]
    plot_uplift_curve(
        y_true=data_train.y,
        uplift=data_out_train.y_pred.flatten(),
        treatment=data_train.treatment,
        perfect=True,
        ax=ax,
    )
    ax.set(title=f"In-sample predictions ({data_out_train.meta_learner_name})")

# out-of-sample
for j, data_out_val in enumerate(val_pred):
    ax = axes[j, 1]
    plot_uplift_curve(
        y_true=data_val.y,
        uplift=data_out_val.y_pred.flatten(),
        treatment=data_val.treatment,
        perfect=True,
        ax=ax,
    )
    ax.set(title=f"Out-sample predictions ({data_out_val.meta_learner_name})")

fig.suptitle("Uplift Curves", fontsize=24);

# COMMAND ----------

# MAGIC %md
# MAGIC A remark on the perfect uplift curve: (Diemert, Eustache, et.al. (2020) “A Large Scale Benchmark for Uplift Modeling”) > A perfect model assigns higher scores to all treated individuals with positive outcomes than any individuals with negative outcomes.

# COMMAND ----------

from sklift.metrics import uplift_curve

num_all, curve_values = uplift_curve(
    y_true=data_train.y, uplift=perfect_uplift_train, treatment=data_train.treatment
)

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
sns.lineplot(
    x=num_all,
    y=curve_values,
    color="C2",
    marker="o",
    markersize=10,
    label="perfect uplift curve",
    ax=ax1,
)
sns.lineplot(
    x=range(data_train.y.size),
    y=np.sort(a=perfect_uplift_train)[::-1],
    color="C3",
    label="np.sort(a=perfect_uplift_train)[::-1]",
    ax=ax2,
)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
ax1.set(
    xlabel="Number targeted",
    ylabel="Number of incremental outcome",
    title="Perfect Uplift Curve",
)
ax2.grid(None)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1);

# COMMAND ----------

# MAGIC %md
# MAGIC We can compare the perfect uplift curve against a random one:

# COMMAND ----------

 number of random uplift curves to generate
n_random_samples = 100
# sample random uplift curves from a uniform distribution
uplift_random_samples = np.random.uniform(
    low=-1,
    high=1,
    size=(data_train.y.size, n_random_samples),
)
# compute uplift curve for each random sample
random_uplift_curves = [
    uplift_curve(
        y_true=data_train.y,
        uplift=uplift_random_samples[:, i],
        treatment=data_train.treatment,
    )
    for i in range(n_random_samples)
]
 plot
fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 10), sharex=True, layout="constrained"
)
# perfect uplift curve
sns.lineplot(
    x=num_all,
    y=curve_values,
    color="C2",
    marker="o",
    markersize=10,
    label="perfect uplift curve",
    ax=ax[1],
)
 random uplift curves
for x, y in random_uplift_curves:
    ax[0].plot(x, y, color="C1", alpha=0.05)
    ax[1].plot(x, y, color="C1", alpha=0.05)
ax[0].set(title="Random Uplift Curves", ylabel="Number of incremental outcome")
ax[1].set(xlabel="Number targeted", ylabel="Number of incremental outcome");

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qini Curve

# COMMAND ----------

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), layout="constrained")
# in-sample
for i, data_out_train in enumerate(train_pred):
    ax = axes[i, 0]
    plot_qini_curve(
        y_true=data_train.y,
        uplift=data_out_train.y_pred.flatten(),
        treatment=data_train.treatment,
        perfect=True,
        ax=ax,
    )
    ax.set(title=f"In-sample predictions ({data_out_train.meta_learner_name})")

# out-of-sample
for j, data_out_val in enumerate(val_pred):
    ax = axes[j, 1]
    plot_qini_curve(
        y_true=data_val.y,
        uplift=data_out_val.y_pred.flatten(),
        treatment=data_val.treatment,
        perfect=True,
        ax=ax,
    )
    ax.set(title=f"Out-sample predictions ({data_out_val.meta_learner_name})")

fig.suptitle("Qini Curves", fontsize=24);

# COMMAND ----------


