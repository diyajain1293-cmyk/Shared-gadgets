import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules


# =========================
# Data loading and helpers
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("Shared_Gadget_Library_Survey_Synthetic_Data.csv")
    return df


def add_target_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target and numeric likelihood score."""
    df = df.copy()
    positive = {
        "Definitely will use (5/5)",
        "Probably will use (4/5)",
    }
    score_map = {
        "Probably will not use (2/5)": 2,
        "Might or might not use (3/5)": 3,
        "Probably will use (4/5)": 4,
        "Definitely will use (5/5)": 5,
    }
    df["target_willing"] = df["Q39_Likelihood_to_Use"].apply(
        lambda x: 1 if x in positive else 0
    )
    df["Likelihood_Score"] = df["Q39_Likelihood_to_Use"].map(score_map)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y and column lists for modelling."""
    X = df.drop(
        columns=[
            "Q39_Likelihood_to_Use",
            "target_willing",
            "Response_ID",
            "Timestamp",
            "Likelihood_Score",
        ]
    )
    y = df["target_willing"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return X, y, categorical_cols, numeric_cols


# =========================
# Model training
# =========================

def train_and_evaluate_models(df: pd.DataFrame):
    """Train Decision Tree, Random Forest and Gradient Boosting models.

    Returns:
        metrics_df: train & test metrics for each model
        cv_df: cross-validation metrics
        confusion_figs: dict of model -> confusion matrix figure
        roc_combined_fig: combined ROC curve figure
        best_pipeline: best performing model pipeline
        feature_columns: list of feature columns used for training
    """
    df_mod = add_target_and_scores(df)
    X, y, categorical_cols, numeric_cols = get_feature_matrix(df_mod)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows = []
    cv_rows = []
    confusion_figs = {}
    roc_curves = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        # Cross-validation on training set
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
                "cv_precision_mean": np.mean(cv_results["test_precision"]),
                "cv_recall_mean": np.mean(cv_results["test_recall"]),
                "cv_f1_mean": np.mean(cv_results["test_f1"]),
                "cv_roc_auc_mean": np.mean(cv_results["test_roc_auc"]),
            }
        )

        # Fit on full training data
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        y_train_proba = pipe.predict_proba(X_train)[:, 1]
        y_test_proba = pipe.predict_proba(X_test)[:, 1]

        metrics_rows.append(
            {
                "model": name,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred),
                "train_recall": recall_score(y_train, y_train_pred),
                "train_f1": f1_score(y_train, y_train_pred),
                "train_roc_auc": roc_auc_score(y_train, y_train_proba),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred),
                "test_recall": recall_score(y_test, y_test_pred),
                "test_f1": f1_score(y_test, y_test_pred),
                "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            }
        )

        # Confusion matrix figure (black & white, readable text)
        cm = confusion_matrix(y_test, y_test_pred)
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, cmap="Greys")
        ax_cm.set_title(f"Confusion Matrix - {name}")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Not willing", "Willing"])
        ax_cm.set_yticklabels(["Not willing", "Willing"])

        max_val = cm.max()
        threshold = max_val / 2.0

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                text_color = "white" if value > threshold else "black"
                ax_cm.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        plt.tight_layout()
        confusion_figs[name] = fig_cm

        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    # Combined ROC figure
    fig_roc, ax_roc = plt.subplots()
    for name, roc_data in roc_curves.items():
        ax_roc.plot(roc_data["fpr"], roc_data["tpr"], label=f"{name} (AUC = {roc_data['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves - All Models")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    roc_combined_fig = fig_roc

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    cv_df = pd.DataFrame(cv_rows).set_index("model")

    # Choose best model by test ROC-AUC
    best_name = metrics_df["test_roc_auc"].idxmax()
    best_pipeline = pipelines[best_name]

    feature_columns = X.columns.tolist()

    return metrics_df, cv_df, confusion_figs, roc_combined_fig, best_pipeline, feature_columns
def train_and_evaluate_regression_models(df: pd.DataFrame):
    """
    Train regression models to predict Likelihood_Score (2–5).

    Returns:
        metrics_df: train & test metrics (R2, MAE, RMSE) for each model
        cv_df: 5-fold CV metrics
        best_name: name of best model by test R2
        scatter_fig: matplotlib figure of predicted vs actual (best model)
    """
    # Create numeric target
    df_mod = add_target_and_scores(df)

    y = df_mod["Likelihood_Score"]
    X = df_mod.drop(
        columns=[
            "Q39_Likelihood_to_Use",
            "target_willing",
            "Likelihood_Score",
            "Response_ID",
            "Timestamp",
        ],
        errors="ignore",
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows = []
    cv_rows = {}

    from sklearn.model_selection import cross_validate

    # For CV we need a dummy classification-like split, so we bin y a bit
    y_binned = pd.qcut(y, q=5, duplicates="drop", labels=False)

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        # 5-fold CV with regression scoring
        cv_results = cross_validate(
            pipe,
            X,
            y,
            cv=cv.split(X, y_binned),
            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
            return_train_score=False,
        )
        cv_rows[name] = {
            "cv_r2_mean": np.mean(cv_results["test_r2"]),
            "cv_mae_mean": -np.mean(cv_results["test_neg_mean_absolute_error"]),
            "cv_rmse_mean": -np.mean(cv_results["test_neg_root_mean_squared_error"]),
        }

        # Fit and evaluate on train/test split
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

        metrics_rows.append(
            {
                "model": name,
                "train_R2": train_r2,
                "test_R2": test_r2,
                "train_MAE": train_mae,
                "test_MAE": test_mae,
                "train_RMSE": train_rmse,
                "test_RMSE": test_rmse,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    cv_df = pd.DataFrame.from_dict(cv_rows, orient="index")

    # Pick best model by test R2
    best_name = metrics_df["test_R2"].idxmax()
    best_model = models[best_name]

    # Refit best model to get predictions for scatter plot
    best_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])
    best_pipe.fit(X_train, y_train)
    y_test_pred_best = best_pipe.predict(X_test)

    # Predicted vs actual scatter plot
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(y_test, y_test_pred_best, alpha=0.7)
    min_val = min(y_test.min(), y_test_pred_best.min())
    max_val = max(y_test.max(), y_test_pred_best.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], "k--")
    ax_scatter.set_xlabel("Actual Likelihood Score")
    ax_scatter.set_ylabel("Predicted Likelihood Score")
    ax_scatter.set_title(f"Predicted vs Actual – {best_name}")
    plt.tight_layout()

    return metrics_df, cv_df, best_name, fig_scatter

# =========================
# Filtering for insights
# =========================

def apply_filters(df: pd.DataFrame):
    df = add_target_and_scores(df)

    st.sidebar.header("Filters")

    # Equipment columns (borrow/interest)
    equipment_cols = [col for col in df.columns if col.startswith("Q15_") or col.startswith("Q19_")]
    equipment_nice = {col: col.replace("Q15_", "").replace("Q19_", "") for col in equipment_cols}

    selected_labels = st.sidebar.multiselect(
        "Equipment people are willing to borrow/lend",
        options=list(equipment_nice.values()),
        default=list(equipment_nice.values())[:5],
    )
    selected_cols = [col for col, nice in equipment_nice.items() if nice in selected_labels]

    min_like = st.sidebar.slider(
        "Minimum likelihood / satisfaction score",
        min_value=int(df["Likelihood_Score"].min()),
        max_value=int(df["Likelihood_Score"].max()),
        value=int(df["Likelihood_Score"].median()),
    )

    df_filtered = df[df["Likelihood_Score"] >= min_like].copy()

    if selected_cols:
        mask_equipment = (df_filtered[selected_cols] == 1).any(axis=1)
        df_filtered = df_filtered[mask_equipment]

    return df_filtered, equipment_cols


# =========================
# Charts for insights
# =========================

def show_insight_charts(df_filtered: pd.DataFrame, equipment_cols):
    # Ensure target and score exist
    if "target_willing" not in df_filtered.columns or "Likelihood_Score" not in df_filtered.columns:
        df_filtered = add_target_and_scores(df_filtered)

    st.subheader("1. Age group vs willingness to use BorrowBox")
    age_pref = (
        df_filtered.groupby("Q1_Age_Group")["target_willing"]
        .mean()
        .reset_index(name="share_willing")
        .sort_values("Q1_Age_Group")
    )
    chart_age = (
        alt.Chart(age_pref)
        .mark_bar()
        .encode(
            x=alt.X("Q1_Age_Group:N", title="Age group"),
            y=alt.Y("share_willing:Q", title="Share willing to use"),
            tooltip=["Q1_Age_Group", alt.Tooltip("share_willing", format=".2f")],
        )
    )
    st.altair_chart(chart_age, use_container_width=True)
    st.caption(
        "Younger age groups, especially 18–34, have a higher share of people willing to use BorrowBox. "
        "This suggests our primary target segment is young adults."
    )

    st.subheader("2. Subscription willingness by age group (monthly fee)")
    wtp_age = (
        df_filtered.groupby(["Q1_Age_Group", "Q30_WTP_Monthly"])
        .size()
        .reset_index(name="count")
    )
    chart_wtp_age = (
        alt.Chart(wtp_age)
        .mark_bar()
        .encode(
            x=alt.X("Q1_Age_Group:N", title="Age group"),
            y=alt.Y("count:Q", stack="normalize", title="Proportion"),
            color=alt.Color("Q30_WTP_Monthly:N", title="Monthly subscription willingness"),
            tooltip=["Q1_Age_Group", "Q30_WTP_Monthly", "count"],
        )
    )
    st.altair_chart(chart_wtp_age, use_container_width=True)
    st.caption(
        "Different age segments show different willingness to pay. Younger users cluster in lower tiers, "
        "while working professionals are more open to mid-level subscription prices."
    )

    st.subheader("3. Most popular equipment categories")
    eq_interest = (
        df_filtered[equipment_cols]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "equipment", 0: "interest_rate"})
    )
    eq_interest["equipment"] = (
        eq_interest["equipment"]
        .str.replace("Q15_", "", regex=False)
        .str.replace("Q19_", "", regex=False)
    )

    chart_eq = (
        alt.Chart(eq_interest.head(15))
        .mark_bar()
        .encode(
            x=alt.X("interest_rate:Q", title="Share of respondents interested"),
            y=alt.Y("equipment:N", sort="-x", title="Equipment"),
            tooltip=["equipment", alt.Tooltip("interest_rate", format=".2f")],
        )
    )
    st.altair_chart(chart_eq, use_container_width=True)
    st.caption(
        "Cleaning appliances, tools, sports gear and outdoor equipment appear at the top. "
        "These should be prioritised when we build the initial BorrowBox inventory."
    )

    st.subheader("4. Income vs likelihood to use (heatmap)")
    heat = (
        df_filtered.groupby(["Q4_Monthly_Income", "Likelihood_Score"])
        .size()
        .reset_index(name="count")
    )
    chart_heat = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Q4_Monthly_Income:N", title="Monthly income"),
            y=alt.Y("Likelihood_Score:O", title="Likelihood score"),
            color=alt.Color("count:Q", title="Number of respondents"),
            tooltip=["Q4_Monthly_Income", "Likelihood_Score", "count"],
        )
    )
    st.altair_chart(chart_heat, use_container_width=True)
    st.caption(
        "BorrowBox is attractive across several income groups, but low- to mid-income respondents show particularly strong interest, "
        "as they benefit most from avoiding large one-off purchases."
    )

    st.subheader("5. Need frequency vs likelihood (boxplot)")
    box_data = df_filtered[["Q11_Need_Frequency", "Likelihood_Score"]].dropna()
    chart_box = (
        alt.Chart(box_data)
        .mark_boxplot()
        .encode(
            x=alt.X("Q11_Need_Frequency:N", title="How often do you need rarely used gadgets?"),
            y=alt.Y("Likelihood_Score:Q", title="Likelihood / satisfaction score"),
            tooltip=["Q11_Need_Frequency"],
        )
    )
    st.altair_chart(chart_box, use_container_width=True)
    st.caption(
        "People who need such gadgets occasionally (for example once a month or a few times a year) "
        "are more willing to use BorrowBox than people who almost never need them."
    )

    st.subheader("6. Accommodation type vs willingness to use BorrowBox")
    if "Q8_Accommodation_Type" in df_filtered.columns:
        acc_pref = (
            df_filtered.groupby("Q8_Accommodation_Type")["target_willing"]
            .mean()
            .reset_index(name="share_willing")
            .sort_values("share_willing", ascending=False)
        )
        chart_acc = (
            alt.Chart(acc_pref)
            .mark_bar()
            .encode(
                x=alt.X("share_willing:Q", title="Share willing to use"),
                y=alt.Y("Q8_Accommodation_Type:N", sort="-x", title="Accommodation type"),
                tooltip=["Q8_Accommodation_Type", alt.Tooltip("share_willing", format=".2f")],
            )
        )
        st.altair_chart(chart_acc, use_container_width=True)
        st.caption(
            "Residents in apartments or smaller units tend to be more willing to use BorrowBox than those in larger homes or villas, "
            "because they have less storage space and more motivation to share."
        )

    st.subheader("7. Storage space at home vs willingness to use BorrowBox")
    if "Q10_Storage_Space" in df_filtered.columns:
        storage_pref = (
            df_filtered.groupby("Q10_Storage_Space")["target_willing"]
            .mean()
            .reset_index(name="share_willing")
            .sort_values("share_willing", ascending=False)
        )
        chart_storage = (
            alt.Chart(storage_pref)
            .mark_bar()
            .encode(
                x=alt.X("share_willing:Q", title="Share willing to use"),
                y=alt.Y("Q10_Storage_Space:N", sort="-x", title="Storage space at home"),
                tooltip=["Q10_Storage_Space", alt.Tooltip("share_willing", format=".2f")],
            )
        )
        st.altair_chart(chart_storage, use_container_width=True)
        st.caption(
            "Households with limited or medium storage space are more interested in BorrowBox, confirming that space-saving "
            "is a key value proposition for the service."
        )

    st.subheader("8. What matters most to high-likelihood users (service features)")
    service_cols = [c for c in df_filtered.columns if c.startswith("Q27_")]
    high_like = df_filtered[df_filtered["Likelihood_Score"] >= 4]

    if not high_like.empty and service_cols:
        service_mean = (
            high_like[service_cols]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "feature", 0: "importance"})
        )
        service_mean["feature"] = (
            service_mean["feature"]
            .str.replace("Q27_", "", regex=False)
            .str.replace("_", " ")
        )

        chart_service = (
            alt.Chart(service_mean.head(10))
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Share of high-likelihood users selecting this"),
                y=alt.Y("feature:N", sort="-x", title="Service feature"),
                tooltip=["feature", alt.Tooltip("importance", format=".2f")],
            )
        )
        st.altair_chart(chart_service, use_container_width=True)
        st.caption(
            "Among users who are highly likely to use BorrowBox, features like cleanliness, item quality, availability and "
            "ease of booking are selected most often. These operational factors are critical to customer satisfaction."
        )
    else:
        st.info("Not enough high-likelihood users in the current filter to show service feature importance.")


# =========================
# Prediction on new data
# =========================

def predict_on_new_data(best_pipeline, feature_columns, df_raw):
    st.subheader("Upload dataset to predict willingness")

    sample = df_raw.head(10)[feature_columns]
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sample input template",
        data=csv_bytes,
        file_name="BorrowBox_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file with the same structure as the original survey to see predictions.")
        return

    new_df = pd.read_csv(uploaded)

    missing = [c for c in feature_columns if c not in new_df.columns]
    if missing:
        st.error("The uploaded file is missing some required columns: " + ", ".join(missing))
        return

    X_new = new_df[feature_columns]
    preds = best_pipeline.predict(X_new)
    probs = best_pipeline.predict_proba(X_new)[:, 1]

    result_df = new_df.copy()
    result_df["Predicted_Willing"] = preds
    result_df["Predicted_Willing_Prob"] = probs

    st.write("Preview of predictions:")
    st.dataframe(result_df.head())

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_out,
        file_name="BorrowBox_willingness_predictions.csv",
        mime="text/csv",
    )


# =========================
# Clustering tab
# =========================

def clustering_tab(df_raw: pd.DataFrame):
    st.subheader("Customer Segmentation using Clustering")

    df = add_target_and_scores(df_raw)

    st.markdown(
        """We use **K-Means clustering** to segment respondents into groups with similar characteristics.


        Select the variables you want to cluster on and choose the number of clusters. We will show the cluster visualisation
        using PCA (2D) and a small profile of each cluster."""
    )

    # Suggest a small set of features for clustering
    candidate_features = [
        "Likelihood_Score",
        "Q1_Age_Group",
        "Q4_Monthly_Income",
        "Q11_Need_Frequency",
        "Q30_WTP_Monthly",
    ]
    available_features = [c for c in candidate_features if c in df.columns]

    selected_features = st.multiselect(
        "Select features for clustering",
        options=available_features,
        default=available_features,
    )

    if not selected_features:
        st.info("Please select at least one feature to run clustering.")
        return

    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=6, value=3)

    # Prepare data
    cluster_df = df[selected_features].copy()
    cluster_df_encoded = pd.get_dummies(cluster_df, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df_encoded)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_clustered["PC1"] = coords[:, 0]
    df_clustered["PC2"] = coords[:, 1]

    st.markdown("### Cluster visualisation (PCA 2D projection)")

    chart_clusters = (
        alt.Chart(df_clustered)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("PC1:Q", title="Principal Component 1"),
            y=alt.Y("PC2:Q", title="Principal Component 2"),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=["Cluster", "Q1_Age_Group", "Q4_Monthly_Income", "Likelihood_Score"],
        )
    )
    st.altair_chart(chart_clusters, use_container_width=True)

    st.markdown("### Cluster profiles")

    profile_cols = ["Cluster", "Likelihood_Score"]
    for col in ["Q1_Age_Group", "Q4_Monthly_Income", "Q11_Need_Frequency", "Q30_WTP_Monthly"]:
        if col in df_clustered.columns:
            profile_cols.append(col)

    # Aggregate some basic info: mean likelihood and top categories
    cluster_summary = df_clustered.groupby("Cluster")["Likelihood_Score"].agg(["mean", "count"]).rename(
        columns={"mean": "avg_likelihood", "count": "cluster_size"}
    )
    st.dataframe(cluster_summary.style.format({"avg_likelihood": "{:.2f}"}))

    st.caption(
        "Clusters with higher average likelihood scores and reasonable size represent attractive target segments "
        "for BorrowBox. You can interpret them based on age, income and frequency distributions."
    )


# =========================
# Association Rules tab
# =========================

def association_rules_tab(df_raw: pd.DataFrame):
    st.subheader("Association Rule Mining on Equipment Interests")

    df = add_target_and_scores(df_raw)

    st.markdown(
        """We use **Association Rule Mining (Apriori)** to find patterns like:

        *“People who are interested in borrowing X and Y are also likely to be interested in Z.”*


        This can help design bundle offers, promotions and inventory combinations."""
    )

    equipment_cols = [c for c in df.columns if c.startswith("Q15_") or c.startswith("Q17_") or c.startswith("Q19_") or c.startswith("Q21_")]
    if not equipment_cols:
        st.info("No equipment columns found for association rule mining.")
        return

    basket = df[equipment_cols].astype(bool)

    min_support = st.slider("Minimum support", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    min_lift = st.slider("Minimum lift", min_value=1.0, max_value=5.0, value=1.2, step=0.1)

    if st.button("Run association rules"):
        freq_items = apriori(basket, min_support=min_support, use_colnames=True)
        if freq_items.empty:
            st.warning("No frequent itemsets found with the chosen support. Try lowering the minimum support.")
            return

        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
        if rules.empty:
            st.warning("No rules found with the chosen lift threshold. Try lowering the minimum lift.")
            return

        # Clean up itemset names
        def format_itemset(itemset):
            names = [col.replace("Q15_", "").replace("Q17_", "").replace("Q19_", "").replace("Q21_", "") for col in itemset]
            return ", ".join(names)

        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: format_itemset(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: format_itemset(list(x)))

        display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        rules_display = rules[display_cols].sort_values("lift", ascending=False)

        st.markdown("### Top association rules")
        st.dataframe(rules_display.head(20).style.format({"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}"}))

        st.caption(
            "These rules show which items tend to be liked together. For example, if many users who want a camping tent "
            "also want sleeping bags and a portable stove, we can design bundle packs or ensure we stock these items together."
        )


# =========================
# Main app
# =========================

def main():
    st.title("BorrowBox Shared Gadget Library – Analytics, Segmentation & Prediction")

    st.markdown(
        """
        BorrowBox is a **shared gadget library** concept for communities in places like Dubai Academic City and Silicon Oasis.
        Instead of everyone buying rarely used gadgets, residents can borrow items such as cleaning appliances, tools,
        sports gear and camping equipment.

        This dashboard uses survey data to answer:
        - Who is interested in BorrowBox?
        - What equipment should we stock first?
        - How much are people willing to pay?
        - How can we predict likely users and segment them into clusters?
        - Which items are naturally bundled together using association rules?
        - How strongly are people likely to use BorrowBox (regression on Likelihood Score)?
        """
    )

    # Load data once
    df_raw = load_data()

    # 6 tabs now (including regression)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Market Insights",
            "Model Performance",
            "Predict on New Data",
            "Clustering",
            "Association Rules",
            "Regression (Likelihood Score)",
        ]
    )

    # ----------------------
    # Tab 1 – Market Insights
    # ----------------------
    with tab1:
        df_filtered, equipment_cols = apply_filters(df_raw)
        st.write(f"Filtered sample size: **{len(df_filtered)}** respondents")

        # KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Respondents (filtered)", len(df_filtered))
        with col2:
            if len(df_filtered) > 0:
                st.metric("% willing", f"{df_filtered['target_willing'].mean() * 100:.1f}%")
            else:
                st.metric("% willing", "N/A")
        with col3:
            if len(df_filtered) > 0 and "Q1_Age_Group" in df_filtered.columns:
                top_age = (
                    df_filtered.groupby("Q1_Age_Group")["target_willing"]
                    .mean()
                    .sort_values(ascending=False)
                    .index[0]
                )
                st.metric("Top age segment", top_age)
            else:
                st.metric("Top age segment", "N/A")

        show_insight_charts(df_filtered, equipment_cols)

    # ----------------------
    # Tab 2 – Model Performance (Classification)
    # ----------------------
    with tab2:
        st.subheader("Run classification models")
        st.markdown(
            "Click the button below to train **Decision Tree**, **Random Forest**, and **Gradient Boosting** models "
            "to predict who is willing to use BorrowBox."
        )

        if st.button("Run classification models"):
            metrics_df, cv_df, confusion_figs, roc_fig, best_pipeline, feature_columns = train_and_evaluate_models(
                df_raw
            )
            st.session_state["metrics_df"] = metrics_df
            st.session_state["cv_df"] = cv_df
            st.session_state["confusion_figs"] = confusion_figs
            st.session_state["roc_fig"] = roc_fig
            st.session_state["best_pipeline"] = best_pipeline
            st.session_state["feature_columns"] = feature_columns

        if "metrics_df" in st.session_state:
            st.markdown("### Test set performance")
            st.dataframe(st.session_state["metrics_df"].style.format("{:.3f}"))

            st.markdown("### 5-fold cross-validation performance")
            st.dataframe(st.session_state["cv_df"].style.format("{:.3f}"))

            # Identify best model
            best_model_name = st.session_state["metrics_df"]["test_roc_auc"].idxmax()
            st.markdown(
                f"**Best model based on test ROC-AUC:** {best_model_name}. "
                "This model is recommended for predicting which users are most likely to subscribe to BorrowBox."
            )

            st.markdown("### Confusion matrices (black & white)")
            for name, fig in st.session_state["confusion_figs"].items():
                st.write(name)
                st.pyplot(fig)

            st.markdown("### ROC curves for all models")
            st.pyplot(st.session_state["roc_fig"])
        else:
            st.info("Models have not been run yet. Click **Run classification models** to see performance.")

    # ----------------------
    # Tab 3 – Predict on New Data
    # ----------------------
    with tab3:
        if "best_pipeline" not in st.session_state or "feature_columns" not in st.session_state:
            st.info("Please run the models in the **Model Performance** tab first so we can use the best model.")
        else:
            predict_on_new_data(
                st.session_state["best_pipeline"],
                st.session_state["feature_columns"],
                df_raw,
            )

    # ----------------------
    # Tab 4 – Clustering
    # ----------------------
    with tab4:
        clustering_tab(df_raw)

    # ----------------------
    # Tab 5 – Association Rules
    # ----------------------
    with tab5:
        association_rules_tab(df_raw)

    # ----------------------
    # Tab 6 – Regression (Likelihood Score)
    # ----------------------
    with tab6:
        st.subheader("Regression: Predicting Likelihood Score (2–5)")
        st.markdown(
            "Here we treat the **Likelihood_Score** as a numeric value and use regression models "
            "to predict how strongly a respondent is likely to use BorrowBox."
        )

        if st.button("Run regression models"):
            reg_metrics_df, reg_cv_df, best_reg_name, scatter_fig = train_and_evaluate_regression_models(df_raw)
            st.session_state["reg_metrics_df"] = reg_metrics_df
            st.session_state["reg_cv_df"] = reg_cv_df
            st.session_state["reg_best_name"] = best_reg_name
            st.session_state["reg_scatter_fig"] = scatter_fig

        if "reg_metrics_df" in st.session_state:
            st.markdown("### Test set regression performance (R², MAE, RMSE)")
            st.dataframe(st.session_state["reg_metrics_df"].style.format("{:.3f}"))

            st.markdown("### 5-fold cross-validation performance")
            st.dataframe(st.session_state["reg_cv_df"].style.format("{:.3f}"))

            st.markdown(
                f"**Best regression model by test R²:** {st.session_state['reg_best_name']}."
            )

            st.markdown("### Predicted vs Actual Likelihood Score (best model)")
            st.pyplot(st.session_state["reg_scatter_fig"])
        else:
            st.info("Click **Run regression models** to train the regression models.")
if __name__ == "__main__":
    main()
