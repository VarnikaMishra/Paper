import os

import joblib
import streamlit as st

from predict import predict


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "defect_model.pth"
)

FEATURES_PATH = os.path.join(
    "models",
    "features.pkl"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "Software Defect Prediction System"
)

st.markdown(
    """
    ### Hybrid LinkNet + Bi-LSTM Model

    Enter the software metrics below to predict whether
    the software instance is **defective** or
    **non-defective**.

    The prediction uses:

    **Improved LinkNet + Bi-LSTM + Score-Level Fusion**
    """
)


# ============================================================
# CHECK TRAINED MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    st.error(
        """
        Trained model not found.

        Please train the model first by running:

        `python train.py`
        """
    )

    st.stop()


if not os.path.exists(
    FEATURES_PATH
):

    st.error(
        """
        Feature configuration not found.

        Please run:

        `python train.py`
        """
    )

    st.stop()


# ============================================================
# LOAD FEATURE LIST
# ============================================================

feature_columns = joblib.load(
    FEATURES_PATH
)


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "About the Dataset",
    expanded=False
):

    st.write(
        """
        This system uses the supplied CM1 software
        defect dataset.

        The dataset contains numerical software metrics
        together with a `defects` label.
        """
    )

    st.write(
        "Number of input features:",
        len(feature_columns)
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "Software Metrics"
)

st.write(
    "Enter the values for the software metrics."
)


# ============================================================
# CREATE INPUT FIELDS
# ============================================================

input_data = {}

columns_per_row = 3


for start in range(
    0,
    len(feature_columns),
    columns_per_row
):

    row_features = feature_columns[
        start:start + columns_per_row
    ]

    columns = st.columns(
        len(row_features)
    )

    for column, feature in zip(
        columns,
        row_features
    ):

        with column:

            input_data[feature] = (
                st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.6f",
                    key=f"input_{feature}"
                )
            )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "Predict Defect",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        with st.spinner(
            "Analyzing software metrics..."
        ):

            result = predict(
                input_data
            )

        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        st.divider()

        st.header(
            "Prediction Result"
        )

        prediction = result[
            "prediction_label"
        ]

        confidence = result[
            "confidence"
        ]

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        if prediction == "Defective":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Final score-level fusion
        # ----------------------------------------------------

        st.subheader(
            "Final Score-Level Fusion"
        )

        fusion_col1, fusion_col2 = (
            st.columns(2)
        )

        with fusion_col1:

            st.metric(
                "Non-Defective",
                (
                    f"{result['non_defective_probability'] * 100:.2f}%"
                )
            )

        with fusion_col2:

            st.metric(
                "Defective",
                (
                    f"{result['defective_probability'] * 100:.2f}%"
                )
            )

        # ----------------------------------------------------
        # Individual model scores
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Model Scores"
        )

        linknet_col, bilstm_col = (
            st.columns(2)
        )

        # ====================================================
        # IMPROVED LINKNET
        # ====================================================

        with linknet_col:

            st.markdown(
                "### Improved LinkNet"
            )

            st.metric(
                "Non-Defective",
                (
                    f"{result['linknet_non_defective'] * 100:.2f}%"
                )
            )

            st.metric(
                "Defective",
                (
                    f"{result['linknet_defective'] * 100:.2f}%"
                )
            )

        # ====================================================
        # BI-LSTM
        # ====================================================

        with bilstm_col:

            st.markdown(
                "### Bi-LSTM"
            )

            st.metric(
                "Non-Defective",
                (
                    f"{result['bilstm_non_defective'] * 100:.2f}%"
                )
            )

            st.metric(
                "Defective",
                (
                    f"{result['bilstm_defective'] * 100:.2f}%"
                )
            )

        # ----------------------------------------------------
        # Probability visualization
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Final Prediction Probabilities"
        )

        probability_data = {

            "Class": [
                "Non-Defective",
                "Defective"
            ],

            "Probability": [

                result[
                    "non_defective_probability"
                ],

                result[
                    "defective_probability"
                ]
            ]
        }

        st.bar_chart(
            probability_data,
            x="Class",
            y="Probability"
        )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Software Defect Prediction | "
    "Hybrid Improved LinkNet + Bi-LSTM + Score-Level Fusion"
)