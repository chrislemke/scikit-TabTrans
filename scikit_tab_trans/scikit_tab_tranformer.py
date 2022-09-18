# -*- coding: utf-8 -*-

from datetime import datetime
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory, ModelCheckpoint
from pytorch_widedeep.metrics import FBetaScore, Precision, Recall
from pytorch_widedeep.models import TabTransformer, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted
from torch.optim import AdamW

date_time = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))


class TabTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple implementation wrapping the [TabTransformer](https://arxiv.org/pdf/2012.06678.pdf)
    from [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) into a [Scikit-learn](https://scikit-learn.org/stable/) model.
    So it can be e.g. used as a part of a `[StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)`.
    Parts of this documentation are copied from [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep). Check it out - it's a great project!

    Parameters
    ----------
    epochs: int ,default = 1
        Number of epochs to train the model.
    batch_size: int, default = 32
        Batch size to train the model.
    scaling: bool, default = True
        Whether to scale the continuous columns in preprocessing.
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded at
        the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    embed_continuous: bool, default = False
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    use_qkv_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers.
    n_blocks: int, default = 4
        Number of Transformer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l,
        4\times l, 2 \times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`
    threshold: float, default = 0.5
        Threshold used for the prediction. If `pred_proba >= threshold` then `1` else `0`.
    model_save_path: str, Optional, default = None
        Path to save the model. If `None` the model will not be saved.
    preprocessor_save_path: str, Optional, default = None
        Path to save the preprocessor. If `None` the preprocessor will not be saved.
    scaler_save_path: str, Optional, default = None
        Path to save the scaler. If `None` the scaler will not be saved.
    focal_loss_alpha: float, default = 0.2
        Focal loss alpha parameter. Checkout the [paper](https://arxiv.org/pdf/1708.02002.pdf) for more information.
    focal_loss_gamma: float, default = 1.0
        Focal loss gamma parameter. Checkout the [paper](https://arxiv.org/pdf/1708.02002.pdf) for more information.
    metric_f_beta: int, default = 1
        Beta value for f beta score.
    verbose: int, default = 1
    seed: int, default = 42
    early_stopping_patience: int, default = 10
        Number of epochs with no improvement after which training will be stopped.
        Number of epochs to keep in the learning rate history.
    checkpoint_filepath: str, Optional, default = None
        Path to save the model checkpoint. If `None` the model checkpoint will not be saved.
    save_best_checkpoint_only: bool, default = False
        Boolean indicating whether or not to save the best model checkpoint only.
    """

    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        scaling: bool = True,
        cat_embed_dropout: float = 0.1,
        use_cat_bias: bool = False,
        cat_embed_activation: Optional[str] = None,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[str] = None,
        embed_continuous: bool = False,
        cont_embed_dropout: float = 0.1,
        use_cont_bias: bool = True,
        cont_embed_activation: Optional[str] = None,
        input_dim: int = 32,
        n_heads: int = 8,
        use_qkv_bias: bool = False,
        n_blocks: int = 4,
        attn_dropout: float = 0.2,
        ff_dropout: float = 0.1,
        transformer_activation: str = "gelu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
        threshold: float = 0.5,
        model_save_path: Optional[str] = None,
        preprocessor_save_path: Optional[str] = None,
        scaler_save_path: Optional[str] = None,
        focal_loss_alpha: float = 0.2,
        focal_loss_gamma: float = 1.0,
        metric_f_beta: int = 1,
        verbose: int = 1,
        seed: int = 42,
        early_stopping_patience: int = 10,
        checkpoint_filepath: Optional[str] = None,
        save_best_checkpoint_only: bool = False,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.scaling = scaling
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.use_qkv_bias = use_qkv_bias
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.threshold = threshold
        self.model_save_path = model_save_path
        self.preprocessor_save_path = preprocessor_save_path
        self.scaler_save_path = scaler_save_path
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.metric_f_beta = metric_f_beta
        self.verbose = verbose
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_filepath = checkpoint_filepath
        self.save_best_checkpoint_only = save_best_checkpoint_only
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocessor_: Optional[TabPreprocessor] = None
        self.tab_transformer_: Optional[TabTransformer] = None
        self.wide_model_: Optional[WideDeep] = None
        self.trainer_: Optional[Trainer] = None
        self.classes_: Optional[np.ndarray] = None
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[np.ndarray] = None

    def fit(
        self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]
    ) -> "TabTransformerClassifier":

        """
        Fit the model to the data. Different to the classical fit method of [Scikit-learn](https://scikit-learn.org/stable/) this method only accepts
        [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) as X. This is necessary because the
        [TabPreprocessor](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/preprocessing.html#pytorch_widedeep.preprocessing.tab_preprocessor.TabPreprocessor)
        expects a DataFrame as input.

        Parameters
        ----------
        X: pd.DataFrame
            The input data which will be proceeded to fit the [TabTransformer](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/model_components.html#pytorch_widedeep.models.tabular.transformers.tab_transformer.TabTransformer) model.
        y: np.ndarray, pd.Series, Union
            The target data. It can be either a numpy array or a pandas Series.

        Returns
        -------
        TabTransformerClassifier: Fitted TabTransformerClassifier model
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame!")

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        tab_preprocessor, cat_embed_input, x_tab = self.__preprocessing(X)
        self.preprocessor_ = tab_preprocessor
        self.tab_transformer_ = self.__tab_transformer(cat_embed_input)

        self.wide_model_ = WideDeep(deeptabular=self.tab_transformer_)
        self.trainer_ = self.__trainer_with_model()

        self.trainer_.fit(
            X_tab=x_tab,
            target=y,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
        )

        if self.model_save_path:
            self.trainer_.save(
                path=self.model_save_path,
                model_filename=f"sk_tab_tansformer_{date_time}.pt",
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target for the input data. Other than the classical predict method of [Scikit-learn](https://scikit-learn.org/stable/)
        this method only accepts [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) as X.
        This is necessary because the [TabPreprocessor](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/preprocessing.html#pytorch_widedeep.preprocessing.tab_preprocessor.TabPreprocessor)
        expects a DataFrame as input. Other then the usual implementation this `predict` method actually used `predict_proba` under the hood so
        that the user can specify the threshold for the probability of the positive class.

        Parameters
        ----------
        X: pd.DataFrame

        Returns
        -------
        np.ndarray: Predicted target
        """
        if self.preprocessor_ is None or self.trainer_ is None:
            raise ValueError("TabPreprocessor and Trainer must be fitted!")

        check_is_fitted(self)
        X = check_array(X)

        x_tab = self.preprocessor_.transform(X)
        return (self.trainer_.predict_proba(x_tab)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Parameters
        ----------
        X: pd.DataFrame

        Returns
        -------
        np.ndarray: Predicted target
        """
        if self.preprocessor_ is None or self.trainer_ is None:
            raise ValueError("TabPreprocessor and Trainer must be fitted!")

        check_is_fitted(self)
        X = check_array(X)

        x_tab = self.preprocessor_.transform(X)
        return self.trainer_.predict_proba(X_tab=x_tab)

    def __tab_transformer(self, cat_embed_input) -> TabTransformer:
        if self.preprocessor_ is None:
            raise ValueError("TabPreprocessor must be fitted!")

        return TabTransformer(
            column_idx=self.preprocessor_.column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=self.cat_embed_dropout,
            use_cat_bias=self.use_cat_bias,
            cat_embed_activation=self.cat_embed_activation,
            full_embed_dropout=self.full_embed_dropout,
            shared_embed=self.shared_embed,
            add_shared_embed=self.add_shared_embed,
            frac_shared_embed=self.frac_shared_embed,
            continuous_cols=self.continuous_cols,
            cont_norm_layer=self.cont_norm_layer,
            embed_continuous=self.embed_continuous,
            cont_embed_dropout=self.cont_embed_dropout,
            use_cont_bias=self.use_cont_bias,
            cont_embed_activation=self.cont_embed_activation,
            input_dim=self.input_dim,
            n_heads=self.n_heads,
            use_qkv_bias=self.use_qkv_bias,
            n_blocks=self.n_blocks,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            transformer_activation=self.transformer_activation,
            mlp_hidden_dims=self.mlp_hidden_dims,
            mlp_activation=self.mlp_activation,
            mlp_dropout=self.mlp_dropout,
            mlp_batchnorm=self.mlp_batchnorm,
            mlp_batchnorm_last=self.mlp_batchnorm_last,
            mlp_linear_first=self.mlp_linear_first,
        )

    def __preprocessing(
        self, df: pd.DataFrame
    ) -> Tuple[TabPreprocessor, List[Tuple[str, int]], np.ndarray]:
        cont_columns = list(df.select_dtypes(include=[np.float32, np.int16]).columns)
        cat_columns = list(df.select_dtypes(include=["category"]).columns)
        cat_embed_input = [(col, len(df[col].unique())) for col in cat_columns]

        tab_preprocessor = TabPreprocessor(
            cat_embed_cols=cat_columns,
            continuous_cols=cont_columns,
            with_attention=True,
            with_cls_token=True,
            shared_embed=True,
            scale=self.scaling,
            verbose=self.verbose,
        )

        x_tab = tab_preprocessor.fit_transform(df)

        if self.preprocessor_save_path:
            joblib.dump(tab_preprocessor, self.preprocessor_save_path)

        if self.scaler_save_path:
            joblib.dump(tab_preprocessor.scaler, self.scaler_save_path)

        return tab_preprocessor, cat_embed_input, x_tab

    def __trainer_with_model(self):
        tab_optimizer = AdamW(
            self.wide_model_.deeptabular.parameters(),
        )

        callbacks = [
            LRHistory(n_epochs=self.epochs),
            EarlyStopping(patience=self.early_stopping_patience),
        ]

        if self.checkpoint_filepath is not None:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.checkpoint_filepath,
                    save_best_only=self.save_best_checkpoint_only,
                )
            )
        if self.wide_model_ is None:
            raise ValueError("WideDeep model must be fitted!")
        return Trainer(
            self.wide_model_,
            "binary_focal_loss",
            alpha=self.focal_loss_alpha,  # the alpha parameter of the focal loss
            gamma=self.focal_loss_gamma,  # the gamma parameter of the focal loss
            optimizers={"deeptabular": tab_optimizer},
            metrics=[FBetaScore(beta=self.metric_f_beta), Recall(), Precision()],
            verbose=self.verbose,
            seed=self.seed,
            callbacks=callbacks,
        )


# if __name__ == "__main__":
#     import dask.dataframe as dd

#     dir_path = "/Users/ch.lemke/Developer/fps-payment-defaults/data/parquet/feature_selected_df_train"
#     df = dd.read_parquet(dir_path).compute().reset_index(drop=True)

#     dir_path = "/Users/ch.lemke/Developer/fps-payment-defaults/data/parquet/feature_selected_df_test"

#     df_test = dd.read_parquet(dir_path).compute().reset_index(drop=True)

#     X_train_raw = df.drop(["is_payment_default"], axis=1)
#     y_train = df["is_payment_default"]
#     X_test_raw = df_test.drop(["is_payment_default"], axis=1)
#     y_test = df_test["is_payment_default"]

#     model = TabTransformerClassifier()

#     model.fit(X_test_raw, y_test)
