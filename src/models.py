from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy.stats as stats
import statsmodels.api as sm
from pymer4.models import Lm, Lmer
# from rpy2.robjects import Formula, globalenv, pandas2ri
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import ListVector
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

class LMM():
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config

    def variance_inflation_factor(
        self,
        columns_of_interest: Optional[List[str]] = None,
    ) -> None:
        if not columns_of_interest:
            columns_of_interest = self.experiment_config.predictors
        X = self.model.data[columns_of_interest]
        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(
            {n: v for n, v in zip(vif_data.feature, vif_data.VIF)}
        )

    def calc_variance_explained(self):
        if hasattr(self.model, "rsquared"):
            print(
                "Var_explained_percent",
                self.model.rsquared * 100,
            )
        elif (
            hasattr(self.model, "data")
            and hasattr(self.model, "residuals")
            and hasattr(self.model, "design_matrix")
        ):
            from pymer4.stats import rsquared

            rsq = rsquared(
                self.model.data["rt_ego"].to_numpy(),
                self.model.residuals,
                any("intercept" in element.lower() for element in self.model.design_matrix.columns),
            )
            print("Var_explained_percent", rsq * 100)

    def breusch_pagan_test(
        self,
        columns_of_interest: Optional[List[str]] = None,
    ) -> None:
        if hasattr(self.model, "residuals"):
            ### p < 0.05 -> heteroscedasticity, p > 0.05 homoscedasticity
            if not columns_of_interest:
                columns_of_interest = self.experiment_config.predictors
            X = self.model.data[columns_of_interest]
            X = sm.add_constant(X)
            bp_test = het_breuschpagan(self.model.residuals, X)
            bp_test_labels = [
                "Lagrange multiplier statistic",
                "p-value",
                "f-value",
                "f p-value",
            ]
            res_dict = {x: y for x, y in zip(bp_test_labels, bp_test)}
            print("Breusch-Pagan (heteroscedasticity):", res_dict)

    def plot_observed_vs_fitted(self) -> None:
        if hasattr(self.model, "fits") and hasattr(self.model, "data"):
            observed = self.model.data["rt_ego"]
            fitted = self.model.fits
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(observed, fitted, alpha=0.3)
            obs_min = observed.min()
            obs_max = observed.max()
            plt.plot(
                [obs_min, obs_max],
                [obs_min, obs_max],
                "r--",
            )  # Line of equality
            plt.ylim(obs_min, obs_max)
            plt.xlabel("Observed Values")
            plt.ylabel("Predicted Values")
            plt.title("Observed vs Fitted")
            plt.grid()

    def plot_fitted_vs_residuals(self) -> None:
        if hasattr(self.model, "residuals") and hasattr(self.model, "data"):
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                self.model.fits,
                self.model.residuals,
            )
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            plt.title("Fitted vs. Residuals")
            predictors = self.experiment_config.predictors
            for predictor in predictors:
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(
                    self.model.data[predictor],
                    self.model.residuals,
                )
                plt.axhline(0, linestyle="--", color="red")
                plt.xlabel(predictor)
                plt.ylabel("Residuals")
                plt.title(f"Residuals vs {predictor}")

    def plot_qq(self, what: str) -> None:
        fig = plt.figure(figsize=(10, 6))
        if what == "residuals":
            if hasattr(self.model, "residuals"):
                stats.probplot(
                    self.model.residuals,
                    dist="norm",
                    plot=plt,
                )
                plt.title("QQ Plot of Residuals")
        elif what == "ranef conversation" and self.experiment_config.model == "Lmer":
            if hasattr(self.model, "grps") and hasattr(self.model, "ranef"):
                if "conversation" in self.model.grps.keys():
                    stats.probplot(
                        np.array(self.model.ranef.iloc[:]["X.Intercept."]),
                        dist="norm",
                        plot=plt,
                    )
                    plt.title("QQ Plot of conversation random effects")
        elif what == "ranef donor" and self.experiment_config.model == "Lmer":
            if hasattr(self.model, "grps") and hasattr(self.model, "ranef"):
                try:
                    stats.probplot(
                        np.array(self.model.ranef.iloc[:]["X.Intercept."]),
                        dist="norm",
                        plot=plt,
                    )
                    plt.title("QQ Plot of donor random effects")
                except (KeyError, IndexError):
                    return
        elif self.experiment_config.model == "Lmer":
            raise NotImplementedError(
                f"QQ plot can be done for 'residuals' or 'ranef' but 'what' was {what}"
            )

    def durbin_watson_autocorrelation(
        self,
    ) -> None:
        ### 2 is no autocorrelation, 0 is positive, 4 negative. <1.5 or >2.5 is autocorrelation.
        dw = durbin_watson(self.model.residuals)
        print("Durbin Watson autocorrelation:",dw)
    
    def shapiro_wilk_normality(self, what: str) -> None:
        try:
            if what == "residuals":
                shapiro_stat, p_val = stats.shapiro(self.model.residuals)
                print(
                    "Shapiro-Wilk residuals p-value",
                    p_val,
                )
            elif what == "ranef conversation" and self.experiment_config.model == "Lmer":
                if "conversation" not in self.model.grps.keys():
                    return
                shapiro_stat, p_val = stats.shapiro(self.model.ranef[:]["X.Intercept."])
                print(
                    "Shapiro-Wilk conversation residuals p-value",
                    p_val,
                )
            elif what == "ranef donor" and self.experiment_config.model == "Lmer":
                if "donor" not in self.model.grps.keys():
                    return
                shapiro_stat, p_val = stats.shapiro(
                    self.model.ranef.iloc[:]["X.Intercept."], nan_policy="raise"
                )
                print(
                    "Shapiro-Wilk donor residuals p-value",
                    p_val,
                )

            elif self.experiment_config.model == "Lmer":
                raise NotImplementedError(
                    f"Shapiro-Wilk can be done for 'residuals' or 'ranef' but 'what' was {what}"
                )
        except (KeyError, ValueError) as e:
            print(f"Skipping Shapiro Wilk due to the following error with 'what={what}': {e}")
            return
    
    def eval_model(self) -> None:
        ### Multicollinearity
        self.variance_inflation_factor()

        self.calc_variance_explained()

        ### Linear relationship between X and Y
        self.plot_fitted_vs_residuals()
        self.plot_observed_vs_fitted()

        ### Homoscedasticity
        self.breusch_pagan_test()

        ### Independence of residuals
        self.durbin_watson_autocorrelation()

        ### Normality of residuals:
        self.shapiro_wilk_normality(what="residuals")
        self.plot_qq(what="residuals")

        ### Normality of random effects:
        self.shapiro_wilk_normality(what="ranef conversation") 
        self.shapiro_wilk_normality(what="ranef donor")
        # self.plot_qq(what="ranef conversation") # Not relevant for donor-level agg
        self.plot_qq(what="ranef donor")

    def fit_lmm(self, df):

        def _get_warnings() -> List:
            warnings = ro.r["warnings"]
            warn_result = warnings()
            print(warn_result)
            return warn_result

        donor_levels = list(df.donor_id.unique())
        if hasattr(df, "conversation_id"):
            conversation_levels = list(df.conversation_id.unique())
            factors = {
                "donor_id": donor_levels,
                "conversation_id": conversation_levels,
            }
        else:
            factors = {"donor_id": donor_levels}

        ### Sanity check
        nan_sum = df.isna().sum().sum()
        inf_sum = np.isinf(df.select_dtypes(include="number")).sum().sum()
        assert (nan_sum == 0) and (inf_sum == 0), f"Model fitting: NaNs in data: {nan_sum}. Infs in data: {inf_sum}"

        if self.experiment_config.model == "Lmer":
            self.model = Lmer(
                formula=self.experiment_config.lmm_formula,
                data=df,
                family=self.experiment_config.distribution,
            )
        elif self.experiment_config.model == "Lm":
            self.model = Lm(
                formula=self.experiment_config.lmm_formula,
                data=df,
                family=self.experiment_config.distribution,
            )
        else:
            raise NotImplementedError(
                f"The specified model {self.experiment_config.model} is not implemented"
            )

        def _fit():
            optCtrl_str = (
                "list("
                + ", ".join(
                    [
                        f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}"
                        for key, value in {
                            key: value
                            for key, value in dict(self.experiment_config.optCtrl).items()
                            if value is not None
                        }.items()
                    ]
                )
                + ")"
            )
            result = self.model.fit(
                verbose=True,
                factors=factors,
                n_boot=self.experiment_config.n_boot,
                control=f"optimizer='{self.experiment_config.optimizer}', optCtrl={optCtrl_str}",
                tol=1e-3,
                disp=True,
            )
            return result

        result = _fit()

        _ = _get_warnings()

        print("AIC", self.model.AIC)
        print("BIC", self.model.BIC)
        print("log likelihood", self.model.logLike)

        def _get_ICC():
            total_var = self.model.ranef_var["Var"].sum()
            residual = self.model.ranef_var.loc["Residual", "Var"]
            return (total_var - residual) / total_var

        print("ICC", _get_ICC())
        return result