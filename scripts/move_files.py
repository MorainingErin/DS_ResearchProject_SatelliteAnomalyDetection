# -*- coding: utf-8 -*-

from pathlib import Path
import os
import shutil


def move_files():

    src_path = Path(r"C:\Users\erin\Documents\Github\DS_ResearchProject\output")
    dst_path = Path(r"C:\Users\erin\Documents\Github\DS_ResearchProject\output\Uploaded")

    mapping_dict = {

        # CryoSat-2
        "CryoSat-2/vis/ARIMA_Brouwer Mean Motion_violins.png": 
            "cryosat2_arima_violin.png",
        "CryoSat-2/vis/XGBoost_Brouwer Mean Motion_violins.png": 
            "cryosat2_xgboost_violin.png",
        "CryoSat-2/exp/Sep_Brouwer Mean Motion.png": 
            "cryosat2-brosepcurve.png",
        "CryoSat-2/vis/ARIMA_Brouwer Mean Motion_residuals_distribution.png": 
            "cryosat2_arima_residual_kdehist.png",
        "CryoSat-2/vis/XGBoost_Brouwer Mean Motion_residuals_distribution.png": 
            "cryosat2_xgboost_residual_kdehist.png",

        # Fengyun-2F
        "Fengyun-2F/vis/ARIMA_Brouwer Mean Motion_violins.png": 
            "fengyun2f_arima_violin.png",
        "Fengyun-2F/vis/XGBoost_Brouwer Mean Motion_violins.png": 
            "fengyun2f_xgboost_violin.png",
        "Fengyun-2F/vis/ARIMA_Brouwer Mean Motion_residuals_distribution.png": 
            "fengyun2f_arima_residual_kdehist.png",
        "Fengyun-2F/vis/XGBoost_Brouwer Mean Motion_residuals_distribution.png": 
            "fengyun2f_xgboost_residual_kdehist.png",
        "Fengyun-2F/exp/Batched_diff_Bro-Ecc-Arg.png": 
            "fengyun2f-batched-diff.png",

        # Fengyun-2H
        "Fengyun-2H/exp/Sep_Brouwer Mean Motion.png": 
            "fengyun2h-brosepcurve.png",

        # SARAL
        "SARAL/vis/ARIMA_Brouwer Mean Motion_violins.png": 
            "saral_arima_violin.png",
        "SARAL/vis/XGBoost_Brouwer Mean Motion_violins.png": 
            "saral_xgboost_violin.png",
        "SARAL/exp/Sep_Brouwer Mean Motion.png": 
            "saral-brosepcurve.png",
        "SARAL/vis/ARIMA_Brouwer Mean Motion_residuals_distribution.png": 
            "saral_arima_residual_kdehist.png",
        "SARAL/vis/XGBoost_Brouwer Mean Motion_residuals_distribution.png": 
            "saral_xgboost_residual_kdehist.png",

        # Sentinel-3A
        "Sentinel-3A/exp/Batched_diff_Bro-Ecc-Arg.png": 
            "sentinel3a-batched-diff.png",

        # PPT
        "Sentinel-3A/exp/Batched_abs_Ecc-Arg-Inc-Mea-Bro-RAA.png": 
            "[pre]sentinel3a-batched-abs-annotated.png",
        "Fengyun-2F/exp/Batched_abs_Ecc-Arg-Inc-Mea-Bro-RAA.png": 
            "[pre]fengyun2f-batched-abs-annotated.png",
        "Fengyun-2F/exp/total_distribution.png": 
            "[pre]fengyun2f-distribution-all-annotated.png",
        "Fengyun-2F/exp/Sep_Brouwer Mean Motion.png": 
            "[pre]fengyun2f-brosepcurve-annotated.png",
        "Fengyun-2F/vis/ARIMA_Brouwer Mean Motion_residual_only.png": 
            "[pre]fengyun2f-arima_residual_curve.png",
        "Fengyun-2F/vis/XGBoost_Brouwer Mean Motion_residual_only.png": 
            "[pre]fengyun2f-xgboost_residual_curve.png",
        "CryoSat-2/vis/ARIMA_Brouwer Mean Motion_residual_only.png": 
            "[pre]cryosat2-arima_residual_curve.png",
        "CryoSat-2/vis/XGBoost_Brouwer Mean Motion_residual_only.png": 
            "[pre]cryosat2-xgboost_residual_curve.png",
        "SARAL/vis/ARIMA_Brouwer Mean Motion_residual_only.png": 
            "[pre]saral-arima_residual_curve.png",
        "SARAL/vis/XGBoost_Brouwer Mean Motion_residual_only.png": 
            "[pre]saral-xgboost_residual_curve.png",
    }

    for src_name, dst_name in mapping_dict.items():
        src_file = src_path / src_name
        if not src_file.exists():
            print(f"Warning: {src_file} not exists.")
            continue

        dst_file = dst_path / dst_name
        if dst_file.exists():
            os.remove(str(dst_file))
        shutil.copy(str(src_file), str(dst_file))
        print(f"Moved: {dst_name}")

    pass


if __name__ == "__main__":
    move_files()
