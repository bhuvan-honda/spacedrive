# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import pickle
import os
import json
from os import path as osp


def main(args):
    pred_path = args.pred_path

    if pred_path[-1] != "/":
        pred_path += "/"

    print("The current pred_path", pred_path)
    cf_dir = osp.join(args.base_path, args.anno_path)
    cf_files = os.listdir(cf_dir)

    tp_collision = 0
    fp_collision = 0
    fn_collision = 0
    tn_no_collision = 0

    tp_red_light = 0
    fp_red_light = 0
    fn_red_light = 0
    tn_no_red_light = 0

    tp_out_of_drivable = 0
    fp_out_of_drivable = 0
    fn_out_of_drivable = 0
    tn_out_of_drivable = 0

    tp_safe = 0  # if no risks are predicted and no risks in annotation
    fp_safe = 0
    fn_safe = 0
    tn_safe = 0

    for cf_file in cf_files:
        if cf_file.endswith(".pkl"):
            sample_id = cf_file.replace(".pkl", "")
            cf_annos = pickle.load(open(osp.join(cf_dir, cf_file), "rb"))

            if os.path.exists(pred_path + sample_id):
                with open(pred_path + sample_id, "r", encoding="utf8") as f:

                    pred_data = json.load(f)
                    pred_question = pred_data[0]["Q"]
                    pred_cf = pred_data[0]["C"]

                    pred_cf_list = []

                    if "red" in pred_cf:
                        pred_cf_list.append("Run the red light")

                    if "Collision" in pred_cf:
                        pred_cf_list.append("Collision")

                    if "drivable area" in pred_cf:
                        pred_cf_list.append("Out of the drivable area")

                    for cf_anno in cf_annos:
                        anno_traj = cf_anno["traj"]
                        anno_status = cf_anno[
                            "status"
                        ]

                        if anno_traj[5:-1] in pred_question:
                            # compare pred_cf_list and anno_status
                            for risk in [
                                "Collision",
                                "Run the red light",
                                "Out of the drivable area",
                            ]:
                                if risk in anno_status and risk in pred_cf_list:
                                    if risk == "Collision":
                                        tp_collision += 1
                                    elif risk == "Run the red light":
                                        tp_red_light += 1
                                    elif risk == "Out of the drivable area":
                                        tp_out_of_drivable += 1
                                elif risk in anno_status and risk not in pred_cf_list:
                                    if risk == "Collision":
                                        fn_collision += 1
                                    elif risk == "Run the red light":
                                        fn_red_light += 1
                                    elif risk == "Out of the drivable area":
                                        fn_out_of_drivable += 1
                                elif risk not in anno_status and risk in pred_cf_list:
                                    if risk == "Collision":
                                        fp_collision += 1
                                    elif risk == "Run the red light":
                                        fp_red_light += 1
                                    elif risk == "Out of the drivable area":
                                        fp_out_of_drivable += 1
                                elif (
                                    risk not in anno_status and risk not in pred_cf_list
                                ):
                                    if risk == "Collision":
                                        tn_no_collision += 1
                                    elif risk == "Run the red light":
                                        tn_no_red_light += 1
                                    elif risk == "Out of the drivable area":
                                        tn_out_of_drivable += 1

                            if anno_status == [] and pred_cf_list == []:
                                tp_safe += 1
                            elif anno_status == [] and pred_cf_list != []:
                                fn_safe += 1
                            elif anno_status != [] and pred_cf_list == []:
                                fp_safe += 1
                            else:
                                tn_safe += 1

    # -------- Precision & Recall helper --------
    def precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(
        "number of samples",
        tp_collision + tn_no_collision + fp_collision + fn_collision,
    )

    print(
        "Collision - TP:",
        tp_collision,
        "FP:",
        fp_collision,
        "FN:",
        fn_collision,
        "TN:",
        tn_no_collision,
    )
    print(
        "Red Light - TP:",
        tp_red_light,
        "FP:",
        fp_red_light,
        "FN:",
        fn_red_light,
        "TN:",
        tn_no_red_light,
    )
    print(
        "Out of Drivable Area - TP:",
        tp_out_of_drivable,
        "FP:",
        fp_out_of_drivable,
        "FN:",
        fn_out_of_drivable,
        "TN:",
        tn_out_of_drivable,
    )
    print("Safe - TP:", tp_safe, "FP:", fp_safe, "FN:", fn_safe, "TN:", tn_safe)

    # -------- Accuracy --------
    print("\nAccuracy:")
    print(
        "Collision accuracy:",
        (tp_collision + tn_no_collision)
        / (tp_collision + tn_no_collision + fp_collision + fn_collision),
    )
    print(
        "Red Light accuracy:",
        (tp_red_light + tn_no_red_light)
        / (tp_red_light + tn_no_red_light + fp_red_light + fn_red_light),
    )
    print(
        "Out of Drivable Area accuracy:",
        (tp_out_of_drivable + tn_out_of_drivable)
        / (
            tp_out_of_drivable
            + tn_out_of_drivable
            + fp_out_of_drivable
            + fn_out_of_drivable
        ),
    )
    print(
        "Safe accuracy:", (tp_safe + tn_safe) / (tp_safe + tn_safe + fp_safe + fn_safe)
    )

    # -------- Precision --------
    print("\nPrecision:")
    print("Collision precision:", precision(tp_collision, fp_collision))
    print("Red Light precision:", precision(tp_red_light, fp_red_light))
    print(
        "Out of Drivable Area precision:",
        precision(tp_out_of_drivable, fp_out_of_drivable),
    )
    print("Safe precision:", precision(tp_safe, fp_safe))

    # -------- Recall --------
    print("\nRecall:")
    print("Collision recall:", recall(tp_collision, fn_collision))
    print("Red Light recall:", recall(tp_red_light, fn_red_light))
    print(
        "Out of Drivable Area recall:", recall(tp_out_of_drivable, fn_out_of_drivable)
    )
    print("Safe recall:", recall(tp_safe, fn_safe))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="../data/nuscenes/",
        help="Base path to the data.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="results_planning_only/",
        help="Path to the prediction results.",
    )
    parser.add_argument(
        "--anno_path", type=str, default="eval_cf", help="Path to the annotation file."
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of threads to use."
    )

    parser.add_argument(
        "--discrete_coords",
        type=float,
        default=0,
        help="resolution of discrete coordinates.",
    )

    args = parser.parse_args()
    main(args)
