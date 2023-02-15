# segmentation_evaluation
Evaluation tools for NHP segmentations. Data organisation should follow BIDS derivatives.


# Procedure:
1a) format_man_dseg for manual seg (merge hemi, redinx, padding)
1b) format_auto_dseg (reindex tissues for 3 classes mostly, for all auto_analysis_names defined in )

2) eval dataset
3) average_errors

(if multiple datasets)
4) merge_average_errors

