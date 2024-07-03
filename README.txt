最终结果保存在full_res文件夹中，有Rs,ts存储了11个相机的位姿；还有点云坐标以及对应的颜色，一一对应。
init_res是初步重建的结果，可以通过save_res.py文件进行可视化和保存。（需要首先修改路径）
no_ba, no_ba_pnp, no_pnp, one_ba分别是报告中提到的消融实验的结果。
matches是11张图两两配对的示意图。
SIFT是特征点提取的示意图。