import preprocessing as pp
import training as tr
import cv2 as cv

# pp.morphological_operations("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000009 (3).png")
# pp.morphological_operations("./Dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000009 (3).png")
# pp.resize("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000009 (3).png")
# pp.generate_dataset("./Dataset/train/normal")
# pp.generate_dataset("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa")
# pp.generate_dataset("./Dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib")
# pp.generate_dataset("./Dataset/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa")
# adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
# large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
# normal
# squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa

# training
# tr.train("model1_4.pth")
tr.eval("model1_4.pth", "./Dataset/test/adenocarcinoma/000117.png")
tr.eval("model1_4.pth", "./Dataset/test/large.cell.carcinoma/000148.png")
tr.eval("model1_4.pth", "./Dataset/test/normal/6.png")
tr.eval("model1_4.pth", "./Dataset/test/squamous.cell.carcinoma/000122.png")
# model1_1 3 epochs
# model1_2 8 epochs
# model1_3 25 epochs
# Dataset\test\adenocarcinoma
# Dataset\test\large.cell.carcinoma
# Dataset\test\normal
# Dataset\test\squamous.cell.carcinoma
