import preprocessing as pp
import training as tr
import testing as te
import cv2 as cv

# pp.morphological_operations("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000009 (3).png")
# pp.morphological_operations("./Dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000009 (3).png")
# pp.resize("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000009 (3).png")
# pp.generate_dataset("./Dataset/test/normal", "./Preprocessed/test/normal")
# pp.generate_dataset("./Dataset/test/large.cell.carcinoma", "./Preprocessed/test/large.cell.carcinoma")
# pp.generate_dataset("./Dataset/test/adenocarcinoma", "./Preprocessed/test/adenocarcinoma")
# pp.generate_dataset("./Dataset/test/squamous.cell.carcinoma", "./Preprocessed/test/squamous.cell.carcinoma")

# pp.generate_dataset("./Dataset/train/normal", "./Preprocessed/train/normal")
# pp.generate_dataset("./Dataset/train/large.cell.carcinoma", "./Preprocessed/train/large.cell.carcinoma")
# pp.generate_dataset("./Dataset/train/adenocarcinoma", "./Preprocessed/train/adenocarcinoma")
# pp.generate_dataset("./Dataset/train/squamous.cell.carcinoma", "./Preprocessed/train/squamous.cell.carcinoma")

# pp.generate_dataset("./Dataset/valid/normal", "./Preprocessed/valid/normal")
# pp.generate_dataset("./Dataset/valid/large.cell.carcinoma", "./Preprocessed/valid/large.cell.carcinoma")
# pp.generate_dataset("./Dataset/valid/adenocarcinoma", "./Preprocessed/valid/adenocarcinoma")
# pp.generate_dataset("./Dataset/valid/squamous.cell.carcinoma", "./Preprocessed/valid/squamous.cell.carcinoma")
# adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
# large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
# normal
# squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa

# training
tr.train("model5_4.pth")
# te.eval("model3_9.pth", "./Preprocessed/test/adenocarcinoma/000117.png")
# te.eval("model3_9.pth", "./Preprocessed/test/large.cell.carcinoma/000148.png")
# te.eval("model3_9.pth", "./Preprocessed/test/normal/6.png")
# te.eval("model3_9.pth", "./Preprocessed/test/squamous.cell.carcinoma/000122.png")
# model1_1 3 epochs
# model1_2 8 epochs
# model1_3 25 epochs
# Dataset\test\adenocarcinoma
# Dataset\test\large.cell.carcinoma
# Dataset\test\normal
# Dataset\test\squamous.cell.carcinoma

# te.test("model5_3.pth")
# tr.check_dataset()

