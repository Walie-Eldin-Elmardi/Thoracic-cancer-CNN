import preprocessing as pp
import training as tr
import testing as te
import cv2 as cv

image = cv.imread("./Dataset/test/large.cell.carcinoma/000174.png")
pp.display_images(image, pp.contrast(pp.gray_scale(image), 6.0))
# pp.sharpen(cv.imread("./Dataset/train/normal/11 (2).png"))
# pp.morphological_operations(cv.imread("./Dataset/train/normal/6.png"))
# pp.morphological_operations("./Dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000009 (3).png")
# pp.resize("./Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000009 (3).png")
# pp.generate_texture_dataset("./Dataset/test/normal", "./Preprocessed/texture/test/normal")
# pp.generate_texture_dataset("./Dataset/test/large.cell.carcinoma", "./Preprocessed/texture/test/large.cell.carcinoma")
# pp.generate_texture_dataset("./Dataset/test/adenocarcinoma", "./Preprocessed/texture/test/adenocarcinoma")
# pp.generate_texture_dataset("./Dataset/test/squamous.cell.carcinoma", "./Preprocessed/texture/test/squamous.cell.carcinoma")

# pp.generate_texture_dataset("./Dataset/train/normal", "./Preprocessed/texture/train/normal")
# pp.generate_texture_dataset("./Dataset/train/large.cell.carcinoma", "./Preprocessed/texture/train/large.cell.carcinoma")
# pp.generate_texture_dataset("./Dataset/train/adenocarcinoma", "./Preprocessed/texture/train/adenocarcinoma")
# pp.generate_texture_dataset("./Dataset/train/squamous.cell.carcinoma", "./Preprocessed/texture/train/squamous.cell.carcinoma")

# pp.generate_texture_dataset("./Dataset/valid/normal", "./Preprocessed/texture/valid/normal")
# pp.generate_texture_dataset("./Dataset/valid/large.cell.carcinoma", "./Preprocessed/texture/valid/large.cell.carcinoma")
# pp.generate_texture_dataset("./Dataset/valid/adenocarcinoma", "./Preprocessed/texture/valid/adenocarcinoma")
# pp.generate_texture_dataset("./Dataset/valid/squamous.cell.carcinoma", "./Preprocessed/texture/valid/squamous.cell.carcinoma")

# pp.generate_edge_dataset("./Dataset/test/normal", "./Preprocessed/edges/test/normal")
# pp.generate_edge_dataset("./Dataset/test/large.cell.carcinoma", "./Preprocessed/edges/test/large.cell.carcinoma")
# pp.generate_edge_dataset("./Dataset/test/adenocarcinoma", "./Preprocessed/edges/test/adenocarcinoma")
# pp.generate_edge_dataset("./Dataset/test/squamous.cell.carcinoma", "./Preprocessed/edges/test/squamous.cell.carcinoma")

# pp.generate_edge_dataset("./Dataset/train/normal", "./Preprocessed/edges/train/normal")
# pp.generate_edge_dataset("./Dataset/train/large.cell.carcinoma", "./Preprocessed/edges/train/large.cell.carcinoma")
# pp.generate_edge_dataset("./Dataset/train/adenocarcinoma", "./Preprocessed/edges/train/adenocarcinoma")
# pp.generate_edge_dataset("./Dataset/train/squamous.cell.carcinoma", "./Preprocessed/edges/train/squamous.cell.carcinoma")

# pp.generate_edge_dataset("./Dataset/valid/normal", "./Preprocessed/edges/valid/normal")
# pp.generate_edge_dataset("./Dataset/valid/large.cell.carcinoma", "./Preprocessed/edges/valid/large.cell.carcinoma")
# pp.generate_edge_dataset("./Dataset/valid/adenocarcinoma", "./Preprocessed/edges/valid/adenocarcinoma")
# pp.generate_edge_dataset("./Dataset/valid/squamous.cell.carcinoma", "./Preprocessed/edges/valid/squamous.cell.carcinoma")
# adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
# large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
# normal
# squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa

# training
# tr.train("model7_3.pth")
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

# te.test("model6_3.pth")
# tr.check_dataset()

