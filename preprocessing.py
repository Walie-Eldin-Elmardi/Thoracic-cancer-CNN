import cv2 as cv
import numpy as np
import os

def resize(image, size=(400,400)):
    # img = cv.imread(image)
    # display_images(image, cv.resize(image, size))
    return cv.resize(image, size)

def gray_scale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def contrast(image, limit):
    contrast = cv.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    return contrast.apply(image)

def normalize(image):
    processed = image.astype(np.float64) / 255.0
    # processed = (processed * 255).astype(np.uint8)
    return processed

def extract_edges(image):
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image = cv.imread(image)
    edges = cv.Canny(image, 10,60)
    return edges

def texture_analysis(image, l=32):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = (gray.astype(np.float64) * l / 256).astype(np.uint8)
    glcm = np.zeros((l,l), dtype=np.float64)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1] - 1):
            i = gray[y, x]
            j = gray[y, x + 1]
            glcm[i, j] += 1

    glcm = glcm / np.sum(glcm)

    #still gotta calculate the features from the glcm matrix
    lvl = glcm.shape[0]

    contrast = 0
    homogeneity = 0 
    energy = 0
    correlation = 0

    meanx = sum(i * np.sum(glcm[i, :]) for i in range(lvl))
    meany = sum(j * np.sum(glcm[:, j]) for j in range(lvl))

    stdx = np.sqrt(sum((i - meanx) ** 2 * np.sum(glcm[i, :]) for i in range(lvl)))
    stdy = np.sqrt(sum((j - meany) ** 2 * np.sum(glcm[:, j]) for j in range(lvl)))

    for i in range(lvl):
        for j in range(lvl):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
            energy += glcm[i, j] ** 2
            if stdx > 0 and stdy > 0:
                correlation += ((i - meanx) * (j - meany) * glcm[i, j]) / (stdx * stdy)
    if stdx * stdy != 0:
        correlation /= (stdx * stdy)
    else:
        correlation = 0
    energy = np.sqrt(energy)
    return contrast, homogeneity, energy, correlation

def preprocess_image(image_path):
    # Load the image
    image = cv.imread(image_path)
    #display the original and processed images
    display_images(image, normalize(gray_scale(resize(image))))

    contrast, homogeneity, energy, correlation = texture_analysis(image)
    print("contrast:", contrast)
    print("homogeneity:", homogeneity)
    print("energy:", energy)
    print("correlation:", correlation)

def generate_dataset(dir, new_dir):
    for filename in os.listdir(dir):
        image_path = os.path.join(dir, filename)
        image = cv.imread(image_path)
        if image is not None:
            output = os.path.join(new_dir, filename)
            cv.imwrite(output, contrast(gray_scale(resize(image)), 3.0))

def display_images(original, processed):
    if len(original.shape) == 2:
        original = cv.cvtColor(original, cv.COLOR_GRAY2BGR)
    if len(processed.shape) == 2:
        processed = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
    combined = np.hstack((original, processed))
    cv.imshow("Original and Processed Images", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

def morphological_operations(image):
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    # dilated = cv.dilate(image, kernel, iterations=2)
    eroded = cv.erode(image, kernel, iterations=1)
    return eroded
    # preprocess_image(eroded)
    # display_images(eroded, extract_edges(eroded))
    # display_images(dilated, extract_edges(dilated))