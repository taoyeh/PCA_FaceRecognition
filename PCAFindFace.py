import numpy as np
import cv2 as cv
import os
import face_recognition
import time

# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# 协方差矩阵
def Cov(dataMat):
    meanVal = np.mean(data, 0)  # 压缩行，返回1*cols矩阵，对各列求均值
    meanVal = np.tile(meanVal, (rows, 1))  # 返回rows行的均值矩阵
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov


# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    print("保留99%信息，降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
    # 重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat

# 将所有的图片PCA降维
def FindPCAImage():
    imagePath = ''
    tmp = 'Images/'
    for i in range(1, 66):
        imagePath = tmp + str(i) + '.jpg'
        print(imagePath)
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rows, cols = image.shape
        reconImage = PCA(image, 0.99)
        reconImage = reconImage.astype(np.uint8)
        cv.imwrite('PCAImages/'+str(i)+'.jpg', reconImage)

# 修改源代码以供自己需求
def compare_faces(known_face_encodings, face_encoding_to_check):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check))
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
def main():
    # imagePath = 'PCAImages/'
    # image_file_name = os.path.join(imagePath, '48.jpg')
    # print('正在抽取以下图片特征',image_file_name)
    # image_file = face_recognition.load_image_file(image_file_name)
    # face_encoding = face_recognition.face_encodings(image_file)
    # print(face_encoding)
    # FindPCAImage()
    user_face_encodings = []

    imagePath = 'PCAImages/'
    for i in range(1, 66):
        # 读取面部特征向量
        # PCAImages/1.jpg
        image_file_name = os.path.join(imagePath, str(i)+'.jpg')
        print('正在抽取以下图片特征',image_file_name)
        image_file = face_recognition.load_image_file(image_file_name)
        face_encoding = face_recognition.face_encodings(image_file)[0]
        user_face_encodings.append(face_encoding)
    image_file = face_recognition.load_image_file('Anchor.jpg')
    Anchor_face_encoding=face_recognition.face_encodings(image_file)[0]

    matchs =compare_faces(user_face_encodings, Anchor_face_encoding)
    Minimum=0
    SecondMinimum=0
    for i in range(len(matchs)):
        if  matchs[i]<matchs[Minimum]:
            Minimum=i
    for i in range(len(matchs)):
        if matchs[i]<matchs[SecondMinimum] and matchs[i]!=matchs[Minimum]:
            SecondMinimum=i
    print(matchs)
    print('距离最近的是第{}张图片' .format(Minimum+1))
    print('距离第二近是第{}张图片'.format(SecondMinimum+1))

if __name__ == '__main__':
    main()





