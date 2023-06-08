"""
1. load the txt file in datapath
2. read the txt file
3. split line the txt file
"""
import os
import cv2
import pickle
data_path = '../VISOL/train/'
data_list = os.listdir(data_path)

label_name_list = [file for file in data_list if file.endswith(".txt")]

data_list = []

# split line the txt file
for label_name in label_name_list:
    label = open(data_path + label_name, 'r')
    label = label.read().splitlines()
    for index, instance in enumerate(label):
        a = list(map(float, instance.split(' ')))
        # 경로, class, x, y, x, y
        data_list.append([data_path + label_name.split('.')[0]+'.png', a[0], a[1], a[2], a[5], a[6], int(index)])

for index in range(len(data_list)):
    img_path = data_list[index][0]
    #print(img_path)
    img = cv2.imread(img_path)
    # crop image. data_list[index][2] is left up x, data_list[index][3] is left up y, data_list[index][4] is right down x, data_list[index][5] is right down y
    # I'll use the half of image. crop (x, cy, x+w, cy+h)
    #ori = img.copy()
    img = img[int((data_list[index][3]+data_list[index][5])/2):int(data_list[index][5]), int(data_list[index][2]):int(data_list[index][4])]
    #ori = ori[int(data_list[index][3]):int(data_list[index][5]), int(data_list[index][2]):int(data_list[index][4])]
    # resize image to 224x224
    img = cv2.resize(img, (224, 224))
    # save image in TRAIN_DATA
    #cv2.imwrite('TRAIN_DATA/' + img_path.split('/')[-1], img)
    #cv2.imwrite('TRAIN_DATA/' + img_path.split('/')[-1].split('.')[0] + '_ori.png', ori)
    # pickle the image and label in TRAIN_DATA

    data = {'image' : img, 'label' : data_list[index][1]}

    with open('TRAIN_DATA/' + img_path.split('/')[-1].split('.')[0] + '_' + str(data_list[index][-1]) + '.pkl', 'wb') as f:
        pickle.dump(data, f)