import numpy as np
import tkinter as tk
import tkinter.font as font
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2 as cv
import os
import time
import json



def read_faces(name):
    cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    offset = 0
    count = 0
    cap = cv.VideoCapture(0)
    os.mkdir(os.path.join("faces",name))
    while cap.isOpened():
        __ , frame = cap.read()
        if __ is True:
            co_oridi = cascade.detectMultiScale(frame,1.3,5)
            if len(co_oridi) == 0:
                continue
            for x,y,w,h in co_oridi:
                short_img = frame[y-offset:y+h+offset,x-offset:x+w+offset]
                short_img = cv.resize(short_img,dsize=(127,127),interpolation=cv.INTER_CUBIC)
                cv.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,127),thickness=1)
                # print("short_img is detected")
                if count > 300:
                    print("finished_collecting the data...pls press q ")
                    break
                path = os.path.join("faces",name,f"{count}.png")
                print(path)
                cv.imwrite(path , short_img)
                count = count + 1
                time.sleep(0.3)
            cv.imshow("frame",frame)
            if cv.waitKey(20) & 0xFF == 27:
                print("intruped ")
                break
    else:
        print("error in asseeing ....web cam")
        cap.release()
        cv.destroyAllWindows()



def train_tf_model(x , y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train.shape , y_train.shape)
    print(x_test.shape,y_test.shape)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(130,activation="relu"),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(50,activation="relu"),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(15,activation="relu"),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(1,activation="sigmoid"),
    ])
    model.compile(optimizer="adam",loss = tf.keras.losses.binary_crossentropy,metrics=["accuracy"])
    model.fit(x_train,y_train,epochs=50)
    model.save("face_recog.h5")
    model.evaluate(x_test,y_test)


def get_data_from_faces():
    index = 0
    map_name_index = {}
    map_index_name = {}
    map_name_listPics = {}
    path = "./faces"
    for name in os.listdir(path):
        map_index_name[index] = name
        map_name_index[name] = index
        index = index + 1 
        temp_list = []
        for pic in os.listdir(os.path.join("faces",name)):
            img = cv.imread(os.path.join("faces",name,pic))
            img = img / 255
            temp_list.append(img)
        map_name_listPics[name] = temp_list
    # print(map_index_name)
    # print(map_name_index)
    # print(len(map_name_listPics["suma"]))
    x_train = []
    y_train = []
    for name , index in map_name_index.items():
        for pic in map_name_listPics[name]:
            x_train.append(pic)
            y_train.append(index)
    # print(x_train[0])
    # print(y_train[0])
    x_np = np.array(x_train)
    y_np = np.array(y_train)
    map_name_index_json = "map_name_index.json"
    with open(map_name_index_json, "w") as file:
        json.dump(map_name_index, file)
    map_index_name_json = "map_index_name.json"
    with open(map_index_name_json, "w") as file:
        json.dump(map_index_name, file)
    train_tf_model(x=x_np,y=y_np)
    
    

        
        
def identify_faces():
    map_name_index_json = "map_name_index.json"
    map_index_name_json = "map_index_name.json"
    model = tf.keras.models.load_model("face_recog.h5")
    with open(map_name_index_json, "r") as file:
        map_name_index = json.load(file)
    with open(map_index_name_json, "r") as file:
        map_index_name = json.load(file)
    print(map_index_name)
    print(map_name_index)
    cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    offset = 0
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        __ , frame = cap.read()
        if __ is True:
            co_oridi = cascade.detectMultiScale(frame,1.3,5)
            if len(co_oridi) == 0:
                continue
            for x,y,w,h in co_oridi:
                short_img = frame[y-offset:y+h+offset,x-offset:x+w+offset]
                short_img = cv.resize(short_img,dsize=(127,127),interpolation=cv.INTER_CUBIC)
                short_img = short_img/255
                short_img = np.reshape(short_img,(1,short_img.shape[0],short_img.shape[1],short_img.shape[2]))
                predict_num = model.predict(short_img)
                print(predict_num)
                predict_num = np.reshape(predict_num,(predict_num.shape[0])).round()
                print(predict_num)
                print(predict_num[0])
                print(int(predict_num[0]))
                name = map_index_name[str(int(predict_num[0]))]
                print(name)
                cv.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,127),thickness=1)
                cv.putText(frame,text=name,org=(x,y-10),fontFace=cv.FONT_ITALIC,fontScale=.8,color=(0,255,0),thickness=2)
            cv.imshow("frame",frame)
            if cv.waitKey(20) & 0xFF == 27:
                print("intruped ")
                break
    else:
        print("error in asseeing ....web cam")
    cap.release()
    cv.destroyAllWindows()


def new_user():
    name = input("enter the new user name  : ")
    read_faces(name)
    get_data_from_faces()
    print("your data is succefully collected")

def old_user():
    identify_faces()

# new_user()


# get_data_from_faces()
# identify_faces()


# def collect_data():
# 	name = input("Enter name of person : ")

# 	count = 1
# 	ids = input("Enter ID: ")

# 	cap = cv2.VideoCapture(0)

# 	filename = "haarcascade_frontalface_default.xml"

# 	cascade = cv2.CascadeClassifier(filename)

# 	while True:
# 		_, frm = cap.read()
# 		gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

# 		faces = cascade.detectMultiScale(gray, 1.4, 1)

# 		for x,y,w,h in faces:
# 			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 2)
# 			roi = gray[y:y+h, x:x+w]

# 			cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
# 			count = count + 1
# 			cv2.putText(frm, f"{count}", (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
# 			cv2.imshow("new", roi)


# 		cv2.imshow("identify", frm)

# 		if cv2.waitKey(1) == 27 or count > 300:
# 			cv2.destroyAllWindows()
# 			cap.release()
# 			train()
# 			break

# def train():
# 	print("training part initiated !")

# 	recog = cv2.face.LBPHFaceRecognizer_create()

# 	dataset = 'persons'

# 	paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

# 	faces = []
# 	ids = []
# 	labels = []
# 	for path in paths:
# 		labels.append(path.split('/')[-1].split('-')[0])

# 		ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))

# 		faces.append(cv2.imread(path, 0))
# 	print("before start training")
# 	recog.train(faces, np.array(ids))
# 	print("after the model is trained")

# 	recog.save('model.yml')

# 	return

# def identify():
# 	cap = cv2.VideoCapture(0)

# 	filename = "haarcascade_frontalface_default.xml"

# 	paths = [os.path.join("persons", im) for im in os.listdir("persons")]
# 	labelslist = {}
# 	for path in paths:
# 		labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

# 	print(labelslist)
# 	recog = cv2.face.LBPHFaceRecognizer_create()

# 	recog.read('model.yml')

# 	cascade = cv2.CascadeClassifier(filename)

# 	while True:
# 		_, frm = cap.read()

# 		gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

# 		faces = cascade.detectMultiScale(gray, 1.3, 2)

# 		for x,y,w,h in faces:
# 			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 2)
# 			roi = gray[y:y+h, x:x+w]

# 			label = recog.predict(roi)

# 			if label[1] < 100:
# 				cv2.putText(frm, f"{labelslist[str(label[0])]} + {int(label[1])}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
# 			else:
# 				cv2.putText(frm, "unkown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

# 		cv2.imshow("identify", frm)

# 		if cv2.waitKey(1) == 27:
# 			cv2.destroyAllWindows()
# 			cap.release()
# 			break



def maincall():


	root = tk.Tk()

	root.geometry("480x100")
	root.title("identify")

	label = tk.Label(root, text="Select below buttons ")
	label.grid(row=0, columnspan=2)
	label_font = font.Font(size=35, weight='bold',family='Helvetica')
	label['font'] = label_font

	btn_font = font.Font(size=25)

	button1 = tk.Button(root, text="Add Member ", command=new_user, height=2, width=20)
	button1.grid(row=1, column=0, pady=(10,10), padx=(5,5))
	button1['font'] = btn_font

	button2 = tk.Button(root, text="Start with known ", command=old_user, height=2, width=20)
	button2.grid(row=1, column=1,pady=(10,10), padx=(5,5))
	button2['font'] = btn_font
	root.mainloop()

	return


