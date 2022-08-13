import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import webbrowser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    angrycounter,avg_angry_systolic,avg_angry_diastolic,avg_angry_heartbeat=0,0,0,0
    disgustedcounter,avg_disg_systolic,avg_disg_diastolic,avg_disg_heartbeat=0,0,0,0
    fearfulcounter,avg_fear_systolic,avg_fear_diastolic,avg_fear_heartbeat=0,0,0,0
    happycounter,avg_happy_systolic,avg_happy_diastolic,avg_happy_heartbeat=0,0,0,0
    neutralcounter,avg_neutral_systolic,avg_neutral_diastolic,avg_neutral_heartbeat=0,0,0,0
    sadcounter,avg_sad_systolic,avg_sad_diastolic,avg_sad_heartbeat=0,0,0,0
    surprisedcounter,avg_surp_systolic,avg_surp_diastolic,avg_surp_heartbeat=0,0,0,0
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES,15)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            xyz=emotion_dict[maxindex]
            if xyz=="Angry" and angrycounter>=1:
                angrycounter=angrycounter+1
            if xyz == "Angry" and angrycounter==0:
                url = "https://www.allmusic.com/mood/soothing-xa0000001099"
                url1="https://agoodmovietowatch.com/mood/feel-good/"
                angrycounter=angrycounter+1
                angry_systolic=[80,75,85,95,100]
                avg_angry_systolic=sum(angry_systolic)/len(angry_systolic)
                angry_diastolic=[120,125,130,129,131]
                avg_angry_diastolic=sum(angry_diastolic)/len(angry_diastolic)
                angry_heartrate=[65,80,75,90,100]
                avg_angry_heartbeat=sum(angry_heartrate)/len(angry_heartrate)
                webbrowser.open(url);
                webbrowser.open(url1);
            if xyz=="Disgusted" and disgustedcounter>=1:
                disgustedcounter=disgustedcounter+1
            if xyz == "Disgusted" and disgustedcounter==0:
                url = "https://www.allmusic.com/mood/thoughtful-xa0000001121"
                url1= "https://agoodmovietowatch.com/mood/heart-warming/"
                disgustedcounter=disgustedcounter+1
                disg_systolic=[75,85,90,91,94]
                avg_disg_systolic=sum(disg_systolic)/len(disg_systolic)
                disg_diastolic=[115,125,120,129,131]
                avg_disg_diastolic=sum(disg_diastolic)/len(disg_diastolic)
                disg_heartrate=[90,91,92,83,84]
                avg_disg_heartbeat=sum(disg_heartrate)/len(disg_heartrate)
                webbrowser.open(url);
                webbrowser.open(url1);
            if xyz=="Fearful" and fearfulcounter>=1:
                fearfulcounter=fearfulcounter+1
            if xyz == "Fearful" and fearfulcounter==0:
                url = "https://www.allmusic.com/mood/angry-xa0000000695"
                fearfulcounter=fearfulcounter+1
                fear_systolic=[88,89,90,95,105]
                avg_fear_systolic=sum(fear_systolic)/len(fear_systolic)
                fear_diastolic=[120,130,131,129,131]
                avg_fear_diastolic=sum(fear_diastolic)/len(fear_diastolic)
                fear_heartrate=[90,95,100,105,106]
                avg_fear_heartbeat=sum(fear_heartrate)/len(fear_heartrate)
                webbrowser.open(url);
            if xyz=="Happy" and happycounter>=1:
                happycounter=happycounter+1
            if xyz == "Happy" and happycounter==0:
                url = "https://www.allmusic.com/mood/happy-xa0000001016"
                happycounter=happycounter+1
                happy_systolic=[75,85,90,95,90]
                avg_happy_systolic=sum(happy_systolic)/len(happy_systolic)
                happy_diastolic=[120,121,122,121,120]
                avg_happy_diastolic=sum(happy_diastolic)/len(happy_diastolic)
                happy_heartrate=[90,95,92,95,90]
                avg_happy_heartbeat=sum(happy_heartrate)/len(happy_heartrate)
                webbrowser.open(url);
            if xyz=="Neutral" and neutralcounter>=1:
                neutralcounter=neutralcounter+1
            if xyz == "Neutral" and neutralcounter==0:
                url = "https://www.allmusic.com/mood/gentle-xa0000001008"
                neutralcounter=neutralcounter+1
                neutral_systolic=[80,85,80,95,80]
                avg_neutral_systolic=sum(neutral_systolic)/len(neutral_systolic)
                neutral_diastolic=[120,120,127,129,121]
                avg_neutral_diastolic=sum(neutral_diastolic)/len(neutral_diastolic)
                neutral_heartrate=[90,95,90,90,90]
                avg_neutral_heartbeat=sum(neutral_heartrate)/len(neutral_heartrate)
                webbrowser.open(url);
            if xyz=="Sad" and sadcounter>=1:
                sadcounter=sadcounter+1
            if xyz == "Sad" and sadcounter==0:
                url = "https://www.allmusic.com/mood/sad-xa0000000761"
                url1= "https://agoodmovietowatch.com/mood/uplifting/"
                sadcounter=sadcounter+1
                sad_systolic=[80,85,85,85,80]
                avg_sad_systolic=sum(sad_systolic)/len(sad_systolic)
                sad_diastolic=[120,115,112,111,110]
                avg_sad_diastolic=sum(sad_diastolic)/len(sad_diastolic)
                sad_heartrate=[90,88,86,88,86]
                avg_sad_heartbeat=sum(sad_heartrate)/len(sad_heartrate)
                webbrowser.open(url);
                webbrowser.open(url1);
            if xyz=="Surprised" and surprisedcounter>=1:
                surprisedcounter=surprisedcounter+1
            if xyz == "Surprised" and surprisedcounter==0:
                url = "https://www.allmusic.com/mood/exciting-xa0000000994"
                surprisedcounter=surprisedcounter+1
                surp_systolic=[80,85,85,95,100]
                avg_surp_systolic=sum(surp_systolic)/len(surp_systolic)
                surp_diastolic=[120,125,137,129,131]
                avg_surp_diastolic=sum(surp_diastolic)/len(surp_diastolic)
                surp_heartrate=[90,95,98,93,99]
                avg_surp_heartbeat=sum(surp_heartrate)/len(surp_heartrate)
                webbrowser.open(url);
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    systolic_emotion_values=[avg_angry_systolic,avg_disg_systolic,avg_fear_systolic,avg_happy_systolic,avg_neutral_systolic,avg_sad_systolic,avg_surp_systolic]
    diastolic_emotion_values=[avg_angry_diastolic,avg_disg_diastolic,avg_fear_diastolic,avg_happy_diastolic,avg_neutral_diastolic,avg_sad_diastolic,avg_surp_diastolic]
    heartbeat_emotion_values=[avg_angry_heartbeat,avg_disg_heartbeat,avg_fear_heartbeat,avg_happy_heartbeat,avg_neutral_heartbeat,avg_sad_heartbeat,avg_surp_heartbeat]
    count_emotion_values=[angrycounter/15,disgustedcounter/15,fearfulcounter/15,happycounter/15,neutralcounter/15,sadcounter/15,surprisedcounter/15]
    #print(count_emotion_values)
    x_axis_emotions=["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x_axis_emotions, systolic_emotion_values, color ='maroon',width = 0.4)
    plt.xlabel("Emotions")
    plt.ylabel("Diastolic Bloood Pressure")
    plt.title("Diastolic Blood Pressure in different mooods")
    plt.show()
    plt.bar(x_axis_emotions, diastolic_emotion_values, color ='maroon',width = 0.4)
    plt.xlabel("Emotions")
    plt.ylabel("Systolic Bloood Pressure")
    plt.title("Systolic Blood Pressure in different mooods")
    plt.show()
    plt.bar(x_axis_emotions, heartbeat_emotion_values, color ='maroon',width = 0.4)
    plt.xlabel("Emotions")
    plt.ylabel("Heartrate")
    plt.title("Heartrate in different mooods")
    plt.show()
    plt.bar(x_axis_emotions, count_emotion_values, color ='maroon',width = 0.4)
    plt.xlabel("Emotions")
    plt.ylabel("Time in seconds")
    plt.title("Time duration of different moods")
    plt.show()
