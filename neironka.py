import json
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array
import os


classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_',
           'Corn_(maize)___healthy',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)',
           'Grape___healthy',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach___healthy',
           'Pepper,_bell___Bacterial_spot',
           'Pepper,_bell___healthy',
           'Potato___Early_blight',
           'Potato___healthy',
           'Potato___Late_blight',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___healthy',
           'Strawberry___Leaf_scorch',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___healthy',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


def load_model(json_file='model.json', h5_file="model.h5"):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_file)
    # print("Loaded model from disk")
    return loaded_model


def make_prediction(loaded_model, formatted_image):
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                         metrics=['accuracy'])
    score = loaded_model.predict(formatted_image)
    return score


def format_image(image_name):
    test_image = Image.open(image_name)
    test_image = test_image.resize((224, 224), Image.LANCZOS)
    test_image = img_to_array(test_image) / 255
    image_batch = np.expand_dims(test_image, axis=0)
    return image_batch


def main():
    directory = input()
    images = os.listdir(directory)
    print(images)
    loaded_model = load_model()
    for image_name in images:
        formatted_image = format_image(directory + "/" + image_name)
        result = make_prediction(loaded_model, formatted_image)
        print("-----------------------")
        print(image_name)
        print(np.argmax(result))
        print(classes[np.argmax(result)])
        print("-----------------------")


if __name__ == "__main__":
    main()

    # import json
# from PIL import Image
# import numpy as np
# from keras.models import model_from_json
# from keras.utils import img_to_array
# import os


# classes = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Blueberry___healthy',
#  'Cherry_(including_sour)___healthy',
#  'Cherry_(including_sour)___Powdery_mildew',
#  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn_(maize)___Common_rust_',
#  'Corn_(maize)___healthy',
#  'Corn_(maize)___Northern_Leaf_Blight',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___healthy',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___healthy',
#  'Potato___Late_blight',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___healthy',
#  'Strawberry___Leaf_scorch',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___healthy',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# def load_model(json_file='model.json', h5_file="model.h5"):
#     json_file = open(json_file, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(h5_file)
#     print("Loaded model from disk")
#     return loaded_model

# def make_prediction(loaded_model, formatted_image):
#     # evaluate loaded model on test data
#     loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
# metrics=['accuracy'])
#     score = loaded_model.predict(formatted_image)
#     return score

# def format_image(image_name):
#     test_image = Image.open(image_name)
#     test_image = test_image.resize((224, 224), Image.ANTIALIAS)
#     test_image = img_to_array(test_image) / 255
#     print(test_image.shape)
#     image_batch = np.expand_dims(test_image, axis = 0)
#     return image_batch


# def main():
#     image_name = input()  # name of the file
#     loaded_model = load_model()
#     formatted_image = format_image(image_name)
#     result= make_prediction(loaded_model, formatted_image)
#     print(np.argmax(result))
#     print(classes[np.argmax(result)])


# if __name__ == "__main__":
#     main()
