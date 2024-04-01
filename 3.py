import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions

image_paths = ['dog.jpg', 'car.jpg', 'cat.jpg']
images = []
for path in image_paths:
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    images.append(image)
preprocessed_images = preprocess_input(np.array(images))
model = VGG16(include_top=True, weights='imagenet')
predictions = model.predict(preprocessed_images)
decoded_predictions = decode_predictions(predictions)
for i, pred in enumerate(decoded_predictions):
    print("Image:", image_paths[i])
    for j, (imagenet_id, label, score) in enumerate(pred):
        print(f"{j + 1}: {label} ({score:.2f})")
