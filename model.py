import mnist
from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape((-1,28*28))
x_test = x_test.reshape((-1,28*28))


x_train = (x_train/256)
x_test = (x_test/256)

clf = MLPClassifier(solver='adam', activation="relu",hidden_layer_sizes=(64,64))
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
acc = confusion_matrix(y_test,predictions)

def accuracy(confusion_matrix):
    diagonal = confusion_matrix.trace()
    elements = confusion_matrix.sum()
    return diagonal/elements

print(accuracy(acc))

img = Image.open('3.png')

img_gray = img.convert('L')
img_resized = img_gray.resize((28, 28))


pixel_data = np.array(img_resized)

grayscale = pixel_data.flatten() / 256 

mean_value = grayscale.mean()

if mean_value > 0.5:
    grayscale = 1 - grayscale
else:
    grayscale = grayscale

p = clf.predict([grayscale])
print(f"Predicted digit: {p[0]}")
