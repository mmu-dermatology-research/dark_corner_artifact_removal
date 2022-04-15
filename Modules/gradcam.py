# -*- coding: utf-8 -*-
"""This class module was originally created by Adrian Rosebrock on pyimagesearch.

It is a class to generate a gradCAM image to display the activations of a CNN
for image classification tasks.

The original webpage source can be found here:
    
    https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
"""

# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    """GradCAM class used to generate gradcam heatmap
    
    Methods
    -------
    __init__
        constructor
    find_target_layer
        extract final conv layer name from model
    compute_heatmap
        generate heatmap
    overlay_heatmap
        merge the heatmap with the original image
    
    """
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        #colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    


def main():
    """Main method used only to test the module
    
    """

    # good examples - InceptionResNetV2
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0072646_oth.jpg"
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0073047_oth.jpg"
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0072259_oth.jpg"
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0060252_oth.jpg"
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/mel/ISIC2019_0057747_mel.jpg"
    # * "D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/mel/ISIC2019_0058513_mel.jpg"

    #base    
    #func_model = tf.keras.models.load_model("D:/OneDrive/Desktop/Lesion_Classification/Models/Baseline/InceptionResNetV2/SGD/64/InceptionResNetV2_batchSize_0_opt_SGD_model.20.h5")   
    #image = cv2.imread("D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0073047_oth.jpg")

    #ns
    func_model = tf.keras.models.load_model("D:/OneDrive/Desktop/Lesion_Classification/Models/Inpaint_NS/InceptionResNetV2/SGD/64/InceptionResNetV2_batchSize_0_opt_SGD_model.19.h5")
    image = cv2.imread("D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224_inpainted_ns/val/oth/ISIC2019_0073047_oth.png")


    #image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    
    preds = func_model.predict(image) 
    i = np.argmax(preds[0])
    
    #for idx in range(len(func_model.layers)):
        #  print(func_model.get_layer(index = idx).name)
        
    print(i)
    print(preds)
        
    icam = GradCAM(func_model, i, 'conv_7b_ac') 
    heatmap = icam.compute_heatmap(image)


    #base
    #image = cv2.imread("D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224/val/oth/ISIC2019_0073047_oth.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #ns
    image = cv2.imread("D:/OneDrive/Desktop/Lesion_Classification/Data/train_balanced_224x224_inpainted_ns/val/oth/ISIC2019_0073047_oth.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(heatmap.shape, image.shape)



    (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

    #plt.imshow(output)

    output2 = np.vstack([image, heatmap, output])
    plt.imshow(output2)

if __name__ == '__main__':
    main()


