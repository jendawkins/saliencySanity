This repository 1) trains a pre-trained Inception model on custom images and 2) computes saliency methods for regression for these images primarily using 2 different packages

This is a work in progress

This work is based on https://arxiv.org/pdf/1810.03292.pdf, but applying these methods to images that are medical in nature (from the Bone Age dataset) and classified using regression 

This repository uses saliency methods from https://github.com/lightdogs/pytorch-smoothgrad and https://github.com/utkuozbulak/pytorch-cnn-visualizations and adapts these methods to the bone age dataset (and regression instead of classification)
