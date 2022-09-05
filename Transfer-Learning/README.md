# Transfer-Learning

Notebook(s) on doing transfer learning, while I am learning

## Transfer-Learning

Training a really deep neural network is time consuming. To make it faster, we can use transfer learning.

In transfer learning, we start with a pretrained network, remove the last layer, and add our own layers on top of it. Then we freeze the weights of the pretrained network, and train only the newly added layers.

By doing this, the inner layers will extract features from the input data, and the outer layers will learn representations from the features.

After training the last layer, we can unfreeze other layers if needed and fine-tune the model.

Transfer learning is also really useful when we have a small dataset to train on.

**Fine-tuning a pre-trained model**: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning. In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.

