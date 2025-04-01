# Traffic Sign Classification with CNN

## Experimentation Process
For this project, I explored different CNN architectures to classify traffic signs accurately.

1. **Baseline Model**: Initially, I tried a single convolutional layer followed by a dense layer, but it underperformed.
2. **Adding More Layers**: I added a second convolutional layer and max-pooling layer, which improved feature extraction.
3. **Dropout Regularization**: To reduce overfitting, I included a dropout layer (0.5).
4. **Optimizers & Loss Functions**: I experimented with `SGD` and `Adam`. Adam provided faster convergence.
5. **Final Model**: The final model consists of:
   - 2 Convolutional layers (32, 64 filters)
   - MaxPooling after each convolution
   - Fully connected layer with ReLU
   - Dropout (0.5) to prevent overfitting
   - Softmax output layer for classification

## Results
- The model achieved **high accuracy** on training data and generalizes well on unseen data.
- Possible improvements: Data augmentation and batch normalization.

## References
- OpenCV Documentation
- TensorFlow Keras Guide
- Harvard’s CS50 AI Course Materials
