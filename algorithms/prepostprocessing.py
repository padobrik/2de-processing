import cv2
import numpy as np
import torch
import os


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image)

def postprocess_prediction(prediction):
    prediction = prediction.detach().numpy()
    prediction = np.argmax(prediction, axis=1)
    prediction = np.squeeze(prediction, axis=0).astype(np.uint8)
    return prediction

if __name__ == "__main__":
    # Load the image
    image_path = "path/to/your/image.png"
    image_tensor = preprocess_image(image_path)

    # Create a U-Net model
    model = UNet(in_channels=1, out_channels=2)

    # Load the pre-trained model weights
    model_weights_path = "path/to/your/model_weights.pth"
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
    else:
        print("Error: Model weights not found. Train the model and save the weights.")

    # Perform contour detection
    with torch.no_grad():
        output = model(image_tensor)

    # Post-process the output
    prediction = postprocess_prediction(output)

    # Display the results
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(original_image, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imshow("Original Image", resized_image)
    cv2.imshow("Contour Detection", prediction * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output
    output_path = "path/to/save/output.png"
    cv2.imwrite(output_path, prediction * 255)
