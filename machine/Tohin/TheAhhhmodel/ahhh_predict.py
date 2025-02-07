import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ahh_colab_architechture import TohinNeuralNet

# Load the trained model
model_path = "models/ahhhhmodel.pth"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    model = torch.load(model_path, map_location=device)
    if isinstance(model, torch.nn.Module):
        print("Successfully loaded full model.")
    else:
        raise Exception("Loaded object is not a model.")
except Exception as e:
    print(f"Failed to load model directly: {e}")
    try:
        print("Attempting to load state_dict instead...")
        state_dict = torch.load(model_path, map_location=device)
        model = TohinNeuralNet()  # Create a new instance of the model
        model.load_state_dict(state_dict)
        model.to(device)
        print("Successfully loaded model from state_dict.")
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # Resize to match the input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Function to make predictions
def predict(image_path, model, transform):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.size}")
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(f"Input tensor shape: {image_tensor.shape}")

    # Make predictions
    with torch.no_grad():
        class_probs, bbox_pred = model(image_tensor)
        
    print("Raw model outputs:")
    print(f"Class probabilities shape: {class_probs.shape}")
    print(f"Class probabilities: {class_probs}")
    print(f"Bounding box predictions shape: {bbox_pred.shape}")
    print(f"Bounding box predictions: {bbox_pred}")

    # Convert predictions to bounding boxes and class labels
    class_probs = class_probs.squeeze(0)  # Remove batch dimension
    bbox_pred = bbox_pred.squeeze(0)      # Remove batch dimension

    # Get the predicted class labels (0: housebot, 1: robot)
    predicted_classes = torch.argmax(class_probs, dim=-1)
    print(f"Predicted classes: {predicted_classes}")

    # Convert bounding box predictions to [x_min, y_min, x_max, y_max] format
    bbox_pred = bbox_pred * 600  # Scale back to image size (600x600)
    print(f"Scaled bbox predictions: {bbox_pred}")

    return predicted_classes, bbox_pred

# Function to visualize the predictions
def visualize_predictions(image_path, predicted_classes, bbox_pred):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    print(f"Visualization image size: {image.size}")
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    print(f"Number of predictions to visualize: {len(predicted_classes)}")
    
    # Draw bounding boxes and labels
    for i in range(len(predicted_classes)):
        class_label = "Robot" if predicted_classes[i] == 1 else "Housebot"
        bbox = bbox_pred[i].numpy()
        x_min, y_min, x_max, y_max = bbox
        
        print(f"Drawing box {i+1}:")
        print(f"Class: {class_label}")
        print(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min - 5, class_label, color='r', fontsize=12, backgroundcolor='white')

    plt.show()

# Example usage
IMG_PATH = "2242_png.rf.e3285eb24831c586a0bfa92f70110a6e.jpg"
IMG_PATH1 = "data/NHRL/test/images/1416_png.rf.e96e59268b7fd89b02f25fc75ca41635.jpg"
IMG_PATH2 = "data/NHRL/train/images/0_png.rf.232a178fe443e307f95e69d6d0330fc3.jpg"
ROTATED_PATH = "rotated1.jpg"
IMG = "test.jpg"

print("\nStarting prediction process...")
predicted_classes, bbox_pred = predict(IMG, model, transform)
print("\nStarting visualization process...")
visualize_predictions(IMG_PATH, predicted_classes, bbox_pred)