import argparse
import torch
from PIL import Image
from config.config import ModelConfig
from src.dataset.transforms import ImageTransform, TextTransform
from src.models.clip_model import CLIPModel

def main():
    """ example inference script for CLIP model """
    parser = argparse.ArgumentParser(description='Inference of the model')
    parser.add_argument('--checkpoint', type=str, required=True, help='path of the model checkpoint')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--texts', type=str, nargs='+', required=True)
    args = parser.parse_args()

    model_config = ModelConfig()
    image_transform = ImageTransform(size=model_config.image_size, is_training=False)
    text_transform = TextTransform(max_length=model_config.max_text_length)
    model = CLIPModel(model_config)

    # load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # process image
    image = Image.open(args.image).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0)

    # process texts
    text_tensors = []
    for text in args.texts:
        text_tensor = text_transform(text)
        text_tensors.append(text_transform(text))
    text_tensors = torch.stack(text_tensors) # use stack to create a batch (a new dimension)

    # inference
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensors)

        # compute similarity
        similarities = torch.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0),
            dim=-1
        ).unsqueeze(0)

        sorted_indices = torch.argsort(similarities)

    print(f"\n Image: {args.image}")
    print("Sorted Texts by Similarity:")
    for i, idx in enumerate(sorted_indices):
        text = args.texts[idx]
        similarity = similarities[idx].item()
        print(f"{i+1}. {text} (similarity: {similarity:.4f})")

if __name__ == '__main__':
    main()
    # Example usage:
    # python scripts /inference.py \
    #     --checkpoint choeckpoints/epoch_29.pt \
    #     --image /scratch/sc232jl/rocov2/test_images/test/ROCOv2_2023_test_000001 \
    #     --texts "CT chest axial view showing a huge ascending aortic aneurysm (*)." "normal chest" "cardiomegaly present"

