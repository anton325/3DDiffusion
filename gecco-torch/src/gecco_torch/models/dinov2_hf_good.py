import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

from PIL import Image
import numpy as np

# Load the image and convert to the appropriate format
image = Image.open("gt.png")

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]
        # len(outputs.hidden_states) is 13, all of shape batch, 785, 384, just like the last hidden state
        print(patch_embeddings.shape) # (batch_size, num_patches, hidden_size) 
        # convert to logits and upsample to the size of the pixel values
    
model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-small")
model(torch.tensor(np.array(image)).reshape(4,400,400)[:3].unsqueeze(0), output_hidden_states = False)


