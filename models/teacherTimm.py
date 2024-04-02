import timm
import torch.nn as nn

class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[0,1, 2, 3,4]
    ):
        super(teacherTimm, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        self.modelName = backbone_name
        self.feature_extractor.eval() 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False   
        self.pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        features_t = self.feature_extractor(x)
        features_t[0]=self.pool(features_t[0])
        
        
        return features_t