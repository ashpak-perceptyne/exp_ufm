import torch
import torch.nn as nn
import torch.nn.functional as F

class hybrid_model(nn.Module): 
    def __init__(self, prob_layer: list): 

        super(hybrid_model, self).__init__() 
        
        self.prob_layer = prob_layer
        

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        

        self.dinov2.eval()
        for param in self.dinov2.parameters():
            param.requires_grad = False

        self.probs = nn.ModuleDict({
            str(layer_id) : nn.Linear(1024 ,1024 , bias = False)
            for layer_id in self.prob_layer
        })

        

    def forward(self, img1, img2): 

        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)


        img_1_ft = self.dinov2.get_intermediate_layers(
            x = img1, 
            n = self.prob_layer, 
            return_class_token = False 
        )

        img_2_ft = self.dinov2.get_intermediate_layers(
            x = img2, 
            n = self.prob_layer, 
            return_class_token = False 
        )


        similarity_matrix = {}

        for i , layer_id in enumerate(self.prob_layer) : 
            img1_feat = img_1_ft[i] 
            img2_feat = img_2_ft[i] 

            projected_img_feat_1=self.prob[str(layer_id)](img1_feat)
            projected_img_feat_2=self.prob[str(layer_id)](img2_feat)
        

            dot = torch.bmm(projected_img_feat_1, projected_img_feat_2.transpose(1, 2)) 
        

            similarity_matrix[layer_id] = dot

        return similarity_matrix

