import torch
import torch.nn as nn
import torch.nn.functional as F
from routing_files import capsule_layers
from routing_files.multimodal_routing import MultimodalRouting

class CapsModel(nn.Module):
    def __init__(self,
                 act_type = 'ONES', 
                 num_routing = 2,
                 dp = .5,
                 layer_norm = False,
                 t_in_dim = 768,
                 i_in_dim = 768,
                 ti_in_dim = 1024,
                 pc_dim = 64,
                 mc_caps_dim = 64,
                 dim_pose_to_vote = 100):
        
        super().__init__()

        self.act_type = act_type
        self.num_routing = num_routing
        self.pc_dim = pc_dim

        # -----------------------------
        # PRIMARY CAPSULE PROJECTIONS
        # -----------------------------
        self.pc_text  = nn.Linear(t_in_dim, pc_dim + 1)
        self.pc_image = nn.Linear(i_in_dim, pc_dim + 1)
        
        # outer product of pc_text image and pc_image
        """
        self.pc_text_image = nn.Sequential(
            nn.Linear(pc_dim**2, (pc_dim**2 // 2)),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear((pc_dim**2 // 2), pc_dim + 1)

        )
        """
        # -----------------------------
        # NUMBER OF DECISION CAPSULES
        # -----------------------------
        self.mc_num_caps = 4

        self.mc_caps_dim = mc_caps_dim

        # -----------------------------
        # DECISION CAPSULE LAYER
        # -----------------------------
        self.mc = capsule_layers.CapsuleFC(
            in_n_capsules=3,                  # ONLY text + image + t_i
            in_d_capsules=pc_dim,
            out_n_capsules=self.mc_num_caps,
            out_d_capsules=mc_caps_dim,
            n_rank=None,
            dp=dp,
            act_type=act_type,
            small_std=not layer_norm,
            dim_pose_to_vote=dim_pose_to_vote
        )
        self.routing = MultimodalRouting(num_concepts=self.mc_num_caps, num_iters=self.num_routing)
        self.W_proj = nn.Parameter(torch.randn(3, pc_dim, self.mc_num_caps, mc_caps_dim))

        # Output embedding (class prototypes)
        self.embedding = nn.Parameter(torch.zeros(self.mc_num_caps, mc_caps_dim))

        # Final bias for logits
        self.bias = nn.Parameter(torch.zeros(self.mc_num_caps))

        #self.classifier = nn.Linear(self.mc_num_caps, 1)
        self.classifier = nn.Identity()

    def forward(self, image, text, text_image):
        """
        text:  (B, T, t_in_dim)
        image: (B, I, i_in_dim)

        → both should already be extracted features (e.g., CLIP embeddings)
        """

        # -----------------------------
        # PRIMARY CAPSULES
        # -----------------------------
        u_t = self.pc_text(text).unsqueeze(1)      # (B, 1, pc_dim+1)
        u_i = self.pc_image(image).unsqueeze(1)    # (B, 1, pc_dim+1)
        #

        text_image = torch.einsum("btd, bid -> btid", u_t[:, :, :self.pc_dim], u_i[:, :, :self.pc_dim])
        text_image = text_image.flatten(start_dim=-2)
        u_tv = self.pc_text_image(text_image).unsqueeze(1)

        # Combine into (B, 3, pc_dim+1)
        pc_input = torch.cat([u_t, u_i, u_tv], dim=1)

        # Pose + Activation
        init_pose = pc_input[:, :, :self.pc_dim]
        init_act  = torch.sigmoid(pc_input[:, :, self.pc_dim:])
        """
        # -----------------------------
        # ROUTING
        # -----------------------------
        pose, act, _ = self.mc(init_pose, init_act, 0)
        
        for r in range(self.num_routing):
            pose, act, routing_coeff = self.mc(init_pose, init_act, r, pose, act)

        # -----------------------------
        # CLASS LOGITS
        # -----------------------------
        pose = pose / (pose.norm(dim=-1, keepdim=True) + 1e-6)
        embedding = self.embedding / (self.embedding.norm(dim=-1, keepdim=True) + 1e-6)
        concept_logits = torch.einsum("bcd,cd -> bc", pose, embedding) + self.bias
        
        logits = self.classifier(concept_logits) 
        """

        concepts, routing_coeff = self.routing(init_pose, init_act.squeeze(-1), self.W_proj)
        concept_logits = torch.einsum("bcd,cd -> bc", concepts, self.embedding)
        logits = self.classifier(concept_logits) 

        return logits, init_act.squeeze(-1), routing_coeff, concept_logits
        
