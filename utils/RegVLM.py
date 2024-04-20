import torch
import copy


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.layer
    newModuleList = torch.nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel

def deleteLaterEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.layer
    newModuleList = torch.nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep, 0, -1):
        newModuleList.append(oldModuleList[-i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel

class RegVLM_Mixer(torch.nn.Module):
    def __init__(self, base_bert,
                 n_layers = 2,
                 n_visual_tokens = 196,
                 vision_embedding_dim = 384,
                 emb_dims = 512,
                 cls_token_id = 101):
        # prepare decoder
        super().__init__()
        self.cls_token_id = cls_token_id
        self.embedding_module = base_bert.embeddings
        self.n_visual_tokens = n_visual_tokens
        self.vision_emb_dim = vision_embedding_dim
        self.base_model = deleteLaterEncodingLayers(base_bert, n_layers).encoder
        
        self.pooler = torch.nn.AdaptiveAvgPool1d(1)
        self.emb_dimension = emb_dims
        
        if self.vision_emb_dim == self.emb_dimension:
            self.dimension_caster = torch.nn.Identity()
        else:
            self.dimension_caster = torch.nn.Linear(self.vision_emb_dim, self.emb_dimension, bias = False)  # no bias here
        
        
    def forward(self, vision_embedding, text_embedding, text_attn_mask):
        # assert len(vision_embedding) == len(text_embedding)
        n_batch = len(vision_embedding)
        
        cls_emb = self.embedding_module(torch.tensor([[self.cls_token_id]]*n_batch, 
                                                     device = vision_embedding.device),
                                        torch.tensor([[1]]*n_batch,
                                                     device = vision_embedding.device))
        
        # normalize dimensions
        new_vision_emb = self.dimension_caster(vision_embedding[:, 1:, :])   # remove cls token here
        
        # concatenate
        concatenated_emb = torch.cat([cls_emb, new_vision_emb, text_embedding], dim = 1)
        
        # create attention mask
        vision_attention_mask = torch.ones(n_batch, self.n_visual_tokens + 1).to(text_attn_mask.device) # add a cls token here
        attn_mask = torch.cat([vision_attention_mask, text_attn_mask], dim = 1)
        
        attn_mask = attn_mask[:, None, None, :]
        
        # forward
        return self.base_model(concatenated_emb, attn_mask)

class RegVLM(torch.nn.Module):
    def __init__(self,
                 vit,
                 bert,
                 vit_num_patches = 196,
                 vit_emb_dim = 384,
                 bert_emb_dim = 512,
                 bert_layers = 2,
                 vocab_size = 30522,
                 mask_token_id = 103,
                 cls_token_id = 101,
                 tau = None):
       super().__init__()
       self.vit = vit
       self.bert = bert.base_model
       self.bert = deleteEncodingLayers(self.bert.base_model, bert_layers)
       self.fusion = RegVLM_Mixer(bert.base_model,
                              n_layers = bert_layers,
                              n_visual_tokens=vit_num_patches,
                              vision_embedding_dim=vit_emb_dim,
                              emb_dims = bert_emb_dim,
                              cls_token_id = cls_token_id)
       
       # vit patches data
       self.vit_num_patches = vit_num_patches
       
       # vocab size
       self.vocab_size = vocab_size
       # mask token
       self.mask_token_id = mask_token_id
       
       # learnable temperature parameter
       self.tau = torch.nn.Parameter(torch.FloatTensor([0.07]))      # uniform in range 1 to 5
       if tau is not None:
           self.tau = torch.nn.Parameter(torch.FloatTensor([tau]))      # uniform in range 1 to 5
       
       self.tau.requires_grad = True

       # joint representation
       self.pooler = torch.nn.Sequential(
           torch.nn.AdaptiveAvgPool1d(1),
           torch.nn.Flatten()
       )
       self.img_proj = torch.nn.Linear(vit_emb_dim, min(vit_emb_dim, bert_emb_dim))
       self.txt_proj = torch.nn.Linear(bert_emb_dim, min(vit_emb_dim, bert_emb_dim))

       
       # masked representation modeling
       self.mrm_proj = torch.nn.Sequential(
            torch.nn.Linear(bert_emb_dim, bert_emb_dim),
            torch.nn.Tanh(),
       )
       
       # head for masked image modeling
       self.mim_proj = torch.nn.Sequential(
           torch.nn.Linear(bert_emb_dim, vit_emb_dim),
       )
        
       # head for masked language modeling
       self.mlm_head = bert.cls
       
       self.itc_head = torch.nn.Linear(bert_emb_dim, bert_emb_dim)
       
       self.itm__head = torch.nn.Sequential(
           torch.nn.Linear(bert_emb_dim, bert_emb_dim),
           torch.nn.LeakyReLU(),
           torch.nn.Linear(bert_emb_dim, 1)
       )
    
       
       
    def forward(self, image, text, attn_mask,
                masked_image = None,
                masked_text = None,
                image_text_matching = False,
                retrieval = False,
                ):
        
        if retrieval is True:
            img_rep = self.vit(image)['last_hidden_state']
            txt_rep = self.bert(text, attn_mask)['last_hidden_state']
            joint_rep = self.fusion(img_rep, txt_rep, attn_mask)['last_hidden_state']
            
            
            # img_rep = self.img_proj(self.pooler(img_rep.transpose(1,2)))
            # txt_rep = self.txt_proj(self.pooler(txt_rep.transpose(1,2)))
            return img_rep, txt_rep, joint_rep, self.itm__head(joint_rep[:, 0, :])
        
        if image_text_matching == True:
            img_rep = self.vit(image)['last_hidden_state']
            txt_rep = self.bert(text, attn_mask)['last_hidden_state']
            joint_rep = self.fusion(img_rep, txt_rep, attn_mask)['last_hidden_state']
            
            return img_rep, txt_rep, joint_rep, self.itm__head(joint_rep[:, 0, :])
        
        else:
            # return mask_img-clean_txt, clean_img,-mask_txt, 
            img_rep = self.vit(image)['last_hidden_state']              # clean image
            txt_rep = self.bert(text, attn_mask)['last_hidden_state']   # clean text
            
            mask_img_rep = self.vit(masked_image)['last_hidden_state']
            mask_txt_rep = self.bert(masked_text, attn_mask)['last_hidden_state']
            
            # multimodal prediction
            c_img_m_txt = self.fusion(img_rep, mask_txt_rep, attn_mask)['last_hidden_state']
            m_img_c_txt = self.fusion(mask_img_rep, txt_rep, attn_mask)['last_hidden_state']
            
            # pure txt
            txt_prediction = self.mlm_head(mask_txt_rep)
            
            # pure fusion
            img_txt_joint = self.fusion(img_rep, txt_rep, attn_mask)['last_hidden_state']
        
            return (c_img_m_txt, m_img_c_txt, img_txt_joint, mask_img_rep, txt_prediction, img_rep, txt_rep)
    
    def get_mrm_loss(self, online_representation, target_representation, mask):
        # remove cls token
        on_rep = torch.nn.functional.normalize(self.mrm_proj(online_representation[:, 1:, :]), dim = -1)
        tr_rep = torch.nn.functional.normalize(self.mrm_proj(target_representation[:, 1:, :]), dim = -1)

        
        loss = torch.nn.functional.mse_loss(on_rep, tr_rep, reduction = 'none')
        mrm_loss = (loss * mask).sum()/(mask.sum() + 1e-5)              # add for 0 division errors
        return mrm_loss
    
    def get_mim_loss(self, online_representation, target_representation, mask):
        on_rep = self.mim_proj(online_representation[:, 1:self.vit_num_patches+1, :]) # omit cls token
        tr_rep = target_representation[:, 1:self.vit_num_patches+1, :]
        
        # # normalize
        on_rep = torch.nn.functional.normalize(on_rep, dim = 2)
        tr_rep = torch.nn.functional.normalize(tr_rep, dim = 2)
        
        loss = torch.nn.functional.l1_loss(on_rep, tr_rep, reduction = 'none')
        if mask.ndim == 2:
            mask = mask[:, :, None]
        mim_loss = (loss * mask).sum() / (mask.sum() + 1e-5)
        return mim_loss 
    
    
    def get_mlm_loss(self, scores, sen, masked_sen):
        labels = torch.where(masked_sen == self.mask_token_id, sen, -100)
        loss = torch.nn.functional.cross_entropy(scores.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        
        return loss
    
    
    def get_itc_loss(self, img_feats, txt_feats):
        # Calculate similarity
        with torch.no_grad():
            self.tau.clamp_(0.001,0.5)

        # pool and flatten text and visual features obtained before fusion
        img_feats = self.img_proj(self.pooler(img_feats.transpose(1,2)))
        txt_feats = self.txt_proj(self.pooler(txt_feats.transpose(1,2)))
        
        
        sim = (img_feats@txt_feats.T)/self.tau
        # sim = torch.clip(sim, max = 1e4, min = 1e-4)
        self_mask = torch.eye(sim.shape[0], device=sim.device)
        
        loss_i2t = -torch.sum(torch.nn.functional.log_softmax(sim, dim = 1)*self_mask, dim = 1).mean()
        loss_t2i = -torch.sum(torch.nn.functional.log_softmax(sim.T, dim = 1)*self_mask, dim = 1).mean()

        return sim, (loss_i2t+loss_t2i)/2.0
    
    def get_samples(self, similarities):
        probs = torch.nn.functional.softmax(similarities, dim = 1)
        probs = probs.fill_diagonal_(0)         # eliminate full samples
        txt_indices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
        img_indices = torch.multinomial(probs.T, num_samples=1, replacement=True).squeeze(1)
        
        return txt_indices, img_indices