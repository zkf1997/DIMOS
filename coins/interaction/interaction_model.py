import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG
from configuration.config import *

# https://github.com/Mathux/ACTOR/blob/d3b0afe674e01fa2b65c89784816c3435df0a9a5/src/models/architectures/transformer.py#L7
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    # input: BxSxD
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

#NERF Positional encoding
# https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dimension=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dimension,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class NormalDistDecoder(nn.Module):
    """
    Linear layers to map input feature to latent normal distribution
    """
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()
        self.num_feat_in = num_feat_in

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        Xout = Xout.reshape(-1, self.num_feat_in)
        # return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))
        return torch.distributions.normal.Normal(self.mu(Xout), torch.exp(0.5 * self.logvar(Xout).clamp(min=-10, max=10)))

def padded_idx_to_code(verb_ids):
    """
    Get one-hot codes from verb indices
    """
    codebook = torch.cat([torch.eye(num_verb, dtype=torch.float32, device=verb_ids.device),
               torch.zeros((1, num_verb), dtype=torch.float32, device=verb_ids.device)], dim=0)
    return codebook[verb_ids.long()]

# https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_sem.py
class PointNet2Encoder(nn.Module):
    """
    c_in: input point feature dimension exculding xyz
    """
    def __init__(self, c_in=6, c_out=128, num_keypoints=256):
        super(PointNet2Encoder, self).__init__()
        self.c_in = c_in
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_keypoints,  # 256
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=True,
            )
        )
        c_out_1 = 128 + 128

        self.num_keypoints = num_keypoints
        self.c_out = c_out
        self.Linear = nn.Linear(c_out_1, c_out - 3)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:3 + self.c_in].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        B, I, P, C = pointcloud.shape
        pointcloud = pointcloud.reshape(B*I, P, C)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # print(l_xyz[-1].shape, l_features[-1].shape)
        # TODO: add lrelu to see whether fix nan in training
        local_keypoints = torch.cat((l_xyz[-1],
                                     torch.nn.LeakyReLU()(self.Linear(l_features[-1].transpose(1, 2)))), dim=-1)  # B*I x Pb x C,
        return local_keypoints.reshape(B, I, self.num_keypoints, self.c_out)

class BodyEmbedding(nn.Module):
    """
    Get embedding for body tokens
    """
    def __init__(self, input_dimension, output_dimension,
                 use_nerf_embed, multires=10):
        super(BodyEmbedding, self).__init__()
        self.use_nerf_embed = use_nerf_embed
        if use_nerf_embed:
            self.nerf_embed, self.nerf_embed_dimension = get_embedder(multires=multires, input_dimension=input_dimension)
        self.linear = nn.Linear(self.nerf_embed_dimension, output_dimension) if use_nerf_embed else nn.Linear(input_dimension, output_dimension)

    def forward(self, body_points):
        B, P, C = body_points.shape
        if self.use_nerf_embed:
            body_points = self.nerf_embed(body_points)
        body_points = self.linear(body_points)
        return body_points

class InteractionVAE(nn.Module):
    """
    Conditional VAE network for both PelvisVAE and BodyVAE.
    """
    def __init__(self, args):
        super(InteractionVAE, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        if not hasattr(args, 'nerf_embed'):
            args.nerf_embed = 0
            args.multires = 0

        self.bodyEmbedding = BodyEmbedding(input_dimension=args.dim_body_points, output_dimension=num_channels,
                                           use_nerf_embed=args.nerf_embed, multires=args.multires, )
        self.objEmbedding = PointNet2Encoder(c_in=args.dim_point_feature, c_out=num_channels, num_keypoints=args.num_obj_keypoints) if args.use_pointnet2 else nn.Linear(9, num_channels)
        self.interactionEmbedding = nn.Parameter(torch.randn(num_verb, num_channels))
        self.positionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)

        from transformer import TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=num_channels,
                                                          nhead=self.args.num_heads,
                                                          dim_feedforward=self.args.ff_size,
                                                          dropout=self.args.dropout,
                                                          activation=self.args.activation,
                                                          batch_first=True)
        self.encoder = TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.args.num_layers)

        self.latentNormal = NormalDistDecoder(num_channels, args.latent_dim)
        self.interactionBias = nn.Parameter(torch.randn(num_verb, args.latent_dim))
        self.latentEmbedding = nn.Linear(args.latent_dim, num_channels)

        # choose to use the latent code as memory or add it to template body tokens
        if self.args.latent_usage == 'memory':
            seqTransDecoderLayer = TransformerDecoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=True)
            self.decoder = TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.args.num_layers)
        else:
            self.decoder = TransformerEncoder(TransformerEncoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=True),
                                                 num_layers=self.args.num_layers)
        self.finalLinear = nn.Linear(num_channels, args.dim_body_points)

    def _get_embeddings(self, x=None, batch=None):
        """
        Get embeddings for tokens of input body, input action-object pairs, and template body.

        Return:
            verb_codes: one-hot codes for verbs
            padding_mask: padding mask for tokens,
            body_embedding: embeddings of input body tokens
            obj_embedding: embeddings of object tokens
            template_embedding: embeddings of template body tokens
        """
        num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
        B, I, _, _ = obj_points.shape
        Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
        padding_mask = (verb_ids == -1).repeat_interleave(Po).reshape((B, Po * 2))  # BxPo*2
        padding_mask = torch.cat([torch.zeros((B, Pb), dtype=torch.bool, device=obj_points.device),
                                  padding_mask], dim=1)
        verb_codes = padded_idx_to_code(verb_ids)  # Bx2xVerb
        verb_embedding = torch.matmul(verb_codes, self.interactionEmbedding)  # Bx2xD
        interaction_embedding = (verb_embedding.sum(dim=1) / num_atomics.unsqueeze(1)).unsqueeze(1)

        body_embedding = None
        if x is not None:
            x = x.reshape(B, Pb, -1)
            body_embedding = self.bodyEmbedding(x)  # BxPbx3 -> BxPbxD
            body_embedding = self.positionalEmbedding(body_embedding)
            if self.args.interaction_bias:
                body_embedding = body_embedding + interaction_embedding

        obj_embedding = self.objEmbedding(obj_points)  # Bx2xPoxD
        # print(obj_embedding.shape)
        obj_embedding = obj_embedding + verb_embedding.unsqueeze(2)
        obj_embedding = obj_embedding.reshape(B, I * Po, D)

        if self.args.template_type == 'tpose':
            template_embedding = self.bodyEmbedding(
            torch.tensor(self.args.template_body, device=obj_points.device).unsqueeze(0).expand(B, -1, -1)
        )
        elif self.args.template_type == 'personal':
            template_embedding = self.bodyEmbedding(batch['template_body'])
        else:
            template_embedding = torch.zeros((B, Pb, D), device=obj_points.device)
        if self.args.mask_body and self.training:  #randomly mask some body joints
            body_mask = (torch.randn(B, Pb) < self.args.mask_prob).unsqueeze(2).expand(B, Pb, D).float().to(template_embedding.device)
            template_embedding = template_embedding * (1 - body_mask) + 0.01 * body_mask
        template_embedding = self.positionalEmbedding(template_embedding)
        if self.args.interaction_bias:
            template_embedding = template_embedding + interaction_embedding

        return verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding

    def _encode(self, body_embedding, obj_embedding, padding_mask, Pb):
        """
        Encode the input embeddings to latent distribtuion.
        """
        encoder_input = torch.cat([body_embedding, obj_embedding], dim=1)  # BxPb+2PoxD
        feature = self.encoder(encoder_input, src_key_padding_mask=padding_mask)[0][:, :Pb, :].mean(dim=1)
        z_dist = self.latentNormal(feature)  # BxD
        return z_dist

    def _decode(self, verb_codes, padding_mask, obj_embedding, template_embedding, z_sample, Pb, num_atomics, attention_mask=None):
        """
        Decode latent code to body/pelvis prediction.
        """
        if self.args.interaction_bias:
            interaction_bias = torch.matmul(verb_codes, self.interactionBias).sum(dim=1) / num_atomics.unsqueeze(1)
            z_sample = z_sample + interaction_bias
        z_sample = self.latentEmbedding(z_sample).unsqueeze(1)  # Bx1xD

        # choose to use the latent code as memory or add it to template body tokens
        if self.args.latent_usage == 'memory':
            decoder_input = torch.cat([template_embedding, obj_embedding], dim=1)  # BxPb+2*PoxD
            decoder_output = self.decoder(tgt=decoder_input, memory=z_sample, tgt_key_padding_mask=padding_mask, tgt_mask=attention_mask)
            pred, attention, _ = decoder_output
        else:
            decoder_input = torch.cat([template_embedding + z_sample, obj_embedding], dim=1)  # BxPb+2*PoxD
            decoder_output = self.decoder(decoder_input, src_key_padding_mask=padding_mask, src_mask=attention_mask)
            pred, attention = decoder_output

        pred = self.finalLinear(pred[:, :Pb, :])  # BxPbx3
        if self.args.use_contact_feature:
            pred = torch.cat((pred[:, :, :-2], torch.sigmoid(pred[:, :, -2:])), dim=-1)
        return pred, attention

    def forward(self, x, batch):
        num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
        Pb = self.args.num_body_points
        verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self._get_embeddings(x=x, batch=batch)

        z_dist = self._encode(body_embedding, obj_embedding, padding_mask, Pb)
        z_sample = z_dist.rsample()
        pred, _ = self._decode(verb_codes, padding_mask, obj_embedding, template_embedding, z_sample, Pb, num_atomics)

        return pred, z_dist

    def sample(self, batch, composition_mask=False):
        """
        Sample body/pelvis.
        """
        set_eval = self.training
        if set_eval:
            self.eval()

        with torch.no_grad():
            num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
            B, I, _, _ = obj_points.shape
            Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
            verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self._get_embeddings(x=None, batch=batch)

            z_sample = torch.distributions.normal.Normal(
                loc=torch.zeros((B, self.args.latent_dim), requires_grad=False, device=obj_points.device),
                scale=torch.ones((B, self.args.latent_dim), requires_grad=False,
                                 device=obj_points.device)).rsample()
            pred, attention = self._decode(verb_codes, padding_mask, obj_embedding, template_embedding, z_sample, Pb, num_atomics,
                                           attention_mask=self.get_composition_mask(composition_mask).to(obj_points.device) if composition_mask else None)

        if set_eval:
            self.train()
        return pred, attention

    def decode(self, batch, z_sample, composition_mask=False):
        """
        Sample body/pelvis using inputted latent codes.
        """
        self.eval()

        num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
        B, I, _, _ = obj_points.shape
        Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
        verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self._get_embeddings(x=None, batch=batch)

        if torch.is_tensor(composition_mask):
            attention_mask = composition_mask.to(obj_points.device)
        else:
            attention_mask = self.get_composition_mask(composition_mask).to(obj_points.device) if composition_mask else None
        pred, attention = self._decode(verb_codes, padding_mask, obj_embedding, template_embedding, z_sample, Pb, num_atomics,
                                       attention_mask=attention_mask)
        return pred, attention

    def get_composition_mask(self, composition_mask='diagonal'):
        """
        Get body composition masks.
        """
        if composition_mask == 'manual':
            if hasattr(self, 'manual_mask'):
                return self.manual_mask
            Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
            composition_mask = torch.zeros((Pb + Po * 2, Pb + Po * 2), dtype=torch.bool)
            # split body by pelvis, pelvis is not masked
            if self.args.body_type == 'joint':
                upper_body, lower_body = [3, 6, 9] + list(range(12, 22)), [1, 2, 4, 5, 7, 8, 10, 11]
            elif self.args.body_type == 'mesh':
                upper_body, lower_body = self.args.body_segment

            composition_mask[upper_body, Pb:Pb+Po] = True
            # composition_mask[Pb:Pb + Po, upper_body] = True
            composition_mask[lower_body, Pb+Po:] = True
            # composition_mask[Pb + Po:, lower_body] = True
            composition_mask[Pb:Pb + Po, Pb + Po:] = True
            composition_mask[Pb + Po:, Pb:Pb + Po] = True
            self.manual_mask = composition_mask
            print('built manual composition mask')
            return self.manual_mask
        elif composition_mask == 'diagonal':
            if hasattr(self, 'diagonal_mask'):
                return self.diagonal_mask
            Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
            composition_mask = torch.zeros((Pb + Po * 2, Pb + Po * 2), dtype=torch.bool)
            composition_mask[Pb:Pb + Po, Pb + Po:] = True
            composition_mask[Pb + Po:, Pb:Pb + Po] = True
            self.diagonal_mask = composition_mask
            print('built diagonal composition mask')
            return self.diagonal_mask
        else:  # learned composition directly passed in
            return composition_mask


    def sample_composition(self, batch):
        """
        sample composite interactions, suppose all inputs have two atomic interactions
        use attention mask to limit upper body to attend to 'touch' related object, lower body to the other object
        """
        self.eval()

        with torch.no_grad():
            num_atomics, obj_points, verb_ids = batch['num_atomics'], batch['object_pointclouds'], batch['verb_ids']
            B, I, _, _ = obj_points.shape
            Po, Pb, D = self.args.num_obj_keypoints, self.args.num_body_points, self.args.embedding_dim
            verb_codes, padding_mask, body_embedding, obj_embedding, template_embedding = self._get_embeddings(x=None, batch=batch)

            z_sample = torch.distributions.normal.Normal(
                loc=torch.zeros((B, self.args.latent_dim), requires_grad=False, device=obj_points.device),
                scale=torch.ones((B, self.args.latent_dim), requires_grad=False,
                                 device=obj_points.device)).rsample()
            pred, attention = self._decode(verb_codes, padding_mask, obj_embedding, template_embedding, z_sample, Pb, num_atomics,
                                           attention_mask=self.get_composition_mask().to(obj_points.device))

        return pred, attention