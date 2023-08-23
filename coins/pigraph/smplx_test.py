import smplx
import torch

from pigraph_config import *

body_model = smplx.create(smplx_model_folder, model_type='smplx',
                              gender='male', ext='npz')
torch_param = {}
torch_param['global_orient'] = torch.randn([1, 3], dtype=torch.float32)
torch_param['transl'] = torch.randn([1, 3], dtype=torch.float32)
torch_param['body_pose'] = torch.randn([1, 63], dtype=torch.float32)

torch_param['betas'] = torch.randn([1, body_model.num_betas], dtype=torch.float32)
torch_param['expression'] = torch.randn([1, body_model.num_expression_coeffs], dtype=torch.float32)
smplx_output = body_model(return_verts=True, **torch_param)
print(smplx_output.joints.detach().cpu().numpy())