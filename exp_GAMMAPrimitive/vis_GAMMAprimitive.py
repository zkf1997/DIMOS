import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx
import cv2
import pickle
import pdb
import re
import glob

sys.path.append(os.getcwd())
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
from exp_GAMMAPrimitive.utils.vislib import *



def visualize(data, data_bparams, datagt=None, gender='male', betas=0,
                outfile_path=None, datatype='gt',seq=0,gen=0,
                show_body=True):
    ## prepare data
    n_frames = data.shape[0]

    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=10
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()

    ### top lighting
    box = o3d.geometry.TriangleMesh.create_box(width=200, depth=1,height=200)
    box.translate(np.array([-200,-200,6]))
    vis.add_geometry(box)
    vis.poll_events()
    vis.update_renderer()

    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh from data
    ball_list = []
    n_markers = int(re.split('_|\.',
                    os.path.basename(results_file_name))[2])

    for i in range(n_markers):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)

    if datagt is not None:
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        pcd.points = o3d.utility.Vector3dVector(datagt[-1])
        pcd.paint_uniform_color(color_hex2rgb('#dce04a')) # "I want hue"
        vis.update_geometry(pcd)

    if show_body:
        body = o3d.geometry.TriangleMesh()
        vis.add_geometry(body)
        vis.poll_events()
        vis.update_renderer()

    ## parse smplx parameters
    bm = get_body_model('smplx', gender, n_frames, device='cpu')
    bparam = {}
    bparam['transl'] = data_bparams[:,:3]
    bparam['global_orient'] = data_bparams[:,3:6]
    bparam['betas'] = np.tile(betas[None,...], (n_frames,1))
    bparam['body_pose'] = data_bparams[:,6:69]
    bparam['left_hand_pose'] = data_bparams[:,69:81]
    bparam['right_hand_pose'] = data_bparams[:,81:]


    ## obtain body mesh sequences
    for key in bparam:
        bparam[key] = torch.FloatTensor(bparam[key])
    verts_seq = bm(return_verts=True, **bparam).vertices.detach().cpu().numpy() #[t,verts, 3]
    jts = bm(return_verts=True, **bparam).joints[:,:22].detach().cpu().numpy() #[t,J, 3]
    # ## from amass coord to world coord
    # verts_seq = np.einsum('ij, tvj->tvi', transf_rotmat, verts_seq)+transf_transl[None,...]
    # if datagt is not None:
    #     datagt = np.einsum('ij, tvj->tvi', transf_rotmat, datagt)+transf_transl[None,...]
    # jts = np.einsum('ij, tvj->tvi', transf_rotmat, jts)+transf_transl[None,...]


    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            b.translate(data[it,i], relative=False)
            vis.update_geometry(b)

        if it <motion_seed_len:
            for ball in ball_list:
                ball.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
        else:
            if datatype == 'gt':
                pass
            else:
                for ball in ball_list:
                    ball.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"
        if show_body:
            ## set body mesh locations
            body.vertices = o3d.utility.Vector3dVector(verts_seq[it])
            body.triangles = o3d.utility.Vector3iVector(bm.faces)
            body.vertex_normals = o3d.utility.Vector3dVector([])
            body.triangle_normals = o3d.utility.Vector3dVector([])
            body.compute_vertex_normals()
            vis.update_geometry(body)

            if it <motion_seed_len:
                body.paint_uniform_color(color_hex2rgb('#c2dd97')) # "I want hue"
            else:
                body.paint_uniform_color(color_hex2rgb('#c7624f')) # "I want hue"

        # ## special colors on head (for debug)
        # ball_list[22].paint_uniform_color(color_hex2rgb('#5874ae'))
        # ball_list[7].paint_uniform_color(color_hex2rgb('#9ebf5e'))
        # ball_list[3].paint_uniform_color(color_hex2rgb('#b25d48'))
        # ball_list[18].paint_uniform_color(color_hex2rgb('#749a83'))

        # if it in [0,15,30,45,59]:
        #     # o3d.visualization.draw_geometries([limb_lines]+ball_list)
        #     # o3d.io.write_line_set('tmp_seq20_gen0_lineset_frame35.ply', limb_lines)
        #     for i, b in enumerate(ball_list):
        #         o3d.io.write_triangle_mesh('tmp_seq{}_gen{}_kps{}_frame{}.ply'.format(seq,gen,i,it), b)

        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        cam_t = body_t + 2.0*np.ones(3)
        ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:05d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)



if __name__=='__main__':
    proj_path = os.getcwd()
    exps = [
            # 'exp_GAMMAPrimitive/MPVAE_1frame_v4',
            # 'exp_GAMMAPrimitive/MPVAE_samp_rollout',
            'exp_GAMMAPrimitive/MPVAE_babel_1frame_noise',
            ]
    n_seq_vis = 10
    n_gen_vis = 3

    for show_body in [True]:
        out_suffix='body' if show_body else 'kps'
        for exp in exps:
            res_file_list = sorted(glob.glob(proj_path+'/results/{}/results/mp_gen_seed0/*/results_ssm2_67*.pkl'.format(exp)))
            for results_file_name in res_file_list:
                print('-- processing: '+results_file_name)
                with open(results_file_name, 'rb') as f:
                    data = pickle.load(f)
                dd = data['markers'][:,:,:,:67,:]
                ddgt = data['gt'][:,:,:,:67,:]
                dd_bparam = data['smplx_params']
                n_seq=dd.shape[0]
                n_gen=dd.shape[1]
                motion_seed_len = 1
                for seq in range(n_seq_vis):
                    gender = data['gender'][seq]
                    betas = data['betas'][seq]
                    for gen in range(n_gen_vis):
                        renderfolder = results_file_name+'_render{}'.format(out_suffix)+'/seq{}_gen{}'.format(seq, gen)
                        if not os.path.exists(renderfolder):
                            os.makedirs(renderfolder)
                        visualize(dd[seq, gen],
                                dd_bparam[seq, gen],
                                # datagt=ddgt[seq, 0], # change this one to none will not visualize the targer marker.
                                datagt = None,
                                gender=gender,
                                betas=betas,
                                outfile_path=renderfolder, datatype='kps',
                                seq=seq,gen=gen,
                                show_body=show_body)
