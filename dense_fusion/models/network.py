import cameramodels
import numpy as np
import numpy.ma as ma
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import quaternion_normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms

from dense_fusion.data import pose_net_pretrained_model_path
from dense_fusion.data import pose_refine_net_pretrained_model_path
from dense_fusion.models.pspnet import PSPNet

psp_models = {
    "resnet18": lambda: PSPNet(
        sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256,
        backend="resnet18"
    ),
}


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


class ModifiedResnet(nn.Module):

    def __init__(self):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models["resnet18"]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):

    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        # 128 + 256 + 1024
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)


class PoseNet(nn.Module):

    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        """Inference 6dof pose with confidence.

        Returns
        -------
        out_rx : torch.Tensor
            quaternion shape of (batch_size, self.num_points, 4)
        out_tx : torch.Tensor
            translation shape of (batch_size, self.num_points, 3)
        out_cx : torch.Tensor
            confidence shape of (batch_size, self.num_points, 1)
        """
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1,
                                                  self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()


class PoseRefineNetFeat(nn.Module):

    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x


class PoseRefineNet(nn.Module):

    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        """Inference 6dof pose.

        Returns
        -------
        out_rx : torch.Tensor
            quaternion shape of (batch_size, self.num_points, 4)
        out_tx : torch.Tensor
            translation shape of (batch_size, self.num_points, 3)
        """
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx


class DenseFusion(nn.Module):

    def __init__(self, num_points=1000, num_obj=21, dataset_type='ycb',
                 iteration=2):
        super(DenseFusion, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.iteration = iteration

        self.estimator = PoseNet(num_points=num_points, num_obj=num_obj)
        self.estimator.load_state_dict(
            torch.load(pose_net_pretrained_model_path(dataset=dataset_type)))

        self.refiner = PoseRefineNet(
            num_points=num_points,
            num_obj=num_obj)
        self.refiner.load_state_dict(
            torch.load(pose_refine_net_pretrained_model_path(
                dataset=dataset_type)))

    def predict(self, rgb_img, depth, label_img,
                label, bboxes, intrinsic_matrix):
        num_points = self.num_points
        iteration = self.iteration
        batch_size = 1
        lst = np.array(label.flatten(), dtype=np.int32)
        img_height, img_width, _ = rgb_img.shape

        cameramodel = cameramodels.PinholeCameraModel.from_intrinsic_matrix(
            intrinsic_matrix, img_height, img_width)

        translations = []
        rotations = []
        for idx in range(len(lst)):
            itemid = lst[idx]

            rmin, cmin, rmax, cmax = bboxes[idx]
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label_img, itemid))
            mask = mask_label * mask_depth
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                translations.append([])
                rotations.append([])
                continue

            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            xmap = np.array([[j for i in range(img_width)]
                             for j in range(img_height)])
            ymap = np.array([[i for i in range(img_width)]
                             for j in range(img_height)])

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][
                :, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][
                :, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][
                :, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cloud = cameramodel.batch_project_pixel_to_3d_ray(
                np.concatenate([ymap_masked, xmap_masked], axis=1),
                depth_masked)

            img_masked = np.array(rgb_img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            with torch.no_grad():
                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = normalize(
                    torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = cloud.cuda()
                choose = choose.cuda()
                img_masked = img_masked.cuda()
                index = index.cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3,
                                             img_masked.size()[1],
                                             img_masked.size()[2])

                pred_rot, pred_trans, pred_score, emb = self.estimator(
                    img_masked, cloud, choose, index)
                pred_rot = pred_rot / torch.norm(
                    pred_rot, dim=2).view(1, num_points, 1)

                pred_score = pred_score.view(batch_size, num_points)
                _, which_max = torch.max(pred_score, 1)
                pred_trans = pred_trans.view(batch_size * num_points, 1, 3)
                points = cloud.view(batch_size * num_points, 1, 3)
                rotation = pred_rot[0][which_max[0]].view(-1).cpu().\
                    data.numpy()
                translation = (points + pred_trans)[which_max[0]].view(-1).\
                    cpu().data.numpy()

                for _ in range(iteration):
                    T = torch.from_numpy(translation.astype(np.float32)).\
                             cuda().view(1, 3).\
                             repeat(num_points, 1).contiguous().\
                             view(1, num_points, 3)
                    trans_matrix = np.eye(4)
                    trans_matrix[:3, :3] = quaternion2matrix(
                        quaternion_normalize(rotation))
                    R = torch.from_numpy(trans_matrix[:3, :3].astype(
                        np.float32)).cuda().view(1, 3, 3)
                    trans_matrix[0:3, 3] = translation
                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    refined_rot, refined_trans = self.refiner(
                        new_cloud, emb, index)
                    refined_rot = refined_rot.view(1, 1, -1)
                    refined_rot = refined_rot / (torch.norm(
                        refined_rot, dim=2).view(1, 1, 1))
                    rotation_2 = refined_rot.view(-1).cpu().data.numpy()
                    translation_2 = refined_trans.view(-1).cpu().data.numpy()
                    trans_matrix_2 = np.eye(4)
                    trans_matrix_2[:3, :3] = quaternion2matrix(
                        quaternion_normalize(rotation_2))

                    trans_matrix_2[0:3, 3] = translation_2

                    trans_matrix_final = np.dot(trans_matrix, trans_matrix_2)
                    rotation_final = matrix2quaternion(
                        trans_matrix_final[:3, :3])
                    translation_final = np.array([trans_matrix_final[0][3],
                                                  trans_matrix_final[1][3],
                                                  trans_matrix_final[2][3]])

                    rotation = rotation_final
                    translation = translation_final
            translations.append(translation)
            rotations.append(quaternion_normalize(rotation))
        return rotations, translations
