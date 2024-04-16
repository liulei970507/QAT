import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr.admin import loading
from ltr import model_constructor
import pdb

class QA_layer(nn.Module):
    def __init__(self, input_channel): # 输入特征算权重
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel//8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel//8, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map):
        ms_weight = self.gap(feature_map).view(feature_map.size(0), -1)
        ms_weight = self.fc(ms_weight).unsqueeze(-1).unsqueeze(-1)
        return ms_weight

class DiMPnet_RGBT(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor_v, feature_extractor_i, rgbt01_SFT_scale_layer2, rgbt01_SFT_shift_layer2,
    rgbt01_SFT_scale_layer3, rgbt01_SFT_shift_layer3, rgbt10_SFT_scale_layer2, rgbt10_SFT_shift_layer2,
    rgbt10_SFT_scale_layer3, rgbt10_SFT_shift_layer3, qa_layer3, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()

        self.feature_extractor_v = feature_extractor_v
        self.feature_extractor_i = feature_extractor_i
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.rgbt01_SFT_scale_layer2=rgbt01_SFT_scale_layer2
        self.rgbt01_SFT_shift_layer2=rgbt01_SFT_shift_layer2
        self.rgbt01_SFT_scale_layer3=rgbt01_SFT_scale_layer3
        self.rgbt01_SFT_shift_layer3=rgbt01_SFT_shift_layer3
        self.rgbt10_SFT_scale_layer2=rgbt10_SFT_scale_layer2
        self.rgbt10_SFT_shift_layer2=rgbt10_SFT_shift_layer2
        self.rgbt10_SFT_scale_layer3=rgbt10_SFT_scale_layer3
        self.rgbt10_SFT_shift_layer3=rgbt10_SFT_shift_layer3
        # self.qa_layer2 = qa_layer2
        self.qa_layer3 = qa_layer3
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W). # 6 20 3 352 352
            test_imgs:  Test image samples (images, sequences, 3, H, W). # 6 20 3 352 352
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4). # 3 20 4
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module. # 3 20 4
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""
        # print('train_imgs.shape, test_imgs.shape, train_bb.shape, test_proposals.shape',train_imgs.shape, test_imgs.shape, train_bb.shape, test_proposals.shape)
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features for RGB and TIR
        train_imgs_v = train_imgs[:3,...]
        test_imgs_v = test_imgs[:3,...]
        train_imgs_i = train_imgs[3:,...]
        test_imgs_i = test_imgs[3:,...]
        # pdb.set_trace()
        train_feat_v_ori = train_feat_v = self.extract_backbone_features_v(train_imgs_v.reshape(-1, *train_imgs_v.shape[-3:]))# layer2 512 layer3 1024
        test_feat_v_ori = test_feat_v = self.extract_backbone_features_v(test_imgs_v.reshape(-1, *test_imgs_v.shape[-3:]))
        # pdb.set_trace()
        train_feat_i_ori = train_feat_i = self.extract_backbone_features_i(train_imgs_i.reshape(-1, *train_imgs_i.shape[-3:]))
        test_feat_i_ori = test_feat_i = self.extract_backbone_features_i(test_imgs_i.reshape(-1, *test_imgs_i.shape[-3:]))
        # rgb_guide_t
        list_weight_train = [None]*16
        list_weight_test = [None]*16
        for idx in range(16):
            list_weight_train[idx] = self.qa_layer3[idx](torch.cat((train_feat_v['layer3'], train_feat_i['layer3']),dim=1))
            list_weight_test[idx] = self.qa_layer3[idx](torch.cat((test_feat_v['layer3'], test_feat_i['layer3']),dim=1))
        # v1
        tensor_weight_train = torch.stack(list_weight_train,1)
        tensor_weight_test = torch.stack(list_weight_test,1)
        train_feat_weight_rgb_guide_t_layer3 = tensor_weight_train.mean(dim=1)
        test_feat_weight_rgb_guide_t_layer3 = tensor_weight_test.mean(dim=1)

        # train_feat_i [:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        train_feat_i = self.rgb_feature_guide_t_feature(train_feat_v_ori, train_feat_i_ori, train_feat_weight_rgb_guide_t_layer3)
        # test_feat_i
        test_feat_i = self.rgb_feature_guide_t_feature(test_feat_v_ori, test_feat_i_ori, test_feat_weight_rgb_guide_t_layer3)
        # train_feat_v
        train_feat_v = self.t_feature_guide_rgb_feature(train_feat_v_ori, train_feat_i_ori, 1-train_feat_weight_rgb_guide_t_layer3)
        # test_feat_v
        test_feat_v = self.t_feature_guide_rgb_feature(test_feat_v_ori, test_feat_i_ori, 1-test_feat_weight_rgb_guide_t_layer3)

        # Classification features
        train_feat_clf_v = self.get_backbone_clf_feat(train_feat_v)
        test_feat_clf_v = self.get_backbone_clf_feat(test_feat_v)
        
        train_feat_clf_i = self.get_backbone_clf_feat(train_feat_i)
        test_feat_clf_i = self.get_backbone_clf_feat(test_feat_i)
        # print('train_feat_clf_v.shape, test_feat_clf_v.shape, train_feat_clf_i.shape, test_feat_clf_i.shape',train_feat_clf_v.shape, test_feat_clf_v.shape, train_feat_clf_i.shape, test_feat_clf_i.shape)
        train_feat_clf = torch.cat((train_feat_clf_v,train_feat_clf_i), dim=1)
        test_feat_clf = torch.cat((test_feat_clf_v,test_feat_clf_i), dim=1)
        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)
        # Get bb_regressor features
        train_feat_iou_v = self.get_backbone_bbreg_feat(train_feat_v) # [], [] prdimp 9 512 36 36 9 1024 18 18
        test_feat_iou_v = self.get_backbone_bbreg_feat(test_feat_v) # [], [] prdimp 9 512 36 36 9 1024 18 18
        train_feat_iou_i = self.get_backbone_bbreg_feat(train_feat_i) # [], [] prdimp 9 512 36 36 3 1024 18 18
        test_feat_iou_i = self.get_backbone_bbreg_feat(test_feat_i) # [] [] prdimp 9 512 36 36 9 1024 18 18
        #pdb.set_trace()
        train_feat_iou = [torch.cat((train_feat_iou_v[idx], train_feat_iou_i[idx]), dim=1) for idx in range(2)] # [] [] prdimp 3 1024 36 36
        test_feat_iou = [torch.cat((test_feat_iou_v[idx], test_feat_iou_i[idx]), dim=1) for idx in range(2)] # [] [] prdimp 9 1024 36 36
        # Run the IoUNet module
        #pdb.set_trace()
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred, tensor_weight_train, tensor_weight_test#train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer3

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features_v(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor_v(im, layers)
    
    def extract_backbone_features_i(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor_i(im, layers)
    
    def rgb_feature_guide_t_feature(self, rgb_feature, t_feature, weight_layer3):
        scale = self.rgbt01_SFT_scale_layer2(rgb_feature['layer2'])
        shift = self.rgbt01_SFT_shift_layer2(rgb_feature['layer2'])
        t_feature['layer2'] = weight_layer3 * (t_feature['layer2'] * scale + shift) + t_feature['layer2']

        scale = self.rgbt01_SFT_scale_layer3(rgb_feature['layer3'])
        shift = self.rgbt01_SFT_shift_layer3(rgb_feature['layer3'])
        t_feature['layer3'] = weight_layer3 * (t_feature['layer3'] * scale + shift) + t_feature['layer3']
        return t_feature
    
    def t_feature_guide_rgb_feature(self, rgb_feature, t_feature, weight_layer3):
        scale = self.rgbt10_SFT_scale_layer2(t_feature['layer2'])
        shift = self.rgbt10_SFT_shift_layer2(t_feature['layer2'])
        rgb_feature['layer2'] = weight_layer3 * (rgb_feature['layer2'] * scale + shift) + rgb_feature['layer2']

        scale = self.rgbt10_SFT_scale_layer3(t_feature['layer3'])
        shift = self.rgbt10_SFT_shift_layer3(t_feature['layer3'])
        rgb_feature['layer3'] = weight_layer3 * (rgb_feature['layer3'] * scale + shift) + rgb_feature['layer3']
        return rgb_feature

    def extract_features(self, im, layers=None): # ONLY used in training stage
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        # RGB: im[:3] TIR: im[3:]
        all_feat1 = self.feature_extractor_v(im[:,:3,...], backbone_layers)
        all_feat2 = self.feature_extractor_i(im[:,3:,...], backbone_layers)
        
        feat_cls = torch.cat((self.extract_classification_feat(all_feat1), self.extract_classification_feat(all_feat2)), dim=1)
        all_feat2['classification'] = self.classifier.extract_classification_feat(feat_cls)

        all_feat2['layer2'] = torch.cat((all_feat1['layer2'], all_feat2['layer2']), dim=1)
        all_feat2['layer3'] = torch.cat((all_feat1['layer3'], all_feat2['layer3']), dim=1)
        
        #all_feat = self.feature_extractor(im, backbone_layers)
        #all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat2[l] for l in layers})


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net_v = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    backbone_net_i = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))
    # rgb guide t
    # layer2
    rgbt01_SFT_scale_layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1), nn.Sigmoid())
    rgbt01_SFT_shift_layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True))
    # layer3
    rgbt01_SFT_scale_layer3 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 1), nn.Sigmoid())
    rgbt01_SFT_shift_layer3 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True))
    # t guide rgb
    # layer2
    rgbt10_SFT_scale_layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1), nn.Sigmoid())
    rgbt10_SFT_shift_layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512, 512, 1, bias=True), nn.ReLU(inplace=True))
    # layer3
    rgbt10_SFT_scale_layer3 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 1), nn.Sigmoid())
    rgbt10_SFT_shift_layer3 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 1, bias=True), nn.ReLU(inplace=True))
    # quality-aware module
    # qa_layer2 = QA_layer(1024)
    qa_layer3 = nn.ModuleList([QA_layer(2048) for _ in range(16)])
    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception
    #print('init clf_feature_extractor')
    clf_feature_extractor = clf_features.residual_bottleneck_concat(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(2*4*128,2*4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    
    # load pretrained model
    usepretrain = updback = updcls = updbb = False
    if usepretrain:
        pretrainmodel_path='/data/liulei/pytracking/ltr/checkpoints/ltr/dimp/dimp50_rgbt/DiMPnet_RGBT_ep0050.pth.tar'
        pretrainmodel = loading.torch_load_legacy(pretrainmodel_path)['net']
        print('pretrained model path', pretrainmodel_path)
        if updback:
            # update backbone
            backbone_dict_v = backbone_net_v.state_dict()
            pretrain_dict_v = {}
            for keys in backbone_dict_v.keys():
                pretrain_dict_v[keys] = pretrainmodel['feature_extractor_v.'+keys]
            backbone_net_v.load_state_dict(pretrain_dict_v)
            backbone_dict_i = backbone_net_i.state_dict()
            pretrain_dict_i = {}
            for keys in backbone_dict_i.keys():
                pretrain_dict_i[keys] = pretrainmodel['feature_extractor_i.'+keys]
            backbone_net_i.load_state_dict(pretrain_dict_i)
        if updcls:
            # update classifier
            classifier_dict = classifier.state_dict()
            pretrain_dict = {k[len('classifier.'):]: v for k, v in pretrainmodel.items() if k[len('classifier.'):] in classifier_dict}
            classifier.load_state_dict(pretrain_dict)
        if updbb:
            # update Bounding box regressor 
            bb_regressor_dict = bb_regressor.state_dict()
            pretrain_dict = {k[len('bb_regressor.'):]: v for k, v in pretrainmodel.items() if k[len('bb_regressor.'):] in bb_regressor_dict}
            bb_regressor.load_state_dict(pretrain_dict)
        print('load pretrained model end!')
    # DiMP network
    net = DiMPnet_RGBT(feature_extractor_v=backbone_net_v, feature_extractor_i=backbone_net_i,  
    rgbt01_SFT_scale_layer2=rgbt01_SFT_scale_layer2, rgbt01_SFT_shift_layer2=rgbt01_SFT_shift_layer2,
    rgbt01_SFT_scale_layer3=rgbt01_SFT_scale_layer3, rgbt01_SFT_shift_layer3=rgbt01_SFT_shift_layer3,
    rgbt10_SFT_scale_layer2=rgbt10_SFT_scale_layer2, rgbt10_SFT_shift_layer2=rgbt10_SFT_shift_layer2,
    rgbt10_SFT_scale_layer3=rgbt10_SFT_scale_layer3, rgbt10_SFT_shift_layer3=rgbt10_SFT_shift_layer3,
    qa_layer3=qa_layer3,
    classifier=classifier, bb_regressor=bb_regressor,classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    # checkpoint_dict_original = loading.torch_load_legacy('/data/liulei/pytracking/ltr/checkpoints/ltr/dimp/dimp50_rgbt/DiMPnet_RGBT_ep0050.pth.tar')
    # for keys in checkpoint_dict_original['net'].keys():
    #     #if checkpoint_dict['net'][keys] == checkpoint_dict_original['net'][keys]:
    #     print(keys, net.state_dict()[keys].cuda() == checkpoint_dict_original['net'][keys].cuda())
    return net