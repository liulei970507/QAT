from . import BaseActor
import torch
import pdb

class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        
        
        if 'qa_layer2' in self.loss_weight.keys() and 'qa_layer3' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer2, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])
        elif 'qa_layer3' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])
        elif 'qa_layer2' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer2 = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])
        else:
            target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])                      
        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        #pdb.set_trace()
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'],test_proposals=data['test_proposals'])
        #print('target_scores.shape, bb_scores.shape', target_scores.shape, bb_scores.shape)
        #pdb.set_trace()
        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats

class DiMPActor_RGBT(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight
        import torch.nn as nn
        self.criterion = nn.BCELoss()
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        # import pdb
        # pdb.set_trace()
        # target_scores, iou_pred = self.net(train_imgs=data['train_images'], test_imgs=data['test_images'], train_bb=data['train_anno'], test_proposals=data['test_proposals'])
        #target_scores, iou_pred = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        if 'qa_layer2' in self.loss_weight.keys() and 'qa_layer3' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer2, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        elif 'qa_layer3' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        elif 'qa_layer2' in self.loss_weight.keys():
            target_scores, iou_pred, train_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer2 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        else:
            target_scores, iou_pred = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])         
        if 'qa_layer2' in self.loss_weight.keys():
            import pdb
            pdb.set_trace()
            loss_qa_weight_layer2_train = self.criterion(train_feat_weight_rgb_guide_t_layer2, data['qa_label_train'].reshape(train_feat_weight_rgb_guide_t_layer2.shape))
            loss_qa_weight_layer2_test = self.criterion(test_feat_weight_rgb_guide_t_layer2, data['qa_label_test'].reshape(test_feat_weight_rgb_guide_t_layer2.shape))
        if 'qa_layer3' in self.loss_weight.keys():
            # import pdb
            # pdb.set_trace()
            loss_qa_weight_layer3_train = self.criterion(train_feat_weight_rgb_guide_t_layer3, data['qa_label_train'].reshape(train_feat_weight_rgb_guide_t_layer3.shape))
            loss_qa_weight_layer3_test = self.criterion(test_feat_weight_rgb_guide_t_layer3, data['qa_label_test'].reshape(test_feat_weight_rgb_guide_t_layer3.shape))

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'][:3], data['test_anno'][:3]) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'][:3])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        # loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf
        # Total loss
        if 'qa_layer2' in self.loss_weight.keys() and 'qa_layer3' in self.loss_weight.keys():
            loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer2_train + loss_qa_weight_layer3_train + loss_qa_weight_layer2_test + loss_qa_weight_layer3_test
        elif 'qa_layer2' in self.loss_weight.keys():
            loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer2_train + loss_qa_weight_layer2_test
        elif 'qa_layer3' in self.loss_weight.keys():
            loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer3_train + loss_qa_weight_layer3_test
        else:
            loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf
        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        if 'qa_layer2' in self.loss_weight.keys():
            stats['Loss/loss_qa_weight_layer2_train'] = loss_qa_weight_layer2_train.item()
            stats['Loss/loss_qa_weight_layer2_test'] = loss_qa_weight_layer2_test.item()
        if 'qa_layer3' in self.loss_weight.keys():
            stats['Loss/loss_qa_weight_layer3_train'] = loss_qa_weight_layer3_train.item()
            stats['Loss/loss_qa_weight_layer3_test'] = loss_qa_weight_layer3_test.item()
        return loss, stats

class KLDiMPActor_RGBT(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        import torch.nn as nn
        self.criterion = nn.BCELoss()
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        #data['train_images'] 6 20 3 352 352
        #data['test_images'] 6 20 3 352 352
        #data['train_anno'] 6 20 4
        #data['test_proposals'] 6 20 128 4
        # if 'qa_layer2' in self.loss_weight.keys():
        #     target_scores, bb_scores, train_feat_weight_rgb_guide_t_layer2, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        # else:
        #     target_scores, bb_scores = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        
        if 'qa_layer2' in self.loss_weight.keys() and 'qa_layer3' in self.loss_weight.keys():
            target_scores, bb_scores, train_feat_weight_rgb_guide_t_layer2, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        elif 'qa_layer3' in self.loss_weight.keys():
            target_scores, bb_scores, train_feat_weight_rgb_guide_t_layer3, test_feat_weight_rgb_guide_t_layer3 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        elif 'qa_layer2' in self.loss_weight.keys():
            target_scores, bb_scores, train_feat_weight_rgb_guide_t_layer2, test_feat_weight_rgb_guide_t_layer2 = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])
        else:
            target_scores, bb_scores = self.net(train_imgs=data['train_images'],test_imgs=data['test_images'],train_bb=data['train_anno'][:3],test_proposals=data['test_proposals'][:3])          

        # import pdb
        # pdb.set_trace()
        # print('self.net.rgbt_transformer_layer2.x_pos_embedding',self.net.rgbt_transformer_layer2.x_pos_embedding)
        #target_scores 6 3 20 23 23
        #bb_scores 3 20 128
        #data['proposal_density'] 6 20 128
        #data['gt_density'] 6 20 128
        # pdb.set_trace()
        # train_feat_weight_rgb_guide_t_layer2 = train_feat_weight_rgb_guide_t_layer2.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        # train_feat_weight_rgb_guide_t_layer3 = train_feat_weight_rgb_guide_t_layer3.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        # loss_qa_weight_layer2_train = self.criterion(train_feat_weight_rgb_guide_t_layer2, data['qa_label_train'].view(-1))
        # loss_qa_weight_layer3_train = self.criterion(train_feat_weight_rgb_guide_t_layer3, data['qa_label_train'].view(-1))
        if 'qa_layer2' in self.loss_weight.keys():
            # import pdb
            # pdb.set_trace()
            loss_qa_weight_layer2_train = self.criterion(train_feat_weight_rgb_guide_t_layer2, data['qa_label_train'].reshape(train_feat_weight_rgb_guide_t_layer2.shape))
            loss_qa_weight_layer2_test = self.criterion(test_feat_weight_rgb_guide_t_layer2, data['qa_label_test'].reshape(test_feat_weight_rgb_guide_t_layer2.shape))
        if 'qa_layer3' in self.loss_weight.keys():
            loss_qa_weight_layer3_train = self.criterion(train_feat_weight_rgb_guide_t_layer3, data['qa_label_train'].reshape(train_feat_weight_rgb_guide_t_layer3.shape))
            loss_qa_weight_layer3_test = self.criterion(test_feat_weight_rgb_guide_t_layer3, data['qa_label_test'].reshape(test_feat_weight_rgb_guide_t_layer3.shape))
        # Reshape bb reg variables
        is_valid = data['test_anno'][:3][:, :, 0] < 99999.0
        bb_scores = bb_scores[:3][is_valid, :]
        proposal_density = data['proposal_density'][:3][is_valid, :]
        gt_density = data['gt_density'][:3][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'][:3], data['test_anno'][:3]) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'][:3], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        if 'qa_layer2' in self.loss_weight.keys() and 'qa_layer3' in self.loss_weight.keys():
            loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                                loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer2_train + loss_qa_weight_layer3_train + loss_qa_weight_layer2_test + loss_qa_weight_layer3_test
        elif 'qa_layer2' in self.loss_weight.keys():
            loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                                loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer2_train + loss_qa_weight_layer2_test
        elif 'qa_layer3' in self.loss_weight.keys():
            loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                                loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_qa_weight_layer3_train + loss_qa_weight_layer3_test
        else:
            loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                                loss_target_classifier + loss_test_init_clf + loss_test_iter_clf
        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)
        if 'qa_layer2' in self.loss_weight.keys():
            stats['Loss/loss_qa_weight_layer2_train'] = loss_qa_weight_layer2_train.item()
            stats['Loss/loss_qa_weight_layer2_test'] = loss_qa_weight_layer2_test.item()
        if 'qa_layer3' in self.loss_weight.keys():
            stats['Loss/loss_qa_weight_layer3_train'] = loss_qa_weight_layer3_train.item()
            stats['Loss/loss_qa_weight_layer3_test'] = loss_qa_weight_layer3_test.item()
        return loss, stats