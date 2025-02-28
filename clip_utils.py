import gc

import torch
# from SCLIP import clip
import clip
import mmcv
import numpy as np

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)

cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


def patch_classification(top_k, patch_padding, image, masked_img, ann, class_list, class_list_extended, w, h,
                         clip_model, name_renovation,
                         return_patches=False,
                         global_feature=None):
    patch = mmcv.imcrop(image, np.array(
        [max(ann['bbox'][0] - patch_padding, 0),
         max(ann['bbox'][1] - patch_padding, 0),
         min(ann['bbox'][0] + ann['bbox'][2] + patch_padding, w),
         min(ann['bbox'][1] + ann['bbox'][3] + patch_padding, h)]),
                        scale=1.0)
    patch = mmcv.imresize(patch, (224, 224))
    if masked_img is not None:
        regional_mask = mmcv.imcrop(masked_img.cpu().numpy() * 1.0, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]))
        regional_mask = mmcv.imresize(regional_mask.astype('float32'), (224, 224))
        patch = patch * np.expand_dims(regional_mask, axis=2)
    mask_categories, probs = sclip_classification_with_probs(patch, class_list, class_list_extended, top_k, clip_model,
                                                             name_renovation, global_feature)
    mask_categories = [mask_categories] if isinstance(mask_categories, str) else mask_categories
    probs = probs.tolist()
    probs = probs if isinstance(probs, list) else [probs]
    # second_index = probs.index(max(probs))
    # second_large_category = mask_categories[second_index]
    if return_patches:
        return mask_categories, probs, patch
    return mask_categories, torch.as_tensor(probs)


def sclip_classification_with_probs(image, class_list, class_list_extended, top_k, clip_model, name_renovation=False,
                                    global_feature=None):
    # extract image feature
    img = image
    img = img[:, :, ::-1].copy()
    img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
    img = (img / 255.0 - pixel_mean) / pixel_std

    # extract text feature
    if not name_renovation:
        txts = [f'a photo of {cls_name}' for cls_name in class_list]
        text = clip.tokenize(txts)
        with torch.no_grad():
            text_features = clip_model.encode_text(text.cuda())
            # print("+++++++++++++++++", text_features.shape)  #[n, 768, n for class nums]
            text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = []
        for cls_names in class_list_extended:
            txts = [f'a photo of {cls_name}' for cls_name in cls_names]
            text = clip.tokenize(txts)
            with torch.no_grad():
                text_features_tmp = clip_model.encode_text(text.cuda())
                # print("+++++++++++++++++", len(text_features_tmp), text_features_tmp[0].shape)
                # get the mean of text features
                text_features_tmp = text_features_tmp.mean(dim=0)
                text_features_tmp /= text_features_tmp.norm(dim=-1, keepdim=True)
                text_features_tmp = torch.unsqueeze(text_features_tmp, 0)
                text_features.append(text_features_tmp)
                # print("===========tmp", text_features_tmp.shape)  # [1, 512]
        del text_features_tmp
        gc.collect()
        text_features = torch.cat(text_features, dim=0)
        # print("3333333+++++++++++++++++", text_features.shape)
        # print("===========all", text_features.shape)  # [n, 512]

    # image_features = clip_model.encode_image(img.cuda(), return_all=True, csa=True)
    image_features = clip_model.encode_image(img.cuda())
    if global_feature is not None:
        f = torch.zeros_like(image_features)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        for g in global_feature:
            sim_score = cosine_similarity(g, image_features)
            f += sim_score * g + (1 - sim_score) * image_features
        image_features = f / len(global_feature)
    else:
        image_features /= image_features.norm(dim=-1, keepdim=True)

    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # print("probs--------------", probs.shape, probs)

    del image_features, text_features
    gc.collect()

    if top_k == 1:
        class_name = class_list[probs.argmax().item()]
        return class_name, torch.max(probs)
    elif top_k == 0:
        return class_list, probs
    else:
        top_k_indices = probs.topk(top_k, dim=1).indices[0]
        top_k_class_names = [class_list[index] for index in top_k_indices]
        top_k_class_probs = probs.topk(top_k, dim=1).values[0]
        # print(top_k_class_names, top_k_class_probs)
        return top_k_class_names, top_k_class_probs


def sclip_logits(image, ann, masked_img=None, patch_padding=0, w=0, h=0, class_list=None, class_list_extended=None,
                 clip_model=None,
                 name_renovation=False):
    img = mmcv.imcrop(image, np.array(
        [max(ann['bbox'][0] - patch_padding, 0),
         max(ann['bbox'][1] - patch_padding, 0),
         min(ann['bbox'][0] + ann['bbox'][2] + patch_padding, w),
         min(ann['bbox'][1] + ann['bbox'][3] + patch_padding, h)]),
                      scale=1.0)
    img = mmcv.imresize(img, (224, 224))
    regional_mask = mmcv.imcrop(masked_img.cpu().numpy() * 1.0, np.array(
        [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]))
    regional_mask = mmcv.imresize(regional_mask.astype('float32'), (224, 224))
    # patch = patch * np.expand_dims(regional_mask, axis=2)
    regional_mask = torch.from_numpy(regional_mask).to('cuda')
    # img = image
    img = img[:, :, ::-1].copy()
    img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
    img = (img / 255.0 - pixel_mean) / pixel_std

    # extract text feature
    if not name_renovation:
        txts = [f'a photo of {cls_name}' for cls_name in class_list]
        text = clip.tokenize(txts)
        with torch.no_grad():
            text_features = clip_model.encode_text(text.cuda())
            # print("+++++++++++++++++", text_features.shape)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = []
        for cls_names in class_list_extended:
            txts = [f'a photo of {cls_name}' for cls_name in cls_names]
            text = clip.tokenize(txts)
            with torch.no_grad():
                text_features_tmp = clip_model.encode_text(text.cuda())
                # get the mean of text features
                text_features_tmp = text_features_tmp.mean(dim=0)
                text_features_tmp /= text_features_tmp.norm(dim=-1, keepdim=True)
                text_features_tmp = torch.unsqueeze(text_features_tmp, 0)
                text_features.append(text_features_tmp)
                # print("===========tmp", text_features_tmp.shape)  # [1, 512]
        text_features = torch.cat(text_features, dim=0)
        # print("===========all", text_features.shape)  # [n, 512]

    # extract image_features
    with torch.no_grad():
        image_features = clip_model.encode_image(img.cuda(), return_all=True, csa=True)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features = image_features[:, 1:]
    logits = image_features @ text_features.T
    # print(logits.shape)  # [1, 197, 2]

    # probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # upsample logits
    patch_size = clip_model.visual.patch_size
    w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
    out_dim = logits.shape[-1]
    # print("===========", w, h, out_dim)  # 14, 14, 2
    logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
    # print("=========logits", logits.shape)  # [1, 2, 14, 14]
    logits = torch.nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
    # print("=========logits", logits.shape)  # # [1, 2, 224, 224]
    # print("========", torch.min(logits[0]), torch.max(logits[0]))
    # tensor(0.1675, device='cuda:0', dtype=torch.float16)
    # tensor(0.2844, device='cuda:0', dtype=torch.float16)
    logits = logits[0] * 40.0
    logits = logits.softmax(0)  # n_queries * w * h
    # print(torch.min(logits), torch.max(logits))
    # tensor(0.0795, device='cuda:0', dtype=torch.float16)
    # tensor(0.9204, device='cuda:0', dtype=torch.float16)
    # print("=========", logits.shape, regional_mask.shape)  # torch.Size([2, 224, 224]) torch.Size([224, 224])
    masked_logits = logits * regional_mask
    # print("=========", masked_logits.shape)  # torch.Size([2, 224, 224])
    # print(torch.min(masked_logits), torch.max(masked_logits))  # (0, 0.9111)
    probs = torch.sum(masked_logits, dim=(1, 2)) / torch.count_nonzero(regional_mask).item()
    # print(probs, torch.count_nonzero(regional_mask).item())

    return class_list, probs.detach().cpu()


def sclip_get_img_features(global_image, ann=None,
                           patch_padding=0, random_padding=False,
                           w=None, h=None, clip_model=None, norm=True,
                           return_all=False, csa=False):
    # extract image feature
    if random_padding:
        f_ = []
        off_set = 20
        for p_w1, p_h1, p_w2, p_h2 in zip([off_set, patch_padding, off_set, patch_padding, patch_padding, off_set],
                                          [off_set, patch_padding, off_set, patch_padding, off_set, patch_padding],
                                          [off_set, patch_padding, patch_padding, off_set, off_set, patch_padding],
                                          [off_set, patch_padding, patch_padding, off_set, patch_padding, off_set]):
            patch_image = mmcv.imcrop(global_image, np.array(
                [max(ann['bbox'][0] - p_w1, 0),
                 max(ann['bbox'][1] - p_h1, 0),
                 min(ann['bbox'][0] + ann['bbox'][2] + p_w2, w),
                 min(ann['bbox'][1] + ann['bbox'][3] + p_h2, h)]),
                                      scale=1.0)
            patch_image = mmcv.imresize(patch_image.astype('float32'), (224, 224))
            img = patch_image
            img = img[:, :, ::-1].copy()
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
            img = (img / 255.0 - pixel_mean) / pixel_std
            image_features = clip_model.encode_image(img.cuda())
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            # TODO
            # image_features = torch.nn.functional.normalize(image_features, dim=-1)
            f_.append(image_features)
        f_ = torch.cat(f_, 0)
        # print(image_features.shape, f_.shape)  # torch.Size([1, 512]) torch.Size([5, 512])
        image_features = torch.mean(f_, 0, True)
        # print(image_features.shape)  # torch.Size([1, 512])

    else:
        if ann is None:
            img = global_image
        else:
            patch_image = mmcv.imcrop(global_image, np.array(
                [max(ann['bbox'][0] - patch_padding, 0),
                 max(ann['bbox'][1] - patch_padding, 0),
                 min(ann['bbox'][0] + ann['bbox'][2] + patch_padding, w),
                 min(ann['bbox'][1] + ann['bbox'][3] + patch_padding, h)]),
                                      scale=1.0)
            patch_image = mmcv.imresize(patch_image.astype('float32'), (224, 224))
            img = patch_image
        img = img[:, :, ::-1].copy()
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        img = (img / 255.0 - pixel_mean) / pixel_std
        image_features = clip_model.encode_image(img.cuda(), return_all=return_all, csa=csa)
        if norm:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        # TODO
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    # print(image_features.shape)  # torch.Size([1, 512])
    return image_features


def sclip_get_global_features(img, ann, w, h, patch_padding, clip_model,
                              return_all=False, csa=False, random_padding=False, ):
    feature_G = []
    feature_G1 = sclip_get_img_features(img, clip_model=clip_model, ann=ann,
                                        # patch_padding=200,
                                        patch_padding=patch_padding,
                                        random_padding=random_padding,
                                        w=w, h=h,
                                        return_all=return_all, csa=csa)
    feature_G2 = sclip_get_img_features(img, clip_model=clip_model, ann=ann,
                                        # patch_padding=200,
                                        patch_padding=patch_padding + 200,
                                        random_padding=random_padding,
                                        w=w, h=h,
                                        return_all=return_all, csa=csa)
    feature_G.append(feature_G1)
    feature_G.append(feature_G2)
    return feature_G
