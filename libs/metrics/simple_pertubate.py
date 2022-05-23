import torch


def create_mapping(perturbed_img,  perturbed_label, perturbed_label_classify):
    N, C, H, W = perturbed_img.shape

    flat_shape = H, W
    perturbed_label_classify = torch.sigmoid(perturbed_label_classify)
    perturbed_label_classify[perturbed_label_classify <= 0.5] = 0
    perturbed_label_classify = perturbed_label_classify.byte()
    flat_img = torch.full_like(perturbed_img, 256, dtype=torch.short)

    '''remove the background of input image(perturbed_img) and forward mapping(perturbed_label)'''
    origin_pixel_position = torch.argwhere(torch.zeros(H, W, dtype=torch.int) == 0).reshape(H * W, 2)

    perturbed_label = perturbed_label.reshape(N, 2, flat_shape[0] * flat_shape[1]).permute(0, 2, 1)
    perturbed_img = perturbed_img.reshape(N, 3, flat_shape[0] * flat_shape[1])
    perturbed_label_classify = perturbed_label_classify.reshape(N, flat_shape[0] * flat_shape[1])
    perturbed_label[perturbed_label_classify != 0, :] += origin_pixel_position[perturbed_label_classify != 0, :]
    pixel_position = perturbed_label[perturbed_label_classify != 0, :]
    pixel = perturbed_img[perturbed_label_classify != 0, :]

    result = torch.nn.functional.grid_sample(pixel, pixel_position)

    return result