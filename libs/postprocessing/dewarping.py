import cv2
import numpy as np
import scipy.spatial.qhull as qhull

K = 0


def create_mapping(perturbed_img,  perturbed_label, perturbed_label_classify):
    flat_shape = perturbed_img.shape[:2]
    perturbed_label_classify = cv2.resize(perturbed_label_classify.astype(np.float32), (flat_shape[1], flat_shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
    perturbed_label_classify[perturbed_label_classify <= 0.5] = 0
    perturbed_label_classify = np.array(perturbed_label_classify).astype(np.uint8)
    flat_img = np.full_like(perturbed_img, 256, dtype=np.uint16)

    '''remove the background of input image(perturbed_img) and forward mapping(perturbed_label)'''
    origin_pixel_position = np.argwhere(np.zeros(flat_shape, dtype=np.uint32) == 0).reshape(
        flat_shape[0] * flat_shape[1], 2)
    perturbed_label = perturbed_label.reshape(flat_shape[0] * flat_shape[1], 2)
    perturbed_img = perturbed_img.reshape(flat_shape[0] * flat_shape[1], 3)
    perturbed_label_classify = perturbed_label_classify.reshape(flat_shape[0] * flat_shape[1])
    perturbed_label[perturbed_label_classify != 0, :] += origin_pixel_position[perturbed_label_classify != 0, :]
    pixel_position = perturbed_label[perturbed_label_classify != 0, :]
    pixel = perturbed_img[perturbed_label_classify != 0, :]
    '''construct Delaunay triangulations in all scattered pixels and then using interpolation'''
    vtx, wts = interp_weights(pixel_position, origin_pixel_position)
    # wts[np.abs(wts)>=1]=0
    wts_sum = np.abs(wts).sum(-1)
    wts = wts[wts_sum <= 1, :]
    vtx = vtx[wts_sum <= 1, :]
    flat_img.reshape(flat_shape[0] * flat_shape[1], 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
    flat_img = flat_img.reshape(flat_shape[0], flat_shape[1], 3)

    flat_img = flat_img.astype(np.uint8)
    flat_img_crop = flat_img
    return flat_img_crop


def interpolate(values, vtx, wts):
    return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)


def interp_weights(xyz, uvw):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    # pixel_triangle = pixel[tri.simplices]
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))